"""
Reconstruct a displacement series U(t, y, x) from overlapping space-time denoiser outputs.

Model:
P_{k,w}(tau, x) = [U(t_k+tau, x) - U(t_k, x)] - [f_w(t_k+tau, x) - f_w(t_k, x)] + noise
with one unknown field per spatial tile w and epoch t (f_w(0) = 0):
  - offsets only (default):  f_w = a_w(t)
  - offsets + tilt (solve_tilt=True):  f_w = a_w(t) + gy_w(t) * eta_y + gx_w(t) * eta_x,
    eta = (pixel - tile centre) / (N/2) in [-1, 1] (gy, gx in data units per half tile).
1. pass 1 (network): per chunk, accumulate the uncorrected field U_base and the overlap statistics
   D_{k,(i,j)}(tau): median of (P_j - P_i) over the overlap (offsets), or the robust plane fitted to (P_j - P_i)
   over the overlap (offsets + tilt: value at the overlap centre and the two slopes)
2. one linear solve, separable in space and time, for the tile fields (optionally constrained by the velocity)
3. U = U_base + Interp(tile fields)                      (exact, because blend weights are static)
4. optionally 'after': low-order velocity detrending of U
5. optionally fit_plane: remove the best-fit plane (gradients only) of every epoch of the final U

velocity_mode: 'none'   -> no velocity information used
               'before' -> low-order velocity constraint inside the offset solve (not available with solve_tilt)
               'after'  -> low-order velocity detrending of the final series
Without an outside constraint the overlaps cannot determine the global plane of each epoch (a systematic tilt of all
tiles is absorbed exactly); with solve_tilt this becomes an explicit gauge (min-norm solution) that velocity_mode='after'
and/or fit_plane then fix.
Units: `velocity` must be in cube-units per year (cube is in cm in flatsim_denoising.py).
"""
import logging
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from utils.profiling import prof

log = logging.getLogger(__name__).info


# ----------------------------------------------------------------------------- inputs
class LazyWindows:
    """Array-like (W, Nt, N, N, 1); crops one window per access."""

    def __init__(self, cube, origins, N, t0, Nt):
        self.cube, self.origins, self.N, self.t0, self.Nt = cube, origins, N, t0, Nt
        self.shape = (len(origins), Nt, N, N, 1)
        self.dtype = np.dtype(np.float32)
        self.ndim = 5

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            y, x = self.origins[i]
            return self.cube[self.t0:self.t0 + self.Nt, y:y + self.N, x:x + self.N, None].copy()
        idx = np.arange(len(self))[i]
        return np.stack([self[int(j)] for j in idx])

    def __array__(self, dtype=None, copy=None):
        raise RuntimeError('LazyWindows must not be materialized')


def yyyymmdd_to_years(dates):
    """Array of YYYYMMDD (int/str/bytes) -> decimal years."""
    out = []
    for d in np.asarray(dates):
        d = d.decode() if isinstance(d, bytes) else str(int(d)) if not isinstance(d, str) else d
        t = np.datetime64(f'{d[:4]}-{d[4:6]}-{d[6:8]}')
        y0 = np.datetime64(f'{d[:4]}-01-01')
        y1 = np.datetime64(f'{int(d[:4]) + 1}-01-01')
        out.append(int(d[:4]) + (t - y0) / (y1 - y0))
    return np.array(out, float)


def fill_cube_inplace(cube, fill_range=10., seed=0):
    rng = np.random.default_rng(seed)
    for t in range(cube.shape[0]):
        nan = np.isnan(cube[t])
        n = int(nan.sum())
        if n:
            cube[t][nan] = rng.uniform(-fill_range, fill_range, n).astype(np.float32)


def nan_all(cube):
    m = np.ones(cube.shape[1:], bool)
    for t in range(cube.shape[0]):
        m &= np.isnan(cube[t])
    return m


# ----------------------------------------------------------------------------- geometry
def _starts(L, N, stride):
    if L < N:
        raise ValueError(f'Dimension {L} smaller than window size {N}')
    s = list(range(0, L - N + 1, stride))
    if s[-1] != L - N:
        s.append(L - N)
    return np.array(s)


def temporal_weights(Nt):
    """Hann weights for tau = 1..Nt-1."""
    return np.hanning(Nt + 2)[1:-1][1:]


def temporal_matrix(T, Nt, t_starts, w, ridge=1e-9):
    """M = sum_k B_k^T W B_k, rows: u(t_k+tau) - u(t_k), unknowns u(1..T-1), u(0)=0."""
    M = np.zeros((T - 1, T - 1))
    for tk in t_starts:
        for tau in range(1, Nt):
            r = np.zeros(T - 1)
            r[tk + tau - 1] += 1.
            if tk > 0:
                r[tk - 1] -= 1.
            M += w[tau - 1] * np.outer(r, r)
    return M + ridge * np.eye(T - 1)


def accumulate(rhs, Y, tk, w):
    """rhs += B_k^T W Y_k. Y: (Nt, ...), rhs: (T-1, ...)."""
    for tau in range(1, Y.shape[0]):
        val = np.nan_to_num(Y[tau]) * w[tau - 1]
        rhs[tk + tau - 1] += val
        if tk > 0:
            rhs[tk - 1] -= val


def prepare_static(topo, mask, N, spatial_stride, edge_trim=8, min_overlap_frac=0.25,
                   min_pixels=200, max_lag=None):
    """Everything that does not depend on the chunk: tiles, blend weights, overlap pairs, topo."""
    ny, nx = mask.shape
    assert topo.shape == (ny, nx)
    ys, xs = _starts(ny, N, spatial_stride), _starts(nx, N, spatial_stride)
    origins = np.array([(y, x) for y in ys for x in xs])
    gidx = np.array([(iy, ix) for iy in range(len(ys)) for ix in range(len(xs))])
    valid = ~mask.astype(bool)
    wvalid = np.stack([valid[y:y + N, x:x + N] for y, x in origins])
    keep = np.flatnonzero(wvalid.any(axis=(1, 2)))
    origins, gidx, wvalid = origins[keep], gidx[keep], wvalid[keep]
    W = len(origins)

    h = np.hanning(N + 2)[1:-1]
    wt = (np.outer(h, h)[None] * wvalid).astype(np.float64)
    den = np.zeros((ny, nx))
    for k, (y, x) in enumerate(origins):
        den[y:y + N, x:x + N] += wt[k]
    hhat = np.empty((W, N, N), np.float32)
    for k, (y, x) in enumerate(origins):
        d = den[y:y + N, x:x + N]
        hhat[k] = np.where(d > 0, wt[k] / np.where(d > 0, d, 1.), 0.)

    topo = topo.astype(np.float32)
    tw = np.empty((W, 1, N, N, 1), np.float32)
    for k, (y, x) in enumerate(origins):
        t = topo[y:y + N, x:x + N]
        mn = np.nanmin(t) if np.isfinite(t).any() else 0.
        rg = (np.nanmax(t) - mn) if np.isfinite(t).any() else 0.
        tw[k, 0, :, :, 0] = np.nan_to_num((t - mn) / (rg if rg > 0 else 1.))

    if max_lag is None:
        s = spatial_stride if np.isscalar(spatial_stride) else min(spatial_stride)
        max_lag = int(np.clip(np.ceil(0.5 * N / s), 1, 6))
    lo, hi = edge_trim, N - edge_trim
    full_area = (hi - lo) ** 2
    lookup = {tuple(g): k for k, g in enumerate(gidx.tolist())}
    offs = [(dy, dx) for dy in range(0, max_lag + 1) for dx in range(-max_lag, max_lag + 1)
            if not (dy == 0 and dx <= 0)]
    pairs = []
    for i, (iy, ix) in enumerate(gidx.tolist()):
        yi, xi = origins[i]
        for dy, dx in offs:
            j = lookup.get((iy + dy, ix + dx))
            if j is None:
                continue
            yj, xj = origins[j]
            y0, y1 = max(yi, yj) + lo, min(yi, yj) + hi
            x0, x1 = max(xi, xj) + lo, min(xi, xj) + hi
            if y1 <= y0 or x1 <= x0 or (y1 - y0) * (x1 - x0) < min_overlap_frac * full_area:
                continue
            ai = (y0 - yi, y1 - yi, x0 - xi, x1 - xi)
            aj = (y0 - yj, y1 - yj, x0 - xj, x1 - xj)
            ok = wvalid[i][ai[0]:ai[1], ai[2]:ai[3]] & wvalid[j][aj[0]:aj[1], aj[2]:aj[3]]
            sel = np.flatnonzero(ok.ravel())
            if len(sel) < min_pixels:
                continue
            pairs.append((i, j, ai, aj, sel))
    if not pairs:
        raise RuntimeError('No valid overlapping pairs (check stride / min_pixels / mask).')
    return dict(origins=origins, gidx=gidx, wvalid=wvalid, hhat=hhat, den=den, topo_w=tw,
                pairs=pairs, N=N, shape=(ny, nx), max_lag=max_lag)


# ----------------------------------------------------------------------------- pass 1
def predict_chunk(prep, filled, t0, Nt, engine, dataset_cls, batch_windows=2000, verbose=True):
    o, N = prep['origins'], prep['N']
    W = len(o)
    P = np.empty((W, Nt, N, N), np.float32)
    for s in range(0, W, batch_windows):
        e = min(s + batch_windows, W)
        ds = dataset_cls(x=LazyWindows(filled, o[s:e], N, t0, Nt), topo=prep['topo_w'][s:e])
        with prof.section('predict.net'):
            out = np.asarray(engine.predict(data=ds, verbose=verbose))
        out = out[..., 0] if out.ndim == 5 else out
        P[s:e] = out
    wv = prep['wvalid']
    with prof.section('predict.post'):
        for k in range(W):
            P[k][:, ~wv[k]] = np.nan
        P -= P[:, :1].copy()                      # increments relative to the first frame
    return P


def overlap_medians(prep, P):
    """D[p, tau-1] = median over the overlap of (P_j - P_i)(tau)."""
    Nt = P.shape[1]
    D = np.zeros((len(prep['pairs']), Nt - 1))
    for p, (i, j, ai, aj, sel) in enumerate(prep['pairs']):
        a = P[i][:, ai[0]:ai[1], ai[2]:ai[3]].reshape(Nt, -1)[:, sel]
        b = P[j][:, aj[0]:aj[1], aj[2]:aj[3]].reshape(Nt, -1)[:, sel]
        D[p] = np.nanmedian(b - a, axis=1)[1:]
    return D


def blend(prep, P):
    Nt = P.shape[1]
    H, Wd = prep['shape']
    N = prep['N']
    Y = np.zeros((Nt, H, Wd), np.float32)
    for k, (y, x) in enumerate(prep['origins']):
        Y[:, y:y + N, x:x + N] += np.nan_to_num(P[k]) * prep['hhat'][k][None]
    return Y


def interp_offsets(prep, E, out):
    """out[1:] += sum_w hhat_w(x) E_w(t)."""
    N = prep['N']
    for k, (y, x) in enumerate(prep['origins']):
        out[1:, y:y + N, x:x + N] += E[k][:, None, None].astype(np.float32) * prep['hhat'][k][None]


# ----------------------------------------------------------------------------- offset solve
def pair_incidence(prep):
    pr = prep['pairs']
    P_, W = len(pr), len(prep['origins'])
    rows = np.repeat(np.arange(P_), 2)
    cols = np.array([[i, j] for i, j, *_ in pr]).ravel()
    vals = np.tile([1., -1.], P_)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(P_, W))
    n = np.array([len(p[4]) for p in pr], float)
    return A, n / n.mean()


def factor_system(A, a, M, tol=1e-10):
    """Eigen-decompositions of L = A^T diag(a) A and M; pseudo-inverses (zero modes = gauge)."""
    L = (A.T @ sp.diags(a) @ A).toarray()
    ls, Vs = np.linalg.eigh(L)
    keep_s = ls > tol * ls.max()
    lt, Vt = np.linalg.eigh(M)
    return dict(ls=ls, Vs=Vs, keep_s=keep_s, lt=lt, Vt=Vt,
                Lp=(Vs[:, keep_s] / ls[keep_s]) @ Vs[:, keep_s].T,
                Minv=(Vt / lt) @ Vt.T, n_null=int((~keep_s).sum()))


def solve_offsets(A, a, R, fs):
    """Solve L E M = A^T diag(a) R (E: W x (T-1)); zero-mean gauge per connected component."""
    C = A.T @ (a[:, None] * R)
    Z = fs['Vs'][:, fs['keep_s']].T @ C @ fs['Vt']
    Z = Z / (fs['ls'][fs['keep_s']][:, None] * fs['lt'][None, :])
    return fs['Vs'][:, fs['keep_s']] @ Z @ fs['Vt'].T


def offset_residual(A, E, D_all, t_starts, Nt, a, w):
    """Weighted rms of D_{k,p}(tau) - model, and of D itself."""
    G = np.concatenate([np.zeros((A.shape[0], 1)), A @ E], axis=1)      # (P, T)
    tk = np.asarray(t_starts)[:, None]
    tt = tk + np.arange(1, Nt)[None]
    Dhat = (G[:, tt] - G[:, tk][:, :, :])                                # (P, K, Nt-1)
    Dhat = np.transpose(Dhat, (1, 0, 2))                                 # (K, P, Nt-1)
    ww = a[None, :, None] * w[None, None, :]
    ww = np.broadcast_to(ww, D_all.shape)
    return (float(np.sqrt((ww * D_all ** 2).sum() / ww.sum())),
            float(np.sqrt((ww * (D_all - Dhat) ** 2).sum() / ww.sum())))


# ----------------------------------------------------------------------------- offsets + tilt
def pair_geometry(prep):
    """Static per-pair quantities for the tilt model: plane basis over the overlap pixels (coordinates relative to the
    overlap centre, in units of N/2) and the offsets of the overlap centre from the two tile centres."""
    N = prep['N']
    h = N / 2.0
    origins = prep['origins']
    geo = []
    for (i, j, ai, aj, sel) in prep['pairs']:
        yi, xi = origins[i]
        yj, xj = origins[j]
        ww = ai[3] - ai[2]
        rows = yi + ai[0] + sel // ww
        cols = xi + ai[2] + sel % ww
        yo, xo = rows.mean(), cols.mean()
        v, u = (rows - yo) / h, (cols - xo) / h
        Psi = np.stack([np.ones(len(sel)), v, u], axis=1)
        geo.append(dict(Psi=Psi, G=np.linalg.pinv(Psi),
                        d=((yo - (yi + (N - 1) / 2)) / h, (xo - (xi + (N - 1) / 2)) / h,
                           (yo - (yj + (N - 1) / 2)) / h, (xo - (xj + (N - 1) / 2)) / h),
                        var=(float(v.var()), float(u.var()))))
    return geo


def pair_incidence_tilt(prep, geo, a):
    """A3 (3P x 3W): rows (c0, cy, cx) of every pair, columns (a, gy, gx) of every tile; ws: row weights.
    c0 = value of the overlap difference at the overlap centre, cy/cx = its slopes (per N/2)."""
    pairs = prep['pairs']
    P_, W = len(pairs), len(prep['origins'])
    rows, cols, vals = [], [], []
    ws = np.zeros(3 * P_)
    for p, ((i, j, *_), g) in enumerate(zip(pairs, geo)):
        dyi, dxi, dyj, dxj = g['d']
        r = 3 * p
        rows += [r] * 6
        cols += [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        vals += [1., dyi, dxi, -1., -dyj, -dxj]
        rows += [r + 1, r + 1, r + 2, r + 2]
        cols += [3 * i + 1, 3 * j + 1, 3 * i + 2, 3 * j + 2]
        vals += [1., -1., 1., -1.]
        ws[r], ws[r + 1], ws[r + 2] = a[p], a[p] * max(g['var'][0], 1e-3), a[p] * max(g['var'][1], 1e-3)
    return sp.csr_matrix((vals, (rows, cols)), shape=(3 * P_, 3 * W)), ws


def overlap_planes(prep, geo, P, clip=3.0):
    """D3[p, :, tau-1] = (c0, cy, cx): plane fitted to (P_j - P_i) over the overlap (one 3-sigma clipping pass)."""
    Nt = P.shape[1]
    out = np.zeros((len(prep['pairs']), 3, Nt - 1))
    for p, ((i, j, ai, aj, sel), g) in enumerate(zip(prep['pairs'], geo)):
        a = P[i][:, ai[0]:ai[1], ai[2]:ai[3]].reshape(Nt, -1)[:, sel]
        b = P[j][:, aj[0]:aj[1], aj[2]:aj[3]].reshape(Nt, -1)[:, sel]
        d = np.nan_to_num((b - a)[1:].T.astype(np.float64))                  # (n, Nt-1)
        coef = g['G'] @ d
        if clip:
            r = d - g['Psi'] @ coef
            s = 1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0) + 1e-12
            wgt = (np.abs(r) <= clip * s).astype(np.float64)
            Mt = np.einsum('np,nt,nq->tpq', g['Psi'], wgt, g['Psi'], optimize=True)
            bt = np.einsum('np,nt,nt->tp', g['Psi'], wgt, d, optimize=True)
            try:
                coef = np.linalg.solve(Mt, bt[..., None])[..., 0].T
            except np.linalg.LinAlgError:
                pass
        out[p] = coef
    return out


def factor_tilt(A3, ws, M, prior=0.0, eps_rel=1e-8):
    """Sparse LU of L3 = A3^T diag(ws) A3 (+ tiny ridge = min-norm gauge, + optional ridge on the tilts) and the
    eigendecomposition of M. In the eigenbasis of M the space-time system decouples into one L3 system per mode."""
    L3 = (A3.T @ sp.diags(ws) @ A3).tocsc()
    n3 = L3.shape[0]
    d = L3.diagonal()
    if prior > 0:
        tilt = (np.arange(n3) % 3 != 0).astype(float)
        L3 = L3 + sp.diags(prior * d[tilt == 0].mean() * tilt)
    lu = splu((L3 + eps_rel * d.max() * sp.eye(n3)).tocsc())
    lt, Vt = np.linalg.eigh(M)
    return dict(lu=lu, lt=lt, Vt=Vt)


def solve_tilt_system(A3, ws, R3, ft):
    """Theta (3W x (T-1)) from  L3 Theta M = A3^T diag(ws) R3; R3: (3P, T-1) accumulated observations."""
    C = A3.T @ (ws[:, None] * R3)
    Th = ft['lu'].solve(np.ascontiguousarray(C @ ft['Vt'])) / ft['lt'][None, :]
    return Th @ ft['Vt'].T


def tilt_residual(A3, Theta, D3_all, t_starts, Nt, ws, w):
    """Weighted rms of the 3-row observations and of their residual after the fit; D3_all: (K, 3P, Nt-1)."""
    G = np.concatenate([np.zeros((A3.shape[0], 1)), A3 @ Theta], axis=1)
    tk = np.asarray(t_starts)[:, None]
    tt = tk + np.arange(1, Nt)[None]
    Dhat = np.transpose(G[:, tt] - G[:, tk], (1, 0, 2))
    ww = np.broadcast_to(ws[None, :, None] * w[None, None, :], D3_all.shape)
    return (float(np.sqrt((ww * D3_all ** 2).sum() / ww.sum())),
            float(np.sqrt((ww * (D3_all - Dhat) ** 2).sum() / ww.sum())))


def interp_offsets_tilt(prep, Theta, out):
    """out[1:] += sum_w hhat_w(x) [a_w(t) + gy_w(t) eta_y + gx_w(t) eta_x]; Theta: (W, 3, T-1)."""
    N = prep['N']
    eta = ((np.arange(N) - (N - 1) / 2) / (N / 2)).astype(np.float32)
    for k, (y, x) in enumerate(prep['origins']):
        a, gy, gx = (Theta[k, r].astype(np.float32)[:, None, None] for r in range(3))
        out[1:, y:y + N, x:x + N] += (a + gy * eta[None, :, None] + gx * eta[None, None, :]) * prep['hhat'][k][None]


# ----------------------------------------------------------------------------- velocity
def regression_weights(dates_years):
    t = np.asarray(dates_years, float)
    t = t - t.mean()
    return t / (t ** 2).sum()


def _basis(yy, xx, shape, order):
    """Columns: [1, u, v, (u^2, uv, v^2)] with u,v in [-1,1] over the frame."""
    H, W = shape
    v = 2 * yy / max(H - 1, 1) - 1
    u = 2 * xx / max(W - 1, 1) - 1
    cols = [np.ones_like(u), u, v]
    if order >= 2:
        cols += [u * u, u * v, v * v]
    return np.stack(cols, axis=-1)


def _sample(prep, delta, order, sub, nsigma=3., iters=3):
    """Robust polynomial fit of delta on a subsampled valid grid. Returns sample indices, pinv, coeffs."""
    H, W = prep['shape']
    yy, xx = np.mgrid[0:H:sub, 0:W:sub]
    ok = (prep['den'][yy, xx] > 0) & np.isfinite(delta[yy, xx])
    yy, xx = yy[ok].astype(float), xx[ok].astype(float)
    d = delta[yy.astype(int), xx.astype(int)].astype(float)
    Psi = _basis(yy, xx, (H, W), order)
    use = np.ones(len(d), bool)
    for _ in range(iters):
        c = np.linalg.lstsq(Psi[use], d[use], rcond=None)[0]
        r = d - Psi @ c
        s = 1.4826 * np.median(np.abs(r[use] - np.median(r[use]))) + 1e-12
        use = np.abs(r) < nsigma * s
    return yy[use].astype(int), xx[use].astype(int), d[use], Psi[use]


def velocity_constraint(prep, v_base, velocity, order, sub):
    """Q (nb-1, W), q* (nb-1): low-order (non-constant) coefficients of Interp(nu) must equal those of v_F - v_base."""
    H, Wd = prep['shape']
    N = prep['N']
    yy, xx, d, Psi = _sample(prep, velocity - v_base, order, sub)
    pinv = np.linalg.pinv(Psi)
    lookup = -np.ones((H, Wd), np.int64)
    lookup[yy, xx] = np.arange(len(yy))
    Hs = np.zeros((len(yy), len(prep['origins'])))
    for k, (y, x) in enumerate(prep['origins']):
        sub_i = lookup[y:y + N, x:x + N]
        m = sub_i >= 0
        Hs[sub_i[m], k] = prep['hhat'][k][m]
    Q = (pinv @ Hs)[1:]
    Q = Q - Q.mean(axis=1, keepdims=True)           # orthogonal to the uniform (gauge) mode
    return Q, (pinv @ d)[1:], len(yy)


def constrained_offsets(E0, fs, g1, Q, qstar, weight):
    """Minimum-energy correction so that Q (E g1) = qstar (hard if weight=inf, soft otherwise)."""
    r = qstar - Q @ (E0 @ g1)
    kappa = g1 @ fs['Minv'] @ g1
    K = (Q @ fs['Lp'] @ Q.T) * kappa
    reg = 0. if np.isinf(weight) else 1. / weight
    lam = np.linalg.lstsq(K + reg * np.eye(len(K)), r, rcond=None)[0]
    dE = np.outer(fs['Lp'] @ (Q.T @ lam), fs['Minv'] @ g1)
    return E0 + dE


def trend(U, g):
    return np.tensordot(g.astype(np.float32), U, axes=(0, 0))


def detrend_after(prep, U, g, dates, velocity, order, sub):
    """Subtract the low-order (non-constant) trend error (v_rec - v_F) * (t - t0)."""
    H, Wd = prep['shape']
    yy, xx, d, Psi = _sample(prep, velocity - trend(U, g), order, sub)
    c = np.linalg.lstsq(Psi, d, rcond=None)[0]
    Y, X = np.mgrid[0:H, 0:Wd]
    B = _basis(Y.astype(float), X.astype(float), (H, Wd), order)
    poly = (B[..., 1:] @ c[1:]).astype(np.float32)
    U += poly[None] * (dates - dates[0]).astype(np.float32)[:, None, None]
    return c[1:]


def velocity_misfit(prep, U, g, velocity, order, sub):
    yy, xx, d, Psi = _sample(prep, trend(U, g) - velocity, order, sub)
    return np.linalg.lstsq(Psi, d, rcond=None)[0][1:]


# ----------------------------------------------------------------------------- plane fit of every epoch
def fit_plane_epochs(prep, U, order=1, sub=4, nsigma=3., iters=3):
    """Fit, for every epoch, a robust polynomial (plane for order 1) to the final U on a subsampled valid grid and
    subtract its non-constant terms (the per-epoch constant is left untouched). In place. Returns the removed
    coefficients, shape (T, nb-1): columns (u = range, v = azimuth, [u^2, uv, v^2]) per unit of normalised
    coordinate in [-1, 1] over the frame, in the units of U."""
    T = U.shape[0]
    H, Wd = prep['shape']
    yy, xx = np.mgrid[0:H:sub, 0:Wd:sub]
    ok = prep['den'][yy, xx] > 0
    yi, xi = yy[ok], xx[ok]
    Psi = _basis(yi.astype(float), xi.astype(float), (H, Wd), order)
    nb = Psi.shape[1]
    Y, X = np.mgrid[0:H, 0:Wd]
    v = (2 * Y / max(H - 1, 1) - 1).astype(np.float32)
    u = (2 * X / max(Wd - 1, 1) - 1).astype(np.float32)
    coefs = np.zeros((T, nb - 1))
    for t in range(1, T):
        d = U[t][yi, xi].astype(np.float64)
        use = np.isfinite(d)
        for _ in range(iters):
            c = np.linalg.lstsq(Psi[use], d[use], rcond=None)[0]
            r = d - Psi @ c
            s = 1.4826 * np.median(np.abs(r[use] - np.median(r[use]))) + 1e-12
            use = np.isfinite(d) & (np.abs(r) < nsigma * s)
        c = np.linalg.lstsq(Psi[use], d[use], rcond=None)[0]
        coefs[t] = c[1:]
        terms = [u, v] + ([u * u, u * v, v * v] if order >= 2 else [])
        for ck, tk_ in zip(c[1:], terms):
            U[t] -= np.float32(ck) * tk_
    return coefs


# ----------------------------------------------------------------------------- driver
def denoise_series(cube, topo, engine, dataset_cls, mask=None, N=128, Nt=9, spatial_stride=64,
                   t_stride=2, edge_trim=8, min_overlap_frac=0.25, min_pixels=200,
                   fill_range=10., seed=0, batch_windows=2000, verbose=True,
                   velocity=None, dates=None, velocity_mode='none', velocity_order=1,
                   velocity_weight=np.inf, velocity_subsample=4,
                   solve_tilt=False, tilt_prior=0.0, fit_plane=False, plane_order=1, plane_subsample=4):
    """cube: (T,H,W) float32 (modified in place: NaNs filled). Returns U (T,H,W), info.

    solve_tilt: also invert for a tilt (gy, gx) per tile and epoch, not only the offset (tilt_prior > 0 adds a weak
                ridge on the tilts). Not available with velocity_mode='before'.
    fit_plane:  after the whole reconstruction, fit a plane (plane_order=1; 2 = quadratic) to every epoch of U and
                remove its gradient terms (the per-epoch constant is kept)."""
    assert velocity_mode in ('none', 'before', 'after')
    T = cube.shape[0]
    assert T >= Nt and 1 <= t_stride <= Nt - 1
    if velocity_mode != 'none' and (velocity is None or dates is None):
        raise ValueError('velocity_mode != none requires `velocity` (H,W; cube units/yr) and `dates` (decimal years)')
    if solve_tilt and velocity_mode == 'before':
        raise ValueError("solve_tilt is not available with velocity_mode='before' (use 'after' and/or fit_plane)")
    if mask is None:
        mask = nan_all(cube)
    with prof.section('setup.static'):
        prep = prepare_static(topo, mask, N, spatial_stride, edge_trim, min_overlap_frac, min_pixels)
    H, Wd = prep['shape']
    W = len(prep['origins'])
    with prof.section('setup.fill'):
        fill_cube_inplace(cube, fill_range, seed)

    with prof.section('setup.matrices'):
        t_starts = _starts(T, Nt, t_stride)
        K = len(t_starts)
        w = temporal_weights(Nt)
        M = temporal_matrix(T, Nt, t_starts, w)
        A, a = pair_incidence(prep)
        P_ = len(prep['pairs'])
        if solve_tilt:
            geo = pair_geometry(prep)
            A3, ws = pair_incidence_tilt(prep, geo, a)
    log(f'{W} tiles, {P_} pairs, {K} chunks, max_lag={prep["max_lag"]}, solve_tilt={solve_tilt}, '
        f'fit_plane={fit_plane}')

    rhs = np.zeros((T - 1, H, Wd), np.float32)
    nrow = 3 * P_ if solve_tilt else P_
    R = np.zeros((T - 1, nrow))
    D_all = np.zeros((K, nrow, Nt - 1))
    t_loop = time.perf_counter()
    for k, t0 in enumerate(t_starts):
        with prof.section('chunk.predict'):
            P = predict_chunk(prep, cube, int(t0), Nt, engine, dataset_cls, batch_windows, verbose)
        with prof.section('chunk.planes' if solve_tilt else 'chunk.medians'):
            D = overlap_planes(prep, geo, P).reshape(nrow, Nt - 1) if solve_tilt else overlap_medians(prep, P)
        D_all[k] = D
        with prof.section('chunk.acc_R'):
            accumulate(R, np.concatenate([np.zeros((1, D.shape[0])), D.T], axis=0), int(t0), w)
        with prof.section('chunk.blend'):
            Y = blend(prep, P)
        with prof.section('chunk.acc_rhs'):
            accumulate(rhs, Y, int(t0), w)
        del P
        eta = (time.perf_counter() - t_loop) / (k + 1) * (K - k - 1)
        log(f'chunk {k + 1}/{K} (frames {t0}-{t0 + Nt - 1}) ETA {eta / 60:.1f} min')

    # uncorrected field: u = M^-1 rhs
    with prof.section('solve.temporal'):
        cf = sla.cho_factor(M)
        flat = rhs.reshape(T - 1, -1)
        for b in range(0, flat.shape[1], 20000):
            flat[:, b:b + 20000] = sla.cho_solve(cf, flat[:, b:b + 20000])
        U = np.zeros((T, H, Wd), np.float32)
        U[1:] = rhs
        del rhs, flat

    info = dict(n_tiles=W, n_windows=W, n_pairs=P_, t_starts=t_starts, velocity_mode=velocity_mode,
                solve_tilt=bool(solve_tilt), fit_plane=bool(fit_plane))
    g = None
    if velocity_mode != 'none':
        dates = np.asarray(dates, float)
        g = regression_weights(dates)

    if solve_tilt:
        with prof.section('solve.tilt'):
            ft = factor_tilt(A3, ws, M, tilt_prior)
            Theta = solve_tilt_system(A3, ws, R.T, ft)
            info['rms_before'], info['rms_after'] = tilt_residual(A3, Theta, D_all, t_starts, Nt, ws, w)
            Theta = Theta.reshape(W, 3, T - 1)
            E = Theta[:, 0, :]
            info['tilt'] = Theta[:, 1:, :]
            info['n_null_modes'] = 3
        with prof.section('assemble.interp'):
            interp_offsets_tilt(prep, Theta, U)
    else:
        with prof.section('solve.factor'):
            fs = factor_system(A, a, M)
        with prof.section('solve.offsets'):
            E = solve_offsets(A, a, R.T, fs)
            info['n_null_modes'] = fs['n_null']
            info['rms_before'], info['rms_after'] = offset_residual(A, E, D_all, t_starts, Nt, a, w)
        if velocity_mode == 'before':
            with prof.section('velocity.before'):
                v_base = trend(U, g)
                Q, qstar, ns = velocity_constraint(prep, v_base, velocity, velocity_order, velocity_subsample)
                E = constrained_offsets(E, fs, g[1:], Q, qstar, velocity_weight)
                info['rms_after_constrained'] = offset_residual(A, E, D_all, t_starts, Nt, a, w)[1]
                info['velocity_samples'] = ns
        with prof.section('assemble.interp'):
            interp_offsets(prep, E, U)

    if velocity_mode == 'after':
        with prof.section('velocity.after'):
            info['velocity_correction'] = detrend_after(prep, U, g, dates, velocity,
                                                        velocity_order, velocity_subsample)
    if fit_plane:
        with prof.section('plane.fit'):
            info['plane_coeffs'] = fit_plane_epochs(prep, U, plane_order, plane_subsample)
    if velocity_mode != 'none':
        info['velocity_misfit_final'] = velocity_misfit(prep, U, g, velocity, velocity_order,
                                                        velocity_subsample)
    U[:, prep['den'] == 0] = np.nan
    info['E'] = E
    info['origins'] = prep['origins']
    return U, info
