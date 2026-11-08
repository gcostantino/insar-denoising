import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from utils.profiling import prof, log


class LazyWindows:
    """Array-like of shape (W, Nt, N, N, 1); crops one window per access, nothing precomputed."""

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
        idx = np.arange(len(self))[i]  # slices / index arrays
        return np.stack([self[int(j)] for j in idx])

    def __array__(self, dtype=None, copy=None):
        raise RuntimeError('LazyWindows must not be materialized: the dataset touched the full array')


class LazyTopo:
    """Array-like of shape (W, 1, N, N, 1); per-window min/max normalization on access."""

    def __init__(self, topo2d, origins, N):
        self.topo2d, self.origins, self.N = topo2d, origins, N
        self.shape = (len(origins), 1, N, N, 1)
        self.dtype = np.dtype(np.float32)
        self.ndim = 5

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            y, x = self.origins[i]
            t = self.topo2d[y:y + self.N, x:x + self.N]
            mn = np.nanmin(t)
            rg = np.nanmax(t) - mn
            t = np.nan_to_num((t - mn) / (rg if rg > 0 else 1.0))
            return t[None, :, :, None].astype(np.float32)
        idx = np.arange(len(self))[i]
        return np.stack([self[int(j)] for j in idx])

    def __array__(self, dtype=None, copy=None):
        raise RuntimeError('LazyTopo must not be materialized')


def _stat(name):
    return {'mean': np.nanmean, 'median': np.nanmedian}[name]


def _starts(L, N, stride):
    if L < N:
        raise ValueError(f'Dimension {L} smaller than window size {N}')
    s = list(range(0, L - N + 1, stride))
    if s[-1] != L - N:
        s.append(L - N)  # flush last window with the edge
    return np.array(s)


def make_grid(shape, N, stride):
    sy, sx = (stride, stride) if np.isscalar(stride) else stride
    ys, xs = _starts(shape[0], N, sy), _starts(shape[1], N, sx)
    origins = np.array([(y, x) for y in ys for x in xs])  # flat index = iy*nxw + ix
    return origins, (len(ys), len(xs))


def extract_windows(arr, origins, N):
    """arr: (..., ny, nx) -> (W, ..., N, N)"""
    return np.stack([arr[..., y:y + N, x:x + N] for y, x in origins])


def window_topo(static, idx):
    """Per-window min/max normalized topo for the windows in idx -> (b,1,N,N,1)."""
    tw = extract_windows(static['topo2d'], static['origins'][idx], static['N'])
    mn = np.nanmin(tw, axis=(1, 2), keepdims=True)
    rg = np.nanmax(tw, axis=(1, 2), keepdims=True) - mn
    tw = np.nan_to_num((tw - mn) / np.where(rg > 0, rg, 1.0))
    return tw[:, None, :, :, None]


def nan_all(cube):
    m = np.ones(cube.shape[1:], bool)
    for t in range(cube.shape[0]):
        m &= np.isnan(cube[t])
    return m


@prof.profiled('fill_nans')
def fill_cube_inplace(cube, fill_range=10., seed=0):
    """Frame-by-frame fill of NaNs (cube must be float32 and writable)."""
    rng = np.random.default_rng(seed)
    for t in range(cube.shape[0]):
        nan = np.isnan(cube[t])
        n = int(nan.sum())
        if n:
            cube[t][nan] = rng.uniform(-fill_range, fill_range, n).astype(np.float32)


def fill_cube(cube, fill_range=10., seed=0):
    """Copy variant, used only when the original NaN cube is needed (anchoring)."""
    filled = cube.astype(np.float32, copy=True)
    fill_cube_inplace(filled, fill_range, seed)
    return filled


@prof.profiled('prepare_static')
def prepare_static(topo, mask, N, spatial_stride, min_valid_frac=0.0):
    ny, nx = mask.shape
    assert topo.shape == (ny, nx), (topo.shape, mask.shape)
    origins, _ = make_grid((ny, nx), N, spatial_stride)
    ys_f, xs_f = np.unique(origins[:, 0]), np.unique(origins[:, 1])
    gidx = np.stack([np.searchsorted(ys_f, origins[:, 0]),
                     np.searchsorted(xs_f, origins[:, 1])], axis=1)
    wmask = extract_windows(mask.astype(bool), origins, N)
    keep = np.flatnonzero((1 - wmask.mean(axis=(1, 2))) >= max(min_valid_frac, 1e-12))
    return dict(origins=origins[keep], gidx=gidx[keep], wmask=wmask[keep],
                topo2d=topo.astype(np.float32), ys_f=ys_f, xs_f=xs_f,
                shape=(ny, nx), N=N)


def temporal_weights(Nt, kind='hann'):
    """Weights for tau = 1..Nt-1 (tau = 0 is the zero reference frame)."""
    if kind == 'uniform':
        return np.ones(Nt - 1)
    if kind == 'hann':
        return np.hanning(Nt + 2)[1:-1][1:]
    raise ValueError(kind)


@prof.profiled('temporal_system')
def temporal_system(T, Nt, t_starts, w, ridge=1e-9):
    """Normal matrix of: u(t_k+tau) - u(t_k) = Y_k(tau), with u(0)=0. Unknowns u(1..T-1)."""
    rows = []
    for tk in t_starts:
        for tau in range(1, Nt):
            r = np.zeros(T - 1)
            r[tk + tau - 1] += 1
            if tk > 0:
                r[tk - 1] -= 1
            rows.append((r, w[tau - 1]))
    B = np.array([r for r, _ in rows])
    wr = np.array([x for _, x in rows])
    return B.T @ (wr[:, None] * B) + ridge * np.eye(T - 1)


@prof.profiled('predict')
def predict_chunk(static, filled, t0, Nt, engine, dataset_cls,
                  raw_cube=None, max_windows_per_call=None, verbose=True):
    o, N = static['origins'], static['N']
    W = len(o)

    def _predict(s, e):
        ds = dataset_cls(x=LazyWindows(filled, o[s:e], N, t0, Nt),
                         topo=LazyTopo(static['topo2d'], o[s:e], N))
        out = np.asarray(engine.predict(data=ds, verbose=verbose))
        out = out[..., 0] if out.ndim == 5 else out  # contiguous view, no copy
        assert out.shape[0] == e - s, out.shape
        return np.asarray(out, dtype=np.float32)

    if max_windows_per_call is None or max_windows_per_call >= W:
        pred = _predict(0, W)
    else:
        pred = np.empty((W, Nt, N, N), np.float32)
        for s in range(0, W, max_windows_per_call):
            e = min(s + max_windows_per_call, W)
            pred[s:e] = _predict(s, e)

    wm = static['wmask']
    for k in range(W):  # in-place masking
        pred[k][:, wm[k]] = np.nan

    raw_rel = None
    if raw_cube is not None:
        raw_rel = extract_windows(raw_cube[t0:t0 + Nt], o, N).astype(np.float32, copy=False)
        raw_rel -= raw_rel[:, :1].copy()
        for k in range(W):
            raw_rel[k][:, wm[k]] = np.nan
    return pred, raw_rel


@prof.profiled('select_subset')
def select_subset(prep, stride):
    """Windows of a coarser grid that are present in the cached fine grid."""
    ny, nx = prep['shape']
    N = prep['N']
    sy, sx = (stride, stride) if np.isscalar(stride) else stride
    ys, xs = _starts(ny, N, sy), _starts(nx, N, sx)
    ys_f, xs_f, g = prep['ys_f'], prep['xs_f'], prep['gidx']
    if not (np.isin(ys, ys_f).all() and np.isin(xs, xs_f).all()):
        raise ValueError('Stride not compatible with the fine grid (use multiples of it).')
    sel = np.flatnonzero(np.isin(ys_f[g[:, 0]], ys) & np.isin(xs_f[g[:, 1]], xs))
    gnew = np.stack([np.searchsorted(ys, ys_f[g[sel, 0]]),
                     np.searchsorted(xs, xs_f[g[sel, 1]])], axis=1)
    return sel, prep['origins'][sel], gnew


def stitch_chunk(static, pred, raw_rel, stride, stat='median', correct=True, taper='hann',
                 edge_trim=8, max_lag=None, min_overlap_frac=0.25, min_pixels=200,
                 anchor_weight=0.0):
    N = static['N']
    sel, origins, gnew = select_subset(static, stride)
    full = len(sel) == len(static['origins'])
    p = pred if full else pred[sel]  # no copy in the common case
    if anchor_weight > 0 and raw_rel is None:
        raise ValueError('anchor_weight > 0 requires raw_rel')
    r = None if raw_rel is None else (raw_rel if full else raw_rel[sel])
    s = stride if np.isscalar(stride) else min(stride)
    if max_lag is None:
        max_lag = int(np.clip(np.ceil(0.5 * N / s), 1, 6))
    info = dict(n_windows=len(sel), max_lag=max_lag)
    if correct:
        c, est = estimate_offsets(p, origins, gnew, N, stat, max_lag, min_overlap_frac,
                                  edge_trim, min_pixels, raw=r, anchor_weight=anchor_weight)
        info.update(est)
    else:
        c = np.zeros(p.shape[:2], np.float32)
    return blend(p, c, origins, static['shape'], N, taper), info


@prof.profiled('blend')
def blend(pred, c, origins, out_shape, N, taper='hann'):
    T = pred.shape[1]
    H, Wd = out_shape
    if taper == 'hann':
        h = np.hanning(N + 2)[1:-1]
        w2d = np.outer(h, h)
    else:
        w2d = np.ones((N, N))
    num = np.zeros((T, H, Wd))
    den = np.zeros((T, H, Wd))
    for k, (y, x) in enumerate(origins):
        pw = pred[k] + c[k][:, None, None]
        wt = w2d[None] * np.isfinite(pw)
        num[:, y:y + N, x:x + N] += np.nan_to_num(pw) * wt
        den[:, y:y + N, x:x + N] += wt
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den == 0] = np.nan
    return out


@prof.profiled('offsets')
def estimate_offsets(pred, origins, gidx, N, stat='median', max_lag=3, min_overlap_frac=0.25,
                     edge_trim=8, min_pixels=200, raw=None, anchor_weight=0.0, ridge=1e-6):
    f = _stat(stat)
    W, T = pred.shape[:2]
    lo, hi = edge_trim, N - edge_trim
    full_area = (hi - lo) ** 2
    gl = gidx.tolist()
    lookup = {tuple(g): k for k, g in enumerate(gl)}
    offs = [(dy, dx) for dy in range(0, max_lag + 1) for dx in range(-max_lag, max_lag + 1)
            if not (dy == 0 and dx <= 0)]

    rows, cols, vals, D, n_pix = [], [], [], [], []
    p = 0
    with prof.section('pairs'):
        for i, (iy, ix) in enumerate(gl):
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
                a = pred[i][:, y0 - yi:y1 - yi, x0 - xi:x1 - xi]
                b = pred[j][:, y0 - yj:y1 - yj, x0 - xj:x1 - xj]
                diff = (b - a).reshape(T, -1)
                n = np.isfinite(diff[0]).sum()
                if n < min_pixels:
                    continue
                D.append(np.nan_to_num(f(diff, axis=1)))  # c_i - c_j = stat(pred_j - pred_i)
                n_pix.append(n)
                rows += [p, p]
                cols += [i, j]
                vals += [1., -1.]
                p += 1
        if p == 0:
            raise RuntimeError('No valid overlapping pairs (check stride / min_pixels / mask).')

    with prof.section('solve'):
        A = sp.csr_matrix((vals, (rows, cols)), shape=(p, W))
        D = np.array(D)
        w = np.array(n_pix, float)
        w /= w.mean()
        M = (A.T @ sp.diags(w) @ A).tocsc()
        rhs = A.T @ (w[:, None] * D)

        if raw is not None and anchor_weight > 0:
            a = f((raw - pred).reshape(W, T, -1), axis=2)
            lam = anchor_weight * np.isfinite(a[:, 0])
            M = M + sp.diags(lam)
            rhs = rhs + lam[:, None] * np.nan_to_num(a)

        c = splu((M + ridge * sp.eye(W)).tocsc()).solve(np.asarray(rhs))
    before = np.sqrt(np.mean(w[:, None] * D ** 2))
    after = np.sqrt(np.mean(w[:, None] * (A @ c - D) ** 2))
    return c, dict(n_pairs=p, rms_before=before, rms_after=after)


@prof.profiled('accumulate')
def add_chunk(rhs, Y, tk, w):
    """Accumulate B^T W Y for one chunk."""
    for tau in range(1, Y.shape[0]):
        val = np.nan_to_num(Y[tau]) * w[tau - 1]
        rhs[tk + tau - 1] += val
        if tk > 0:
            rhs[tk - 1] -= val


def estimate_workload(shape, T, N=128, Nt=9, spatial_stride=16, t_stride=2):
    ny, nx = shape
    ny_w = len(_starts(ny, N, spatial_stride))
    nx_w = len(_starts(nx, N, spatial_stride))
    n_chunks = len(_starts(T, Nt, t_stride))
    W = ny_w * nx_w
    mb_chunk = W * Nt * N * N * 4 / 1e6  # float32 frames for one chunk
    log(f'spatial windows: {ny_w} x {nx_w} = {W}')
    log(f'time chunks: {n_chunks} (Nt={Nt}, t_stride={t_stride})')
    log(f'total crops to denoise: {W * n_chunks}')
    log(f'~{mb_chunk / 1e3:.1f} GB per chunk for inputs (same again for predictions)')


def denoise_series(cube, topo, engine, dataset_cls, mask=None, N=128, Nt=9,
                   spatial_stride=16, t_stride=2, temporal_taper='hann',
                   min_valid_frac=0.0, fill_range=10., seed=0, keep_chunks=False,
                   batch_windows=2000, verbose=True, **stitch_kw):
    T = cube.shape[0]
    assert T >= Nt, f'Need at least {Nt} frames'
    assert 1 <= t_stride <= Nt - 1
    if mask is None:
        mask = nan_all(cube)  # before any filling
    static = prepare_static(topo, mask, N, spatial_stride, min_valid_frac)
    H, W = static['shape']
    nW = len(static['origins'])

    if stitch_kw.get('anchor_weight', 0.0) > 0:
        filled, raw_cube = fill_cube(cube, fill_range, seed), cube
    else:
        fill_cube_inplace(cube, fill_range, seed)
        filled, raw_cube = cube, None

    t_starts = _starts(T, Nt, t_stride)
    w = temporal_weights(Nt, temporal_taper)
    M = temporal_system(T, Nt, t_starts, w)
    rhs = np.zeros((T - 1, H, W))
    bad = np.zeros((H, W), bool)
    infos, chunks = [], []

    n_grid = len(_starts(H, N, spatial_stride)) * len(_starts(W, N, spatial_stride))
    log(f"windows per chunk: {len(static['origins'])} kept of {n_grid} "
        f"({n_grid - len(static['origins'])} skipped as empty)")
    log(f"chunks: {len(t_starts)} -> total crops to denoise: "
        f"{len(static['origins']) * len(t_starts)}")
    estimate_workload(static['shape'], T, N=N, Nt=Nt, spatial_stride=spatial_stride, t_stride=t_stride)

    t_loop = time.perf_counter()
    for k, t0 in enumerate(t_starts):
        pred, raw_rel = predict_chunk(static, filled, int(t0), Nt, engine, dataset_cls,
                                      raw_cube=raw_cube, max_windows_per_call=batch_windows,
                                      verbose=verbose)
        Y, info = stitch_chunk(static, pred, raw_rel, spatial_stride, **stitch_kw)
        del pred, raw_rel
        add_chunk(rhs, Y, int(t0), w)
        bad |= ~np.isfinite(Y[1:]).all(axis=0)
        infos.append(info)
        if keep_chunks:
            chunks.append(Y.astype(np.float32))
        del Y

        eta = (time.perf_counter() - t_loop) / (k + 1) * (K - k - 1)
        msg = (f'chunk {k + 1}/{K} (frames {t0}-{t0 + Nt - 1}) | {nW} windows | '
               f'rms {info.get("rms_before", np.nan):.3f}->{info.get("rms_after", np.nan):.3f} | '
               f'ETA {eta / 60:.1f} min')
        if prof.enabled:
            msg += (f' | predict {prof.get("predict"):.1f}s ({nW / prof.get("predict"):.0f} win/s)'
                    f' | offsets {prof.get("offsets"):.1f}s | blend {prof.get("blend"):.1f}s')
        log(msg)

    log('temporal solve...')
    with prof.section('temporal_solve'):
        u = np.tensordot(np.linalg.inv(M), rhs, axes=1)
        U = np.concatenate([np.zeros((1, H, W)), u], axis=0)
        U[:, bad] = np.nan
        U = U.astype(np.float32)
    return U, dict(chunks=chunks, infos=infos, t_starts=t_starts, n_windows=nW)
