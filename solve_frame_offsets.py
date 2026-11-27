#!/usr/bin/env python
"""
Per-epoch frame offsets from the overlaps (hypothesis: the input data are perfect, so every frame differs from the
truth by one scalar e_i(t) per epoch).

For every pair of series of the same orbit that share at least one cell:
  D_ij(t) = median over ALL common cells (no minimum area, no subsampling, no edge trimming) of u_i(t) - u_j(t),
  on a common regular time grid (each series is interpolated linearly in time at the grid dates, only inside its own
  span and only inside the common window of the pair).
Then, independently for every grid date, solve  e_i - e_j = D_ij  by least squares over the frames that have data at
that date (equal weights by default; --weight sqrt = sqrt(#cells)). The gauge is set by --ref-track frames (zero mean
offset); a connected group without a reference frame uses its most connected frame. No smoothing, no robust
re-weighting: the residuals are saved so that you can see where the hypothesis fails.

The cells are only used to pair the pixels of different frames (the pixels of two tracks are not aligned); the
offsets are scalars, nothing is resampled in the output. Corrected series = u_i(t) - e_i(t), applied by the cube writer.

Output (HDF5): /frames, /time, /e (frames x dates, NaN = no data or no overlap at that date), /n_edges, /pairs/*
(median, n valid cells, MAD, residual after the solve) + a PNG with e_i(t).

Examples:
  python solve_frame_offsets.py --results-dir results/<tag> --match MEXIQUE --ref-track D070 --out off_mexico.h5
  python solve_frame_offsets.py --results-dir results/<tag> --match TIBET --ref-track D077 --out off_tibet.h5
Needs build_disp_mosaic.py in the same folder.
"""
import argparse
import glob
import itertools
import os
import re
import time
from datetime import datetime

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from build_disp_mosaic import attr_str, native_spacing_deg, parse_dates, pixel_cells, track_label


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def fmt(t):
    return datetime.fromordinal(int(t)).strftime('%Y-%m-%d')


# ----------------------------------------------------------------------------- frames
def frame_geometry(path, res, nx_g):
    """Unmasked valid pixels of a series and the id of the cell each one falls in."""
    with h5py.File(path, 'r') as f:
        uniq, inv, ok, masked, _ = pixel_cells(f, res, nx_g)
        dates = parse_dates(f['dates'][...])
        meta = dict(orbit=attr_str(f.attrs.get('orbit', 'unknown')).lower(),
                    key=attr_str(f.attrs.get('key', os.path.basename(path))),
                    region=attr_str(f.attrs.get('region', '')))
    flat = np.flatnonzero(ok.ravel())
    keep = ~masked.ravel()[flat]
    cid = uniq[inv][keep]
    ids = np.unique(cid)
    g = dict(path=path, dates=dates, shape=ok.shape, pix=flat[keep], cid=cid, ids=ids, **meta)
    if len(ids):
        ix, iy = ids % nx_g, ids // nx_g
        g['box'] = (ix.min(), ix.max(), iy.min(), iy.max())
    g['label'] = f'{track_label(meta["key"])} {fmt(dates[0])[:7]}..{fmt(dates[-1])[:7]}'
    return g


class CellSeries:
    """Mean displacement of the valid pixels of every cell of `common`, epoch by epoch (read in time blocks)."""

    def __init__(self, g, common, block=32):
        pos = np.minimum(np.searchsorted(common, g['cid']), len(common) - 1)
        hit = common[pos] == g['cid']
        self.cp, pix = pos[hit], g['pix'][hit]
        ny, nx = g['shape']
        r, c = np.divmod(pix, nx)
        self.r0, self.r1, self.c0, self.c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
        self.loc = (r - self.r0) * (self.c1 - self.c0) + (c - self.c0)
        self.n, self.path, self.T, self.block, self.cache = len(common), g['path'], len(g['dates']), block, {}

    def _load(self, b):
        k0, k1 = b * self.block, min((b + 1) * self.block, self.T)
        with h5py.File(self.path, 'r') as f:
            blk = f['displacement'][k0:k1, self.r0:self.r1, self.c0:self.c1].reshape(k1 - k0, -1)
        out = np.empty((k1 - k0, self.n), np.float32)
        for q in range(k1 - k0):
            v = blk[q, self.loc].astype(np.float64)
            fin = np.isfinite(v)
            s = np.bincount(self.cp, weights=np.where(fin, v, 0.0), minlength=self.n)
            c = np.bincount(self.cp, weights=fin.astype(float), minlength=self.n)
            out[q] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        return out

    def get(self, k):
        b = k // self.block
        if b not in self.cache:
            for old in [x for x in self.cache if abs(x - b) > 1]:
                del self.cache[old]
            self.cache[b] = self._load(b)
        return self.cache[b][k - b * self.block]

    def at(self, dates, t):
        """Linear interpolation in time at ordinal day t (inside the series span)."""
        k = int(np.clip(np.searchsorted(dates, t, 'right') - 1, 0, len(dates) - 2))
        den = float(dates[k + 1] - dates[k])
        fr = 0.0 if den <= 0 else min(max((t - dates[k]) / den, 0.0), 1.0)
        a = self.get(k)
        if fr == 0.0:
            return a
        b = self.get(k + 1)
        return b if fr == 1.0 else (1.0 - fr) * a + fr * b


def pair_series(gi, gj, common, grid, min_cells, block):
    """Median / n valid / MAD of u_i - u_j over the common cells at every grid date inside the common window."""
    n = len(grid)
    med, mad = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    nv = np.zeros(n, np.int64)
    ta, tb = max(gi['dates'][0], gj['dates'][0]), min(gi['dates'][-1], gj['dates'][-1])
    idx = np.flatnonzero((grid >= ta) & (grid <= tb))
    if len(idx) == 0:
        return med, nv, mad
    si, sj = CellSeries(gi, common, block), CellSeries(gj, common, block)
    for q in idx:
        d = si.at(gi['dates'], grid[q]) - sj.at(gj['dates'], grid[q])
        d = d[np.isfinite(d)]
        nv[q] = len(d)
        if len(d) >= min_cells:
            m = np.median(d)
            med[q], mad[q] = m, 1.4826 * np.median(np.abs(d - m))
    return med, nv, mad


# ----------------------------------------------------------------------------- per-epoch least squares
def solve_epoch(n, a, b, obs, w, ref):
    """x_a - x_b = obs (weights w on the squared residuals). Gauge per connected group: mean of the reference
    frames = 0, or the most connected frame if the group has no reference. Returns x (NaN for frames without
    any edge), edge residuals, degree."""
    x = np.full(n, np.nan)
    m = len(a)
    deg = np.zeros(n, int)
    if m == 0:
        return x, np.zeros(0), deg
    deg = np.bincount(np.r_[a, b], minlength=n)
    parent = list(range(n))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for p, q in zip(a, b):
        parent[find(p)] = find(q)
    roots = np.array([find(u) for u in range(n)])
    used = np.flatnonzero(deg > 0)
    A = np.zeros((m, n))
    A[np.arange(m), a] = 1.0
    A[np.arange(m), b] = -1.0
    rows = []
    for r in np.unique(roots[used]):
        mem = used[roots[used] == r]
        row = np.zeros(n)
        refm = mem[ref[mem]]
        if len(refm):
            row[refm] = 1.0 / len(refm)
        else:
            row[mem[np.argmax(deg[mem])]] = 1.0
        rows.append(row)
    G = np.array(rows)
    sw = np.sqrt(np.r_[w, np.full(len(G), 1e3 * max(float(np.max(w)), 1.0))])
    M = np.vstack([A, G]) * sw[:, None]
    sol = np.linalg.lstsq(M, np.r_[obs, np.zeros(len(G))] * sw, rcond=None)[0]
    x[used] = sol[used]
    return x, A @ sol - obs, deg


# ----------------------------------------------------------------------------- main
def collect_files(a):
    excl = list(a.exclude)
    if a.exclude_file:
        excl += [l.strip() for l in open(a.exclude_file) if l.strip() and not l.lstrip().startswith('#')]
    files = list(a.files)
    if a.results_dir:
        for p in sorted(glob.glob(os.path.join(a.results_dir, '**', '*.h5'), recursive=True)):
            if os.path.basename(p).startswith('master') or 'DAUX' in p:
                continue
            if not a.match or any(m in p for m in a.match):
                files.append(p)
    out = []
    for p in dict.fromkeys(files):
        if any(x in p for x in excl):
            log(f'excluded: {os.path.basename(p)}')
            continue
        try:
            with h5py.File(p, 'r') as f:
                if bool(f.attrs.get('complete', 1)):
                    out.append(p)
        except OSError:
            log(f'cannot open {p}')
    return out


def plot_offsets(frames, grid, e, out_png):
    groups = {}
    for k, g in enumerate(frames):
        groups.setdefault((g['orbit'], g['region']), []).append(k)
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 3.3 * len(groups)), sharex=True, squeeze=False)
    dt = [datetime.fromordinal(int(t)) for t in grid]
    for ax, ((orb, reg), ks) in zip(axes[:, 0], groups.items()):
        for k in ks:
            ax.plot(dt, e[k], lw=0.8, label=frames[k]['label'])
        ax.axhline(0, color='k', lw=0.4)
        ax.set_title(f'{reg} | {orb}', fontsize=9)
        ax.set_ylabel('offset e_i(t) (mm)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5, ncol=3, loc='upper left')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*')
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--match', nargs='+', default=None, help='keep paths containing any of these strings')
    ap.add_argument('--exclude', nargs='+', default=[],
                    help='drop series whose path contains any of these strings (e.g. TIBET-D135NORD TIBET-D004)')
    ap.add_argument('--exclude-file', default=None,
                    help='text file with one string per line (# for comments), same meaning as --exclude')

    ap.add_argument('--out', default='frame_offsets.h5')
    ap.add_argument('--step-days', type=float, default=12.0, help='common time grid')
    ap.add_argument('--ref-track', nargs='+', default=None, help='track labels (substring) of the reference frames')
    ap.add_argument('--min-cells', type=int, default=1, help='minimum valid cells for a pair median at one date')
    ap.add_argument('--weight', choices=['none', 'sqrt'], default='none')
    ap.add_argument('--res', type=float, default=None, help='cell size in degrees (default: native pixel size)')
    ap.add_argument('--block', type=int, default=32, help='epochs read at once')
    ap.add_argument('--max-pairs', type=int, default=None)
    a = ap.parse_args()

    files = collect_files(a)
    log(f'{len(files)} series selected')
    if len(files) < 2:
        raise SystemExit('need at least two series')
    if a.res is None:
        sp = []
        for p in files:
            with h5py.File(p, 'r') as f:
                sp.append(native_spacing_deg(f))
        res = round(float(np.nanmedian(sp)), 7)
    else:
        res = a.res
    nx_g = int(np.ceil(360.0 / res))
    log(f'cell = {res} deg (only used to pair the pixels of different frames)')
    frames = []
    for p in files:
        g = frame_geometry(p, res, nx_g)
        if not len(g['ids']):
            log(f'  SKIPPED (no unmasked cell): {g["label"]}')
            continue
        frames.append(g)
        log(f'  {g["orbit"][:4]} {g["region"]:<28} {g["label"]}: {len(g["ids"])} cells')
    nf = len(frames)
    t0, t1 = min(g['dates'][0] for g in frames), max(g['dates'][-1] for g in frames)
    grid = np.arange(t0, t1 + 1, a.step_days).round().astype(np.int64)
    log(f'time grid: {len(grid)} dates, {fmt(grid[0])} .. {fmt(grid[-1])}, step {a.step_days} d')
    ref = np.array([bool(a.ref_track) and any(s in track_label(g['key']) for s in a.ref_track) for g in frames])
    log(f'reference frames: {[frames[k]["label"] for k in np.flatnonzero(ref)] or "automatic"}')

    pairs = []
    for i, j in itertools.combinations(range(nf), 2):
        gi, gj = frames[i], frames[j]
        if gi['orbit'] != gj['orbit']:
            continue
        bi, bj = gi['box'], gj['box']
        if bi[1] < bj[0] or bj[1] < bi[0] or bi[3] < bj[2] or bj[3] < bi[2]:
            continue
        common = np.intersect1d(gi['ids'], gj['ids'])
        if len(common) < a.min_cells:
            continue
        if max(gi['dates'][0], gj['dates'][0]) > min(gi['dates'][-1], gj['dates'][-1]):
            continue
        if a.max_pairs is not None and len(pairs) >= a.max_pairs:
            break
        t = time.time()
        med, nv, mad = pair_series(gi, gj, common, grid, a.min_cells, a.block)
        if not np.isfinite(med).any():
            continue
        pairs.append(dict(i=i, j=j, n_common=len(common), med=med, nv=nv, mad=mad))
        log(f'pair {len(pairs):3d}: {gi["label"]} - {gj["label"]}: {len(common)} cells, '
            f'{int(np.isfinite(med).sum())} dates ({time.time() - t:.0f}s)')
    if not pairs:
        raise SystemExit('no overlapping pairs')

    P = len(pairs)
    pi, pj = np.array([p['i'] for p in pairs]), np.array([p['j'] for p in pairs])
    med, nv, mad = (np.array([p[k] for p in pairs]) for k in ('med', 'nv', 'mad'))
    e = np.full((nf, len(grid)), np.nan, np.float32)
    res_p = np.full((P, len(grid)), np.nan, np.float32)
    n_edges = np.zeros((nf, len(grid)), np.int16)
    for k in range(len(grid)):
        ok = np.flatnonzero(np.isfinite(med[:, k]))
        w = np.ones(len(ok)) if a.weight == 'none' else np.sqrt(nv[ok, k].astype(float))
        x, r, deg = solve_epoch(nf, pi[ok], pj[ok], med[ok, k].astype(float), w, ref)
        e[:, k], res_p[ok, k], n_edges[:, k] = x, r, deg

    with h5py.File(a.out, 'w') as out:
        fr = out.create_group('frames')
        for key in ('path', 'orbit', 'key', 'region', 'label'):
            fr.create_dataset(key, data=np.array([g[key] for g in frames], dtype=object), dtype=h5py.string_dtype())
        fr.create_dataset('is_reference', data=ref.astype(np.uint8))
        fr.create_dataset('first_ord', data=np.array([g['dates'][0] for g in frames]))
        fr.create_dataset('last_ord', data=np.array([g['dates'][-1] for g in frames]))
        tg = out.create_group('time')
        tg.create_dataset('ordinal', data=grid)
        tg.create_dataset('yyyymmdd', data=np.array([int(fmt(t).replace('-', '')) for t in grid]))
        out.create_dataset('e', data=e, compression='gzip')
        out.create_dataset('n_edges', data=n_edges, compression='gzip')
        pg = out.create_group('pairs')
        pg.create_dataset('frame_i', data=pi)
        pg.create_dataset('frame_j', data=pj)
        pg.create_dataset('n_common_cells', data=np.array([p['n_common'] for p in pairs]))
        pg.create_dataset('median', data=med, compression='gzip')
        pg.create_dataset('n_valid', data=nv, compression='gzip')
        pg.create_dataset('mad', data=mad, compression='gzip')
        pg.create_dataset('residual', data=res_p, compression='gzip')
        out.attrs.update(dict(res_deg=res, step_days=a.step_days, weight=a.weight, min_cells=a.min_cells,
                              ref_track=' '.join(a.ref_track or []), units='mm',
                              convention='corrected = u_i(t) - e_i(t); pair median = u_i - u_j',
                              created=datetime.now().isoformat()))
    plot_offsets(frames, grid, e, os.path.splitext(a.out)[0] + '.png')

    log('--- consistency of the hypothesis (residual of the per-epoch solve, mm) ---')
    for p, r in zip(pairs, res_p):
        rr = r[np.isfinite(r)]
        log(f'  {frames[p["i"]]["label"]} - {frames[p["j"]]["label"]}: rms {np.sqrt(np.mean(rr ** 2)):.2f}, '
            f'max {np.abs(rr).max():.2f}, median MAD over the overlap {np.nanmedian(p["mad"]):.2f}')
    log('--- offsets ---')
    for k, g in enumerate(frames):
        v = e[k][np.isfinite(e[k])]
        nocorr = int(((n_edges[k] == 0) & (grid >= g['dates'][0]) & (grid <= g['dates'][-1])).sum())
        log(f'  {"REF " if ref[k] else "    "}{g["label"]:<34} range {v.min():+.1f}..{v.max():+.1f} mm, '
            f'{nocorr} dates of its span without any overlap' if len(v) else f'    {g["label"]}: no offset')
    log(f'saved {a.out} and {os.path.splitext(a.out)[0] + ".png"}')


if __name__ == '__main__':
    main()
