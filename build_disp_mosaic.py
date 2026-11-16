#!/usr/bin/env python
"""Build a sparse global mosaic of cumulative LOS displacement (last - first epoch) from the denoised
FLATSIM series (.h5), separately for ascending / descending, with optional inter-frame offset correction.

Offset correction (--offset-mode), solved independently for each orbit direction, constant in space per frame:
  none        raw mosaic only
  rate        (option A) one rate offset r_i [mm/yr] per frame. Observations: median over each overlap of the
              displacement difference over the pair's COMMON time window, divided by the window length
              (time series interpolated linearly at the window dates). Cumulative correction = r_i * span_i,
              rate correction = r_i  (so cumulative and rate stay consistent).
  cumulative  (option B) one cumulative offset c_i [mm] per frame. Observations: median over each overlap of
              (last-first)_i - (last-first)_j. Pairs whose spans differ by more than --b-max-span-diff-days
              are ignored (their real signal differs). Rate correction = c_i / span_i.
Overlap statistic = median over ALL overlapping cells (no edge trimming). Weights = sqrt(#valid overlap cells),
Huber-robust IRLS. Gauge: frames of --ref-track (zero mean offset) or, if absent in a connected group of
frames, the most connected frame of that group.

Output .h5: per orbit sparse cells (raw fields and, if corrected, *_corr_* fields), /frames table, /pairs table.
Units: mm and mm/yr, positive away from the satellite.

Example:
  python build_disp_mosaic.py --results-dir results/<tag> --out disp_mosaic.h5 --merge median \
         --offset-mode rate --ref-track D004
"""
import argparse
import glob
import os
import re
import shutil
import time
import traceback
import warnings
from datetime import datetime

import h5py
import numpy as np

LON0, LAT0 = -180.0, -90.0          # global grid origin; cell (ix, iy) spans [LON0+ix*res, LON0+(ix+1)*res)
ORBITS = ('ascending', 'descending')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def parse_dates(raw):
    out = []
    for x in np.asarray(raw).ravel():
        if isinstance(x, (bytes, np.bytes_)):
            s = x.decode()
        elif isinstance(x, str):
            s = x
        else:
            s = str(int(x))
        out.append(datetime.strptime(s.strip()[:8], '%Y%m%d').toordinal())
    return np.array(out)


def attr_str(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def track_label(key):
    m = re.search(r'S1_(.+?)-VV', key)
    return m.group(1) if m else key.split('/')[-1]


def list_series(results_dir, out_path):
    files = sorted(glob.glob(os.path.join(results_dir, '**', '*.h5'), recursive=True))
    keep = []
    for p in files:
        if os.path.basename(p).startswith('master') or os.path.abspath(p) == os.path.abspath(out_path):
            continue
        try:
            with h5py.File(p, 'r') as f:
                if bool(f.attrs.get('complete', 0)):
                    keep.append(p)
        except OSError:
            log(f'cannot open {p}, skipped')
    return keep


def native_spacing_deg(f, win=200):
    """Pixel size in degrees (sqrt of the median pixel area, longitude scaled by cos(lat))."""
    ny, nx = f['lon'].shape
    y0, x0 = max(0, ny // 2 - win // 2), max(0, nx // 2 - win // 2)
    lo = f['lon'][y0:y0 + win, x0:x0 + win].astype(float)
    la = f['lat'][y0:y0 + win, x0:x0 + win].astype(float)
    c = np.cos(np.deg2rad(np.nanmean(la)))
    ex, ey_ = (lo[:-1, 1:] - lo[:-1, :-1]) * c, la[:-1, 1:] - la[:-1, :-1]
    fx, fy = (lo[1:, :-1] - lo[:-1, :-1]) * c, la[1:, :-1] - la[:-1, :-1]
    return float(np.sqrt(np.nanmedian(np.abs(ex * fy - ey_ * fx))))


# ----------------------------------------------------------------------------- pixel -> cell binning
def pixel_cells(f, res, nx_g):
    """Cell index of every pixel. Returns sorted unique cell ids, pixel->cell index (valid-geometry pixels),
    boolean (ny,nx) mask of pixels with finite lon/lat, boolean (ny,nx) masked-out pixels, bbox."""
    lon, lat = f['lon'][...].astype(np.float64), f['lat'][...].astype(np.float64)
    lon = (lon + 180.0) % 360.0 - 180.0
    ok = np.isfinite(lon) & np.isfinite(lat)
    ix = np.floor((lon[ok] - LON0) / res).astype(np.int64)
    iy = np.floor((lat[ok] - LAT0) / res).astype(np.int64)
    uniq, inv = np.unique(iy * nx_g + ix, return_inverse=True)
    masked = f['mask'][...].astype(bool)
    bbox = (float(lon[ok].min()), float(lon[ok].max()), float(lat[ok].min()), float(lat[ok].max()))
    return uniq, inv, ok, masked, bbox


def bin_to_cells(field, inv, ok, masked, ncell):
    """Mean of the valid pixels of a (ny,nx) field in every cell (NaN if none)."""
    v = np.array(field, dtype=np.float64)
    v[masked] = np.nan
    v = v[ok]
    valid = np.isfinite(v)
    s = np.bincount(inv, weights=np.where(valid, v, 0.0), minlength=ncell)
    c = np.bincount(inv, weights=valid, minlength=ncell)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, s / c, np.nan).astype(np.float32)


def frame_cells(f, res, nx_g, n_avg):
    """Per-frame sparse result: sorted cell ids, cumulative displacement per cell (NaN if no valid pixel)."""
    U = f['displacement']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        last = np.nanmean(U[-n_avg:], axis=0)
        first = np.nanmean(U[:n_avg], axis=0)
    d = (last - first).astype(np.float32)
    uniq, inv, ok, masked, bbox = pixel_cells(f, res, nx_g)
    val = bin_to_cells(d, inv, ok, masked, len(uniq))
    dm = d.copy()
    dm[masked] = np.nan
    stats = dict(lon_min=bbox[0], lon_max=bbox[1], lat_min=bbox[2], lat_max=bbox[3],
                 n_pixels=int(ok.sum()), frac_masked=float(masked.mean()),
                 frac_nan_unmasked=float(np.isnan(d[~masked]).mean()) if (~masked).any() else np.nan,
                 frac_valid=float(np.isfinite(dm[ok]).sum() / max(ok.sum(), 1)))
    return uniq, val, stats


# ----------------------------------------------------------------------------- overlaps and observations
def overlap_pairs(ks, meta, scratch, min_cells):
    """Pairs of frames (same orbit) sharing at least min_cells grid cells. Saves the aligned cell positions."""
    pairs = []
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            i, j = ks[a], ks[b]
            mi, mj = meta[i], meta[j]
            if (mi['lon_max'] < mj['lon_min'] or mj['lon_max'] < mi['lon_min'] or
                    mi['lat_max'] < mj['lat_min'] or mj['lat_max'] < mi['lat_min']):
                continue
            ids_i = np.load(f'{scratch}/ids_{i}.npy', mmap_mode='r')
            ids_j = np.load(f'{scratch}/ids_{j}.npy', mmap_mode='r')
            if not len(ids_i) or not len(ids_j):
                continue
            lo, hi = max(ids_i[0], ids_j[0]), min(ids_i[-1], ids_j[-1])
            if hi < lo:
                continue
            si = slice(np.searchsorted(ids_i, lo), np.searchsorted(ids_i, hi, 'right'))
            sj = slice(np.searchsorted(ids_j, lo), np.searchsorted(ids_j, hi, 'right'))
            common, ia, ja = np.intersect1d(np.asarray(ids_i[si]), np.asarray(ids_j[sj]),
                                            assume_unique=True, return_indices=True)
            if len(common) < min_cells:
                continue
            np.save(f'{scratch}/ov_{i}_{j}.npy', ia + si.start)
            np.save(f'{scratch}/ov_{j}_{i}.npy', ja + sj.start)
            pairs.append(dict(i=i, j=j, n_common=int(len(common))))
    return pairs


def obs_cumulative(p, scratch):
    di = np.load(f'{scratch}/d_{p["i"]}.npy', mmap_mode='r')[np.load(f'{scratch}/ov_{p["i"]}_{p["j"]}.npy')]
    dj = np.load(f'{scratch}/d_{p["j"]}.npy', mmap_mode='r')[np.load(f'{scratch}/ov_{p["j"]}_{p["i"]}.npy')]
    v = np.isfinite(di) & np.isfinite(dj)
    if not v.any():
        return np.nan, 0
    return float(np.median(di[v].astype(np.float64) - dj[v])), int(v.sum())


def interp_cells(get, dates, t):
    """Cell field at ordinal day t, linear interpolation between the bracketing epochs."""
    k = int(np.clip(np.searchsorted(dates, t, 'right') - 1, 0, len(dates) - 2))
    den = float(dates[k + 1] - dates[k])
    fr = 0.0 if den <= 0 else min(max((t - dates[k]) / den, 0.0), 1.0)
    a = get(k)
    return a if fr == 0.0 else (1.0 - fr) * a + fr * get(k + 1)


def window_fields(ks, meta, pairs, scratch, res, nx_g, min_window_yr):
    """For every pair, displacement over the common window [ta, tb] of each frame at the overlap cells."""
    for p in pairs:
        mi, mj = meta[p['i']], meta[p['j']]
        p['ta'] = max(mi['first_ord'], mj['first_ord'])
        p['tb'] = min(mi['last_ord'], mj['last_ord'])
        p['T'] = (p['tb'] - p['ta']) / 365.25
        p['use'] = p['T'] >= min_window_yr
    for i in ks:
        mine = [p for p in pairs if p['use'] and i in (p['i'], p['j'])]
        if not mine:
            continue
        with h5py.File(meta[i]['path'], 'r') as f:
            dates = parse_dates(f['dates'][...])
            uniq, inv, ok, masked, _ = pixel_cells(f, res, nx_g)
            U, cache = f['displacement'], {}

            def get(k):
                if k not in cache:
                    cache[k] = bin_to_cells(U[k], inv, ok, masked, len(uniq))
                return cache[k]

            for p in mine:
                other = p['j'] if p['i'] == i else p['i']
                idx = np.load(f'{scratch}/ov_{i}_{other}.npy')
                da, db = interp_cells(get, dates, p['ta']), interp_cells(get, dates, p['tb'])
                np.save(f'{scratch}/w_{i}_{other}.npy', (db[idx] - da[idx]).astype(np.float32))
        log(f'  window fields: frame {i} ({len(mine)} pairs)')


def obs_rate(p, scratch):
    wi = np.load(f'{scratch}/w_{p["i"]}_{p["j"]}.npy')
    wj = np.load(f'{scratch}/w_{p["j"]}_{p["i"]}.npy')
    v = np.isfinite(wi) & np.isfinite(wj)
    if not v.any():
        return np.nan, 0
    return float(np.median(wi[v].astype(np.float64) - wj[v]) / p['T']), int(v.sum())


# ----------------------------------------------------------------------------- least-squares solution
def solve_offsets(n, edges, ref, k_huber=1.5, floor=0.1, iters=60):
    """x_a - x_b = obs for each edge (a, b, obs, weight). Gauge per connected group of frames.
    ref: boolean array of user-chosen reference frames (zero-mean gauge). Huber IRLS."""
    m = len(edges)
    A, b, w = np.zeros((m, n)), np.zeros(m), np.zeros(m)
    deg = np.zeros(n)
    parent = list(range(n))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for r, (a_, b_, o, wt) in enumerate(edges):
        A[r, a_], A[r, b_], b[r], w[r] = 1.0, -1.0, o, wt
        deg[a_] += 1
        deg[b_] += 1
        parent[find(a_)] = find(b_)
    roots = np.array([find(u) for u in range(n)])
    comp = np.unique(roots, return_inverse=True)[1]
    rows, is_ref = [], np.zeros(n, bool)
    for c in np.unique(comp):
        mem = np.flatnonzero(comp == c)
        refm = [u for u in mem if ref[u]]
        row = np.zeros(n)
        if refm:
            row[refm] = 1.0 / len(refm)
            is_ref[refm] = True
        else:
            f0 = mem[np.argmax(deg[mem])]
            row[f0] = 1.0
            is_ref[f0] = True
        rows.append(row)
    Ag = np.array(rows)
    wg = 1e3 * max(float(w.max()) if m else 1.0, 1.0)
    hw, x = np.ones(m), np.zeros(n)
    for _ in range(iters):
        W = np.concatenate([w * hw, np.full(len(Ag), wg)])
        sw = np.sqrt(W)
        Af, bf = np.vstack([A, Ag]), np.concatenate([b, np.zeros(len(Ag))])
        xn = np.linalg.lstsq(sw[:, None] * Af, sw * bf, rcond=None)[0]
        res = A @ xn - b
        if m:
            s = max(1.4826 * np.median(np.abs(res)), floor)
            hwn = np.where(np.abs(res) <= k_huber * s, 1.0, k_huber * s / np.maximum(np.abs(res), 1e-12))
        else:
            hwn = hw
        done = np.max(np.abs(xn - x)) < 1e-7
        x, hw = xn, hwn
        if done:
            break
    res = A @ x - b
    return dict(x=x, res=res, hw=hw, comp=comp, deg=deg, is_ref=is_ref,
                rms_before=float(np.sqrt(np.sum(w * b ** 2) / w.sum())) if m else np.nan,
                rms_after=float(np.sqrt(np.sum(w * res ** 2) / w.sum())) if m else np.nan)


# ----------------------------------------------------------------------------- merging into the mosaic
def group_merge(ids, d, fid, span_all, merge, corr_all=None):
    """Merge per-frame cell values (one per frame per cell) into one value per cell."""
    order = np.argsort(ids, kind='stable')
    ids, d, fid = ids[order], d[order].astype(np.float64), fid[order]
    sp = span_all[fid]
    uniq, start, cnt = np.unique(ids, return_index=True, return_counts=True)
    valid = np.isfinite(d)
    k = np.add.reduceat(valid.astype(np.int64), start)

    def agg(x):
        xv = np.where(valid, x, 0.0)
        s, s2 = np.add.reduceat(xv, start), np.add.reduceat(xv * xv, start)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.where(k > 0, s / k, np.nan)
            std = np.sqrt(np.maximum(np.where(k > 1, s2 / k - mean ** 2, np.nan), 0))
        if merge == 'mean':
            return mean, std
        xs = np.where(valid, x, np.nan)
        v = xs[np.lexsort((xs, ids))]            # NaNs last inside each group
        lo, hi = start + np.maximum(k - 1, 0) // 2, start + k // 2
        return np.where(k > 0, 0.5 * (v[lo] + v[hi]), np.nan), std

    cum, std = agg(d)
    rate, _ = agg(d / sp)
    smax = np.maximum.reduceat(np.where(valid, sp, -np.inf), start)
    smin = np.minimum.reduceat(np.where(valid, sp, np.inf), start)
    ssum = np.add.reduceat(np.where(valid, sp, 0.0), start)
    with np.errstate(invalid='ignore', divide='ignore'):
        span_mean = np.where(k > 0, ssum / k, np.nan)
    span_rng = np.where(k > 0, smax - smin, np.nan)
    out = dict(ids=uniq, cumulative_mm=cum, rate_mm_yr=rate, std_mm=std, n_frames=cnt,
               n_valid=k, span_mean_yr=span_mean, span_range_yr=span_rng)
    if corr_all is not None:
        dc = d - corr_all[fid]
        out['cumulative_corr_mm'] = agg(dc)[0]
        out['rate_corr_mm_yr'] = agg(dc / sp)[0]
    return out


class SparseWriter:
    DTYPES = dict(ix='i4', iy='i4', cumulative_mm='f4', rate_mm_yr='f4', std_mm='f4',
                  n_frames='i2', n_valid='i2', span_mean_yr='f4', span_range_yr='f4')

    def __init__(self, grp, nx_g, corrected=False):
        self.nx_g, self.n = nx_g, 0
        self.dtypes = dict(self.DTYPES)
        if corrected:
            self.dtypes.update(cumulative_corr_mm='f4', rate_corr_mm_yr='f4')
        self.ds = {k: grp.create_dataset(k, shape=(0,), maxshape=(None,), dtype=t, chunks=(65536,),
                                         compression='gzip', compression_opts=4, shuffle=True)
                   for k, t in self.dtypes.items()}

    def append(self, r):
        r = dict(r)
        ids = r.pop('ids')
        r['iy'], r['ix'] = ids // self.nx_g, ids % self.nx_g
        m = len(ids)
        for k, ds in self.ds.items():
            ds.resize(self.n + m, axis=0)
            ds[self.n:self.n + m] = np.asarray(r[k]).astype(self.dtypes[k])
        self.n += m


def merge_orbit(ks, scratch, span_all, corr_all, nx_g, band_rows, merge, writer):
    ids_mm = {k: np.load(f'{scratch}/ids_{k}.npy', mmap_mode='r') for k in ks}
    d_mm = {k: np.load(f'{scratch}/d_{k}.npy', mmap_mode='r') for k in ks}
    nonempty = [k for k in ks if len(ids_mm[k])]
    if not nonempty:
        return
    iy_min = min(int(ids_mm[k][0] // nx_g) for k in nonempty)
    iy_max = max(int(ids_mm[k][-1] // nx_g) for k in nonempty)
    for r0 in range(iy_min, iy_max + 1, band_rows):
        lo_id, hi_id = r0 * nx_g, (r0 + band_rows) * nx_g
        pi, pd, pf = [], [], []
        for k in nonempty:
            a, b = np.searchsorted(ids_mm[k], lo_id, 'left'), np.searchsorted(ids_mm[k], hi_id, 'left')
            if b > a:
                pi.append(np.asarray(ids_mm[k][a:b]))
                pd.append(np.asarray(d_mm[k][a:b]))
                pf.append(np.full(b - a, k, np.int32))
        if pi:
            writer.append(group_merge(np.concatenate(pi), np.concatenate(pd), np.concatenate(pf),
                                      span_all, merge, corr_all))


def write_str_or_num(grp, key, vals):
    if isinstance(vals[0], str):
        grp.create_dataset(key, data=np.array(vals, dtype=object), dtype=h5py.string_dtype())
    else:
        grp.create_dataset(key, data=np.array(vals))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', required=True, help='folder with the denoised .h5 files (searched recursively)')
    ap.add_argument('--out', default='slip_mosaic.h5')
    ap.add_argument('--merge', choices=['mean', 'median'], default='mean', help='how overlapping frames are combined')
    ap.add_argument('--res', type=float, default=None, help='cell size in degrees (default: native pixel size)')
    ap.add_argument('--n-avg', type=int, default=1, help='epochs averaged at each end (1 = literal last - first)')
    ap.add_argument('--span-flag-days', type=float, default=90., help='flag frames whose span differs from the median by more than this')
    ap.add_argument('--offset-mode', choices=['none', 'rate', 'cumulative'], default='none',
                    help='inter-frame offset correction: rate = option A, cumulative = option B')
    ap.add_argument('--ref-track', default=None, help='substring of the track label (e.g. D004) used as reference; its frames get zero mean offset')
    ap.add_argument('--min-overlap-cells', type=int, default=500, help='minimum overlap (cells) for a pair to be used')
    ap.add_argument('--min-window-yr', type=float, default=1.0, help='[rate] minimum common time window of a pair (years)')
    ap.add_argument('--b-max-span-diff-days', type=float, default=90., help='[cumulative] ignore pairs whose spans differ more than this')
    ap.add_argument('--band-deg', type=float, default=5., help='latitude band height used while merging (memory control)')
    ap.add_argument('--scratch', default=None, help='scratch folder (default: next to --out)')
    ap.add_argument('--keep-scratch', action='store_true')
    ap.add_argument('--limit', type=int, default=None, help='only the first N series (testing)')
    a = ap.parse_args()

    files = list_series(a.results_dir, a.out)[:a.limit]
    log(f'{len(files)} complete series found')
    if not files:
        raise SystemExit('nothing to do')

    if a.res is None:
        sp = []
        for p in files:
            try:
                with h5py.File(p, 'r') as f:
                    sp.append(native_spacing_deg(f))
            except Exception:
                log(f'spacing failed for {p}')
        res = round(float(np.nanmedian(sp)), 7)
        log(f'native pixel size: median {res:.6f} deg (min {np.nanmin(sp):.6f}, max {np.nanmax(sp):.6f})')
    else:
        res = a.res
    nx_g = int(np.ceil(360.0 / res))
    log(f'grid: cell = {res} deg, {nx_g} columns (global, sparse storage)')

    scratch = a.scratch or os.path.join(os.path.dirname(os.path.abspath(a.out)), '_scratch_mosaic')
    os.makedirs(scratch, exist_ok=True)
    err_log = os.path.splitext(a.out)[0] + '_errors.log'

    meta = []
    for p in files:
        t0 = time.time()
        try:
            with h5py.File(p, 'r') as f:
                orbit = attr_str(f.attrs.get('orbit', 'unknown')).lower()
                if orbit not in ORBITS:
                    log(f'{p}: unknown orbit "{orbit}", skipped')
                    continue
                dates = parse_dates(f['dates'][...])
                n = a.n_avg
                span = (dates[-n:].mean() - dates[:n].mean()) / 365.25
                ids, val, st = frame_cells(f, res, nx_g, n)
                k = len(meta)
                np.save(f'{scratch}/ids_{k}.npy', ids)
                np.save(f'{scratch}/d_{k}.npy', val)
                d0, d1 = (datetime.fromordinal(int(x)).strftime('%Y%m%d') for x in (dates[0], dates[-1]))
                meta.append(dict(key=attr_str(f.attrs.get('key', os.path.basename(p))),
                                 region=attr_str(f.attrs.get('region', '')), orbit=orbit,
                                 path=os.path.abspath(p),
                                 first_date=int(d0), last_date=int(d1),
                                 first_ord=int(dates[0]), last_ord=int(dates[-1]), span_yr=float(span),
                                 lon_min=st['lon_min'], lon_max=st['lon_max'],
                                 lat_min=st['lat_min'], lat_max=st['lat_max'],
                                 n_pixels=st['n_pixels'], frac_valid=st['frac_valid'],
                                 heading_deg=float(f.attrs.get('heading_deg', np.nan)),
                                 incidence_deg=float(f.attrs.get('incidence_deg', np.nan)),
                                 n_cells=len(ids)))
            log(f'[{len(meta)}/{len(files)}] {orbit[:4]} {meta[-1]["key"]} span {span:.2f} yr, '
                f'{len(ids)} cells, valid {st["frac_valid"]:.0%}, masked {st["frac_masked"]:.0%} '
                f'({time.time() - t0:.0f}s)')
            if st['frac_masked'] > 0.9 or (st['frac_nan_unmasked'] > 0.5):
                log('  WARNING: mask/NaN fractions look odd, check the mask convention for this file')
        except Exception:
            with open(err_log, 'a') as ef:
                ef.write(f'{p}\n{traceback.format_exc()}\n')
            log(f'FAILED {p} (see {err_log})')

    span_all = np.array([m['span_yr'] for m in meta])
    for m in meta:
        m['span_flag'] = 0
    for orbit in ORBITS:
        ks = [i for i, m in enumerate(meta) if m['orbit'] == orbit]
        if not ks:
            continue
        med = np.median(span_all[ks])
        flagged = [i for i in ks if abs(span_all[i] - med) > a.span_flag_days / 365.25]
        for i in flagged:
            meta[i]['span_flag'] = 1
        log(f'{orbit}: {len(ks)} frames, median span {med:.2f} yr, range {span_all[ks].min():.2f}-{span_all[ks].max():.2f} yr')
        for i in flagged:
            log(f'  SPAN FLAG: {meta[i]["key"]}  span {span_all[i]:.2f} yr ({meta[i]["first_date"]} -> {meta[i]["last_date"]})')

    # ---- inter-frame offsets
    mode = a.offset_mode
    corr_all = np.zeros(len(meta)) if mode != 'none' else None
    pair_rows, orbit_stats = [], {}
    if mode != 'none':
        unit = 'mm/yr' if mode == 'rate' else 'mm'
        for m in meta:
            m.update(offset_value=0.0, offset_corr_mm=0.0, offset_component=-1, offset_is_reference=0, offset_n_pairs=0)
        for orbit in ORBITS:
            ks = [i for i, m in enumerate(meta) if m['orbit'] == orbit]
            if len(ks) < 2:
                continue
            log(f'{orbit}: offset correction ({mode}), searching overlaps...')
            pairs = overlap_pairs(ks, meta, scratch, a.min_overlap_cells)
            log(f'  {len(pairs)} overlapping pairs (>= {a.min_overlap_cells} cells)')
            if mode == 'rate':
                window_fields(ks, meta, pairs, scratch, res, nx_g, a.min_window_yr)
            edges, used = [], []
            loc = {k: n for n, k in enumerate(ks)}
            for p in pairs:
                if mode == 'rate':
                    if not p['use']:
                        log(f'  pair {p["i"]}-{p["j"]} dropped: common window {p["T"]:.2f} yr < {a.min_window_yr}')
                        continue
                    o, nv = obs_rate(p, scratch)
                else:
                    dsp = abs(meta[p['i']]['span_yr'] - meta[p['j']]['span_yr']) * 365.25
                    if dsp > a.b_max_span_diff_days:
                        log(f'  pair {p["i"]}-{p["j"]} dropped: spans differ by {dsp:.0f} d > {a.b_max_span_diff_days}')
                        continue
                    o, nv = obs_cumulative(p, scratch)
                if nv < a.min_overlap_cells or not np.isfinite(o):
                    log(f'  pair {p["i"]}-{p["j"]} dropped: only {nv} valid overlapping cells')
                    continue
                edges.append((loc[p['i']], loc[p['j']], o, float(np.sqrt(nv))))
                used.append((p, o, nv))
            ref = np.array([a.ref_track is not None and a.ref_track in track_label(meta[k]['key']) for k in ks])
            if a.ref_track and not ref.any():
                log(f'  note: no {orbit} frame matches --ref-track "{a.ref_track}", using automatic references')
            sol = solve_offsets(len(ks), edges, ref)
            for n_, k in enumerate(ks):
                x = float(sol['x'][n_])
                meta[k].update(offset_value=x, offset_corr_mm=x * meta[k]['span_yr'] if mode == 'rate' else x,
                               offset_component=int(sol['comp'][n_]), offset_is_reference=int(sol['is_ref'][n_]),
                               offset_n_pairs=int(sol['deg'][n_]))
                corr_all[k] = meta[k]['offset_corr_mm']
            for (p, o, nv), r_, h_ in zip(used, sol['res'], sol['hw']):
                pair_rows.append(dict(frame_i=p['i'], frame_j=p['j'], key_i=meta[p['i']]['key'], key_j=meta[p['j']]['key'],
                                      orbit=orbit, n_cells=nv, window_yr=p.get('T', np.nan), obs=o, resid=float(r_), huber_w=float(h_)))
            orbit_stats[orbit] = (sol['rms_before'], sol['rms_after'], len(edges), int(sol['comp'].max() + 1))
            log(f'  {orbit}: {len(edges)} pairs used, {orbit_stats[orbit][3]} connected group(s) of frames; '
                f'weighted overlap RMS {sol["rms_before"]:.3f} -> {sol["rms_after"]:.3f} {unit}')
            for n_, k in enumerate(ks):
                tag = 'REF ' if sol['is_ref'][n_] else '    '
                warn = '  <- not connected to any other frame, left uncorrected' if sol['deg'][n_] == 0 else ''
                log(f'    {tag}{track_label(meta[k]["key"]):<28} offset {sol["x"][n_]:+8.3f} {unit}  '
                    f'(correction to cumulative {corr_all[k]:+8.2f} mm, {int(sol["deg"][n_])} pairs){warn}')
            bad = [(p, r_, h_) for (p, o, nv), r_, h_ in zip(used, sol['res'], sol['hw']) if h_ < 0.5]
            for p, r_, h_ in bad:
                log(f'  DOWN-WEIGHTED pair {track_label(meta[p["i"]]["key"])} / {track_label(meta[p["j"]]["key"])}: '
                    f'residual {r_:+.2f} {unit} (Huber weight {h_:.2f})')

    band_rows = max(1, int(round(a.band_deg / res)))
    with h5py.File(a.out, 'w') as out:
        for orbit in ORBITS:
            ks = [i for i, m in enumerate(meta) if m['orbit'] == orbit]
            if not ks:
                continue
            w = SparseWriter(out.create_group(orbit), nx_g, corrected=mode != 'none')
            merge_orbit(ks, scratch, span_all, corr_all, nx_g, band_rows, a.merge, w)
            out[orbit].attrs['n_cells'] = w.n
            out[orbit].attrs['n_frames'] = len(ks)
            log(f'{orbit}: {w.n} occupied cells written')
        fr = out.create_group('frames')
        for key in meta[0]:
            write_str_or_num(fr, key, [m[key] for m in meta])
        if pair_rows:
            pg = out.create_group('pairs')
            for key in pair_rows[0]:
                write_str_or_num(pg, key, [r[key] for r in pair_rows])
        attrs = dict(
            res_deg=res, lon0=LON0, lat0=LAT0, nx_global=nx_g, merge=a.merge, n_avg=a.n_avg,
            span_flag_days=a.span_flag_days, units_cumulative='mm', units_rate='mm/yr',
            convention='LOS displacement, positive away from the satellite, last epoch minus first epoch',
            nan_policy='cells covered by a frame but without valid pixels are kept with NaN values (n_valid = 0)',
            cell_definition='cell (ix, iy) spans lon [lon0+ix*res, lon0+(ix+1)*res), lat [lat0+iy*res, lat0+(iy+1)*res)',
            offset_mode=mode, ref_track=a.ref_track or '', min_overlap_cells=a.min_overlap_cells,
            source_dir=os.path.abspath(a.results_dir), created=datetime.now().isoformat())
        if mode != 'none':
            attrs['offset_unit'] = 'mm/yr' if mode == 'rate' else 'mm'
            attrs['offset_note'] = ('corrected = raw - offset_corr_mm per frame; rate: offset_corr_mm = offset_value * span_yr; '
                                    'cumulative: offset_corr_mm = offset_value. Offsets are relative to the reference frames.')
            for orbit, (rb, ra, ne, nc) in orbit_stats.items():
                attrs[f'rms_overlap_before_{orbit}'] = rb
                attrs[f'rms_overlap_after_{orbit}'] = ra
        out.attrs.update(attrs)
    if not a.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    log(f'saved {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
