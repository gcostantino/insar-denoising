#!/usr/bin/env python
"""Build a sparse global mosaic of LOS velocity and cumulative displacement (last - first) from the denoised FLATSIM
series (.h5), separately for ascending / descending, with optional inter-frame offset corrections.

Per frame: rate = regression slope over all epochs (mm/yr); cumulative = last - first epoch (mm);
detrended = cumulative - rate * span (mm), i.e. the departure of the net displacement from the linear trend
(= residual of the regression at the last minus at the first epoch). All binned to a global lon/lat grid
(default cell = native pixel size).

Frame errors in last - first:  r_i * span_i  (constant rate offset)  +  d_i  (nonlinear part: epoch-level constants).
  rate offsets r_i [mm/yr]: obs_ij = median over the overlap of (rate_i - rate_j)          ~ r_i - r_j
  level offsets d_i [mm]  : obs_ij = median over the overlap of (detrended_i - detrended_j) ~ d_i - d_j
  (the detrended field is blind to r_i, so the two systems are independent.)
--offset-mode none : raw mosaic only
--offset-mode rate : r_i only.   rate_corr = rate - r_i ; cumulative_corr = cumulative - r_i*span_i
--offset-mode full : r_i and d_i. rate_corr as above ; cumulative_corr = cumulative - r_i*span_i - d_i ;
                     detrended_corr = detrended - d_i
The real trend is NOT removed before the solves: it is common to both frames of an overlap and cancels.

Weights 1/var_ij, var_ij = sigma_sys^2 + s_ij^2 / n_eff, s_ij = 1.4826 MAD of the overlap differences,
n_eff = n_cells / corr_cells; sigma_sys is smaller for same-track pairs (same geometry and acquisitions).
One weighted linear least-squares solve per orbit and per quantity (minimum-norm, exact gauge per connected group
of frames: zero mean over the frames of --ref-track, or over the group if it has none). No edge trimming.
--reject-z > 0 optionally drops edges with |normalized residual| above the threshold once and re-solves.

Output .h5: per orbit sparse cells (raw fields and, if corrected, *_corr fields), /frames table, /pairs table.
Units: mm and mm/yr, positive away from the satellite.

Example:
python build_disp_mosaic.py --results-dir results/ --out slip_mosaic.h5 --merge mean --offset-mode full --ref-track D004
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
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

LON0, LAT0 = -180.0, -90.0     # global grid origin; cell (ix, iy) spans [LON0+ix*res, LON0+(ix+1)*res)
ORBITS = ('ascending', 'descending')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ----------------------------------------------------------------------------- small helpers
def parse_dates(raw):
    out = []
    for x in np.asarray(raw).ravel():
        s = x.decode() if isinstance(x, (bytes, np.bytes_)) else x if isinstance(x, str) else str(int(x))
        out.append(datetime.strptime(s.strip()[:8], '%Y%m%d').toordinal())
    return np.array(out)


def attr_str(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def track_label(key):
    m = re.search(r'S1_(.+?)-VV', key)
    return m.group(1) if m else key.split('/')[-1]


def track_id(label):
    m = re.match(r'[AD]\d+', label)
    return m.group(0) if m else label


def list_series(results_dir, out_path):
    keep = []
    for p in sorted(glob.glob(os.path.join(results_dir, '**', '*.h5'), recursive=True)):
        if os.path.basename(p).startswith('master') or os.path.abspath(p) == os.path.abspath(out_path):
            continue
        try:
            with h5py.File(p, 'r') as f:
                if bool(f.attrs.get('complete', 0)):
                    keep.append(p)
        except OSError:
            log(f'cannot open {p}, skipped')
    return keep


def valid_geometry(lon, lat):
    """Finite lon/lat, excluding the (0, 0) fill value found outside the footprint."""
    return np.isfinite(lon) & np.isfinite(lat) & ~((lon == 0) & (lat == 0))


def native_spacing_deg(f, win=200):
    """Pixel size in degrees (geometric mean of the median pixel steps, longitude scaled by cos(lat))."""
    ny, nx = f['lon'].shape
    y0, x0 = max(0, ny // 2 - win // 2), max(0, nx // 2 - win // 2)
    lo = f['lon'][y0:y0 + win, x0:x0 + win].astype(float)
    la = f['lat'][y0:y0 + win, x0:x0 + win].astype(float)
    bad = ~valid_geometry(lo, la)
    lo[bad], la[bad] = np.nan, np.nan
    c = np.cos(np.deg2rad(np.nanmean(la)))
    sx = np.nanmedian(np.hypot(np.diff(lo, axis=1) * c, np.diff(la, axis=1)))
    sy = np.nanmedian(np.hypot(np.diff(lo, axis=0) * c, np.diff(la, axis=0)))
    return float(np.sqrt(sx * sy))


def pixel_cells(f, res, nx_g):
    """Sorted unique cell ids, pixel->cell index (valid-geometry pixels), valid-geometry mask, masked-out mask, bbox."""
    lon, lat = f['lon'][...].astype(np.float64), f['lat'][...].astype(np.float64)
    lon = (lon + 180.0) % 360.0 - 180.0
    ok = valid_geometry(lon, lat)
    masked = f['mask'][...].astype(bool)
    ix = np.floor((lon[ok] - LON0) / res).astype(np.int64)
    iy = np.floor((lat[ok] - LAT0) / res).astype(np.int64)
    uniq, inv = np.unique(iy * nx_g + ix, return_inverse=True)
    good = ok & ~masked
    if not good.any():
        good = ok
    bbox = (float(lon[good].min()), float(lon[good].max()), float(lat[good].min()), float(lat[good].max()))
    return uniq, inv, ok, masked, bbox


def bin_to_cells(field, inv, ok, masked, ncell):
    """Mean of the valid pixels of a (ny, nx) field in every cell (NaN if none)."""
    v = np.array(field, dtype=np.float64)
    v[masked] = np.nan
    v = v[ok]
    valid = np.isfinite(v)
    s = np.bincount(inv, weights=np.where(valid, v, 0.0), minlength=ncell)
    c = np.bincount(inv, weights=valid, minlength=ncell)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, s / c, np.nan).astype(np.float32)


# ----------------------------------------------------------------------------- per-frame products
def frame_products(f, dates_ord, res, nx_g, n_avg, block_elems):
    """rate (regression slope, mm/yr) and cumulative (last - first, mm) binned to cells."""
    U = f['displacement']
    T, ny, nx = U.shape
    yrs = (dates_ord - dates_ord.mean()) / 365.25
    g = (yrs / (yrs ** 2).sum()).astype(np.float32)           # slope = sum_t g_t U_t
    rate = np.empty((ny, nx), np.float32)
    cum = np.empty((ny, nx), np.float32)
    rb = max(1, int(block_elems // max(T * nx, 1)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for r0 in range(0, ny, rb):
            blk = U[:, r0:r0 + rb, :]
            rate[r0:r0 + rb] = np.tensordot(g, blk, axes=(0, 0))
            cum[r0:r0 + rb] = np.nanmean(blk[-n_avg:], axis=0) - np.nanmean(blk[:n_avg], axis=0)
    uniq, inv, ok, masked, bbox = pixel_cells(f, res, nx_g)
    r_c = bin_to_cells(rate, inv, ok, masked, len(uniq))
    c_c = bin_to_cells(cum, inv, ok, masked, len(uniq))
    dm = rate.copy()
    dm[masked] = np.nan
    stats = dict(lon_min=bbox[0], lon_max=bbox[1], lat_min=bbox[2], lat_max=bbox[3],
                 n_pixels=int(ok.sum()), frac_masked=float(masked.mean()),
                 frac_valid=float(np.isfinite(dm[ok]).sum() / max(ok.sum(), 1)))
    return uniq, r_c, c_c, stats


# ----------------------------------------------------------------------------- pair observations
def _robust(dd, n, corr_cells, sigma_sys):
    med = float(np.median(dd))
    s = float(1.4826 * np.median(np.abs(dd - med)))
    return med, s, sigma_sys ** 2 + s ** 2 / max(n / corr_cells, 1.0)


def pair_observations(ks, meta, scratch, a):
    """For every overlapping pair of frames (same orbit): medians of the rate and detrended differences."""
    rows = []
    for ia_, i in enumerate(ks):
        for j in ks[ia_ + 1:]:
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
            if len(common) < a.min_overlap_cells:
                continue
            dsp = abs(mi['span_yr'] - mj['span_yr']) * 365.25
            if dsp > a.max_span_diff_days:
                log(f' pair {i}-{j} dropped: spans differ by {dsp:.0f} d > {a.max_span_diff_days}')
                continue

            def load(name, k, s, idx):
                return np.asarray(np.load(f'{scratch}/{name}_{k}.npy', mmap_mode='r')[s])[idx].astype(np.float64)
            ri, rj = load('rate', i, si, ia), load('rate', j, sj, ja)
            ci, cj = load('cum', i, si, ia), load('cum', j, sj, ja)
            di, dj = ci - ri * mi['span_yr'], cj - rj * mj['span_yr']
            v = np.isfinite(ri) & np.isfinite(rj) & np.isfinite(ci) & np.isfinite(cj)
            n = int(v.sum())
            if n < a.min_overlap_cells:
                log(f' pair {i}-{j} dropped: only {n} valid overlapping cells')
                continue
            same = track_id(track_label(mi['key'])) == track_id(track_label(mj['key']))
            obs, s, var = _robust(ri[v] - rj[v], n, a.corr_cells, a.sigma_sys_same_track if same else a.sigma_sys)
            obs_d, s_d, var_d = _robust(di[v] - dj[v], n, a.corr_cells,
                                        a.sigma_sys_det_same_track if same else a.sigma_sys_det)
            rows.append(dict(i=i, j=j, n=n, obs=obs, s=s, var=var, obs_det=obs_d, s_det=s_d, var_det=var_d,
                             same_track=int(same)))
    return rows


# ----------------------------------------------------------------------------- linear solve
def solve_offsets(n, ea, eb, obs, var, ref):
    """x_a - x_b = obs with weights 1/var. Minimum-norm solution per connected group, zero-mean gauge
    (over the reference frames of the group if any, else over the group)."""
    m = len(ea)
    w = 1.0 / var if m else np.zeros(0)
    if m:
        ncomp, comp = connected_components(sp.coo_matrix((np.ones(m), (ea, eb)), shape=(n, n)), directed=False)
    else:
        ncomp, comp = n, np.arange(n)
    A = np.zeros((m, n))
    A[np.arange(m), ea], A[np.arange(m), eb] = 1.0, -1.0
    L, rhs = A.T @ (w[:, None] * A), A.T @ (w * obs)
    x, is_ref = np.zeros(n), np.zeros(n, bool)
    for c in range(ncomp):
        idx = np.flatnonzero(comp == c)
        if len(idx) == 1:
            continue
        xc = np.linalg.lstsq(L[np.ix_(idx, idx)], rhs[idx], rcond=None)[0]
        xc -= xc.mean()
        rc = ref[idx]
        if rc.any():
            xc -= xc[rc].mean()
            is_ref[idx[rc]] = True
        x[idx] = xc
    res = A @ x - obs
    deg = np.bincount(ea, minlength=n) + np.bincount(eb, minlength=n)
    dof = m - (n - ncomp)                                    # independent cycles
    out = dict(x=x, res=res, z=res / np.sqrt(var) if m else res, comp=comp, deg=deg, is_ref=is_ref,
               n_cycles=int(dof), n_groups=int(ncomp))
    if m:
        out['rms_before'] = float(np.sqrt(np.sum(w * obs ** 2) / w.sum()))
        out['rms_after'] = float(np.sqrt(np.sum(w * res ** 2) / w.sum()))
        out['chi2_per_cycle'] = float(np.sum(out['z'] ** 2) / dof) if dof > 0 else np.nan
    return out


def solve_system(n, ea, eb, obs, var, ref, reject_z):
    """solve_offsets, optionally dropping once the edges with |z| > reject_z. Returns sol and the 'used' mask."""
    sol = solve_offsets(n, ea, eb, obs, var, ref)
    used = np.ones(len(ea), bool)
    if reject_z > 0 and len(ea):
        used = np.abs(sol['z']) <= reject_z
        if not used.all():
            log(f' rejecting {int((~used).sum())} edge(s) with |z| > {reject_z}')
            sol = solve_offsets(n, ea[used], eb[used], obs[used], var[used], ref)
            sol['res'] = sol['x'][ea] - sol['x'][eb] - obs
            sol['z'] = sol['res'] / np.sqrt(var)
    return sol, used


# ----------------------------------------------------------------------------- merging
def grouped(ids, vals, merge):
    """Per-cell mean/median and std of vals (NaN ignored)."""
    order = np.lexsort((vals, ids))                          # NaN sorted last inside each cell
    ids, v = ids[order], vals[order].astype(np.float64)
    uniq, start, cnt = np.unique(ids, return_index=True, return_counts=True)
    valid = np.isfinite(v)
    k = np.add.reduceat(valid.astype(np.int64), start)
    xv = np.where(valid, v, 0.0)
    s, s2 = np.add.reduceat(xv, start), np.add.reduceat(xv * xv, start)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(k > 0, s / k, np.nan)
        std = np.sqrt(np.maximum(np.where(k > 1, s2 / k - mean ** 2, np.nan), 0))
    if merge == 'median':
        lo, hi = start + np.maximum(k - 1, 0) // 2, start + k // 2
        mean = np.where(k > 0, 0.5 * (v[lo] + v[hi]), np.nan)
    return uniq, mean, std, k, cnt


class SparseWriter:
    DTYPES = dict(ix='i4', iy='i4', rate_mm_yr='f4', cumulative_mm='f4', detrended_mm='f4', rate_std_mm_yr='f4',
                  n_frames='i2', n_valid='i2', span_mean_yr='f4', span_range_yr='f4')

    def __init__(self, grp, nx_g, mode):
        self.nx_g, self.n = nx_g, 0
        self.dtypes = dict(self.DTYPES)
        if mode in ('rate', 'full'):
            self.dtypes.update(rate_corr_mm_yr='f4', cumulative_corr_mm='f4')
        if mode == 'full':
            self.dtypes.update(detrended_corr_mm='f4')
        self.ds = {k: grp.create_dataset(k, shape=(0,), maxshape=(None,), dtype=t, chunks=(65536,),
                                         compression='gzip', compression_opts=4, shuffle=True)
                   for k, t in self.dtypes.items()}

    def append(self, r):
        ids = r.pop('ids')
        r['iy'], r['ix'] = ids // self.nx_g, ids % self.nx_g
        m = len(ids)
        for k, ds in self.ds.items():
            ds.resize(self.n + m, axis=0)
            ds[self.n:self.n + m] = np.asarray(r[k]).astype(self.dtypes[k])
        self.n += m


def merge_orbit(ks, scratch, span, off_rate, off_cum, off_det, nx_g, band_rows, merge, writer, mode):
    """off_rate [mm/yr], off_cum [mm] (total correction of last - first), off_det [mm], all indexed by frame."""
    ids_mm = {k: np.load(f'{scratch}/ids_{k}.npy', mmap_mode='r') for k in ks}
    rate_mm = {k: np.load(f'{scratch}/rate_{k}.npy', mmap_mode='r') for k in ks}
    cum_mm = {k: np.load(f'{scratch}/cum_{k}.npy', mmap_mode='r') for k in ks}
    nonempty = [k for k in ks if len(ids_mm[k])]
    if not nonempty:
        return
    iy_min = min(int(ids_mm[k][0] // nx_g) for k in nonempty)
    iy_max = max(int(ids_mm[k][-1] // nx_g) for k in nonempty)
    for r0 in range(iy_min, iy_max + 1, band_rows):
        lo_id, hi_id = r0 * nx_g, (r0 + band_rows) * nx_g
        pi, pr, pc, pf = [], [], [], []
        for k in nonempty:
            a, b = np.searchsorted(ids_mm[k], lo_id, 'left'), np.searchsorted(ids_mm[k], hi_id, 'left')
            if b > a:
                pi.append(np.asarray(ids_mm[k][a:b]))
                pr.append(np.asarray(rate_mm[k][a:b]).astype(np.float64))
                pc.append(np.asarray(cum_mm[k][a:b]).astype(np.float64))
                pf.append(np.full(b - a, k, np.int32))
        if not pi:
            continue
        ids, rate, cum, fid = map(np.concatenate, (pi, pr, pc, pf))
        sp_ = span[fid]
        det = cum - rate * sp_
        uniq, r_m, r_s, k_valid, cnt = grouped(ids, rate, merge)
        order = np.argsort(ids, kind='stable')
        start = np.unique(ids[order], return_index=True)[1]
        sps = sp_[order]
        out = dict(ids=uniq, rate_mm_yr=r_m, rate_std_mm_yr=r_s, n_valid=k_valid, n_frames=cnt,
                   cumulative_mm=grouped(ids, cum, merge)[1], detrended_mm=grouped(ids, det, merge)[1],
                   span_mean_yr=np.add.reduceat(sps, start) / cnt,
                   span_range_yr=np.maximum.reduceat(sps, start) - np.minimum.reduceat(sps, start))
        if mode in ('rate', 'full'):
            out['rate_corr_mm_yr'] = grouped(ids, rate - off_rate[fid], merge)[1]
            out['cumulative_corr_mm'] = grouped(ids, cum - off_cum[fid], merge)[1]
        if mode == 'full':
            out['detrended_corr_mm'] = grouped(ids, det - off_det[fid], merge)[1]
        writer.append(out)


def write_table(grp, key, vals):
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
    ap.add_argument('--n-avg', type=int, default=1, help='epochs averaged at each end of the cumulative (1 = literal)')
    ap.add_argument('--offset-mode', choices=['none', 'rate', 'full'], default='none',
                    help='none | rate (r_i only) | full (r_i and the level offsets d_i of the detrended field)')
    ap.add_argument('--ref-track', default=None, help='substring of the track label (e.g. D004) used as reference')
    ap.add_argument('--min-overlap-cells', type=int, default=500)
    ap.add_argument('--max-span-diff-days', type=float, default=90., help='ignore pairs whose spans differ more')
    ap.add_argument('--sigma-sys', type=float, default=0.3, help='systematic std of a cross-track pair, rate (mm/yr)')
    ap.add_argument('--sigma-sys-same-track', type=float, default=0.1, help='same, same-track pairs (mm/yr)')
    ap.add_argument('--sigma-sys-det', type=float, default=2.0, help='systematic std, detrended field, cross-track (mm)')
    ap.add_argument('--sigma-sys-det-same-track', type=float, default=0.7, help='same, same-track pairs (mm)')
    ap.add_argument('--corr-cells', type=float, default=2500., help='cells per independent sample in an overlap')
    ap.add_argument('--reject-z', type=float, default=0., help='>0: drop edges with |z| above this once, re-solve')
    ap.add_argument('--block-elems', type=float, default=1e7, help='elements per block when reading a series')
    ap.add_argument('--band-deg', type=float, default=5., help='latitude band height used while merging')
    ap.add_argument('--scratch', default=None)
    ap.add_argument('--keep-scratch', action='store_true')
    ap.add_argument('--limit', type=int, default=None, help='only the first N series (testing)')
    a = ap.parse_args()

    files = list_series(a.results_dir, a.out)[:a.limit]
    log(f'{len(files)} complete series found')
    if not files:
        raise SystemExit('nothing to do')
    if a.res is None:
        sp_ = []
        for p in files:
            try:
                with h5py.File(p, 'r') as f:
                    sp_.append(native_spacing_deg(f))
            except Exception:
                log(f'spacing failed for {p}')
        res = round(float(np.nanmedian(sp_)), 7)
        log(f'native pixel size: median {res:.6f} deg (min {np.nanmin(sp_):.6f}, max {np.nanmax(sp_):.6f})')
    else:
        res = a.res
    nx_g = int(np.ceil(360.0 / res))
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
                ids, rate, cum, st = frame_products(f, dates, res, nx_g, n, a.block_elems)
                k = len(meta)
                np.save(f'{scratch}/ids_{k}.npy', ids)
                np.save(f'{scratch}/rate_{k}.npy', rate)
                np.save(f'{scratch}/cum_{k}.npy', cum)
                meta.append(dict(key=attr_str(f.attrs.get('key', os.path.basename(p))),
                                 region=attr_str(f.attrs.get('region', '')), orbit=orbit, path=os.path.abspath(p),
                                 first_date=int(datetime.fromordinal(int(dates[0])).strftime('%Y%m%d')),
                                 last_date=int(datetime.fromordinal(int(dates[-1])).strftime('%Y%m%d')),
                                 span_yr=float(span), lon_min=st['lon_min'], lon_max=st['lon_max'],
                                 lat_min=st['lat_min'], lat_max=st['lat_max'], n_pixels=st['n_pixels'],
                                 frac_valid=st['frac_valid'], n_cells=len(ids)))
                log(f'[{len(meta)}/{len(files)}] {orbit[:4]} {meta[-1]["key"]} span {span:.2f} yr, {len(ids)} cells, '
                    f'valid {st["frac_valid"]:.0%}, masked {st["frac_masked"]:.0%} ({time.time() - t0:.0f}s)')
        except Exception:
            with open(err_log, 'a') as ef:
                ef.write(f'{p}\n{traceback.format_exc()}\n')
            log(f'FAILED {p} (see {err_log})')
    if not meta:
        raise SystemExit('no usable frame')

    mode = a.offset_mode
    span_all = np.array([m['span_yr'] for m in meta])
    off_rate, off_det = np.zeros(len(meta)), np.zeros(len(meta))
    for m in meta:
        m.update(offset_rate_mm_yr=0.0, offset_det_mm=0.0, offset_cum_mm=0.0, offset_group=-1,
                 offset_is_reference=0, offset_n_pairs=0)
    pair_rows, orbit_stats = [], {}
    if mode != 'none':
        for orbit in ORBITS:
            ks = [i for i, m in enumerate(meta) if m['orbit'] == orbit]
            if len(ks) < 2:
                continue
            log(f'{orbit}: searching overlaps...')
            rows = pair_observations(ks, meta, scratch, a)
            loc = {k: n for n, k in enumerate(ks)}
            ref = np.array([a.ref_track is not None and a.ref_track in track_label(meta[k]['key']) for k in ks])
            if a.ref_track and not ref.any():
                log(f' note: no {orbit} frame matches --ref-track "{a.ref_track}", using zero mean per group')
            ea = np.array([loc[r['i']] for r in rows], int)
            eb = np.array([loc[r['j']] for r in rows], int)
            systems = {'rate': ('obs', 'var', 'mm/yr')}
            if mode == 'full':
                systems['det'] = ('obs_det', 'var_det', 'mm')
            sols = {}
            for name, (ko, kv, unit) in systems.items():
                obs = np.array([r[ko] for r in rows])
                var = np.array([r[kv] for r in rows])
                sol, used = solve_system(len(ks), ea, eb, obs, var, ref, a.reject_z)
                sols[name] = (sol, used, obs, var)
                orbit_stats[(orbit, name)] = (sol.get('rms_before', np.nan), sol.get('rms_after', np.nan), len(rows),
                                              sol['n_groups'], sol['n_cycles'], sol.get('chi2_per_cycle', np.nan))
                log(f' {orbit} [{name}]: {len(rows)} pairs, {sol["n_groups"]} group(s), {sol["n_cycles"]} cycle(s); '
                    f'weighted RMS {orbit_stats[(orbit, name)][0]:.3f} -> {orbit_stats[(orbit, name)][1]:.3f} {unit}, '
                    f'chi2/cycle {orbit_stats[(orbit, name)][5]:.2f}')
                for r, z_ in zip(rows, sol['z']):
                    if abs(z_) > 4:
                        log(f' LARGE RESIDUAL [{name}] {track_label(meta[r["i"]]["key"])} / '
                            f'{track_label(meta[r["j"]]["key"])}: z = {z_:+.1f}')
            sr = sols['rate'][0]
            sd = sols['det'][0] if 'det' in sols else None
            for n_, k in enumerate(ks):
                r_i = float(sr['x'][n_])
                d_i = float(sd['x'][n_]) if sd is not None else 0.0
                off_rate[k], off_det[k] = r_i, d_i
                meta[k].update(offset_rate_mm_yr=r_i, offset_det_mm=d_i, offset_cum_mm=r_i * meta[k]['span_yr'] + d_i,
                               offset_group=int(sr['comp'][n_]), offset_is_reference=int(sr['is_ref'][n_]),
                               offset_n_pairs=int(sr['deg'][n_]))
                tag = 'REF ' if sr['is_ref'][n_] else '    '
                warn = ' <- not connected, left uncorrected' if sr['deg'][n_] == 0 else ''
                log(f' {tag}{track_label(meta[k]["key"]):<28} rate offset {r_i:+7.3f} mm/yr'
                    + (f', level offset {d_i:+7.2f} mm' if sd is not None else '')
                    + f' (cumulative {meta[k]["offset_cum_mm"]:+8.2f} mm, {int(sr["deg"][n_])} pairs){warn}')
            for e_, r in enumerate(rows):
                row = dict(frame_i=r['i'], frame_j=r['j'], key_i=meta[r['i']]['key'], key_j=meta[r['j']]['key'],
                           orbit=orbit, n_cells=r['n'], same_track=r['same_track'])
                for name, (sol, used, obs, var) in sols.items():
                    row.update({f'obs_{name}': float(obs[e_]), f'sigma_{name}': float(np.sqrt(var[e_])),
                                f'resid_{name}': float(sol['res'][e_]), f'z_{name}': float(sol['z'][e_]),
                                f'used_{name}': int(used[e_])})
                pair_rows.append(row)
    off_cum = off_rate * span_all + off_det

    band_rows = max(1, int(round(a.band_deg / res)))
    with h5py.File(a.out, 'w') as out:
        for orbit in ORBITS:
            ks = [i for i, m in enumerate(meta) if m['orbit'] == orbit]
            if not ks:
                continue
            w = SparseWriter(out.create_group(orbit), nx_g, mode)
            merge_orbit(ks, scratch, span_all, off_rate, off_cum, off_det, nx_g, band_rows, a.merge, w, mode)
            out[orbit].attrs['n_cells'] = w.n
            out[orbit].attrs['n_frames'] = len(ks)
            log(f'{orbit}: {w.n} occupied cells written')
        fr = out.create_group('frames')
        for key in meta[0]:
            write_table(fr, key, [m[key] for m in meta])
        if pair_rows:
            pg = out.create_group('pairs')
            for key in pair_rows[0]:
                write_table(pg, key, [r[key] for r in pair_rows])
        out.attrs.update(dict(
            res_deg=res, lon0=LON0, lat0=LAT0, nx_global=nx_g, merge=a.merge, n_avg=a.n_avg,
            units_cumulative='mm', units_rate='mm/yr',
            convention=('LOS, positive away from the satellite; rate = regression slope, cumulative = last - first, '
                        'detrended = cumulative - rate * span'),
            cell_definition='cell (ix, iy) spans lon [lon0+ix*res, lon0+(ix+1)*res), lat [lat0+iy*res, lat0+(iy+1)*res)',
            geometry_policy='pixels with non-finite lon/lat or lon=lat=0 are ignored; frame boxes from unmasked pixels',
            offset_mode=mode, ref_track=a.ref_track or '', min_overlap_cells=a.min_overlap_cells,
            sigma_sys=a.sigma_sys, sigma_sys_same_track=a.sigma_sys_same_track, sigma_sys_det=a.sigma_sys_det,
            sigma_sys_det_same_track=a.sigma_sys_det_same_track, corr_cells=a.corr_cells,
            reject_z=a.reject_z, source_dir=os.path.abspath(a.results_dir), created=datetime.now().isoformat()))
        if mode != 'none':
            out.attrs['offset_note'] = ('error of last - first = rate_offset*span + level_offset. rate: corrected = raw - '
                                        'rate_offset. cumulative: corrected = raw - offset_cum_mm (rate_offset*span, plus '
                                        'level_offset in mode full). detrended (mode full): corrected = raw - level_offset. '
                                        'Relative to the reference frames or zero-mean per group.')
            for (orbit, name), (rb, ra, ne, ng, nc, chi) in orbit_stats.items():
                out.attrs.update({f'rms_overlap_before_{name}_{orbit}': rb, f'rms_overlap_after_{name}_{orbit}': ra,
                                  f'n_pairs_{orbit}': ne, f'n_groups_{orbit}': ng, f'n_cycles_{name}_{orbit}': nc,
                                  f'chi2_per_cycle_{name}_{orbit}': chi})
    if not a.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    log(f'saved {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
