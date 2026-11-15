#!/usr/bin/env python
"""Build a sparse global mosaic of cumulative LOS displacement (last - first epoch)
from the denoised FLATSIM series (.h5), separately for ascending / descending.

Output: one .h5 with, per orbit, the occupied grid cells only (sparse), plus a per-frame table
(including each frame's bounding box, used by the plot script to draw region footprints).
Units: mm (cumulative), mm/yr (rate). Positive = away from the satellite (as in the inputs).

Example:
  python build_disp_mosaic.py --results-dir results/<tag> --out disp_mosaic.h5 --merge median
"""
import argparse
import glob
import os
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


def frame_cells(f, res, nx_g, n_avg):
    """Per-frame sparse result: sorted cell ids, cumulative displacement per cell (NaN if no valid pixel)."""
    U = f['displacement']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        last = np.nanmean(U[-n_avg:], axis=0)
        first = np.nanmean(U[:n_avg], axis=0)
    d = (last - first).astype(np.float32)
    masked = f['mask'][...].astype(bool)
    d[masked] = np.nan
    lon, lat = f['lon'][...].astype(np.float64), f['lat'][...].astype(np.float64)
    lon = (lon + 180.0) % 360.0 - 180.0
    ok = np.isfinite(lon) & np.isfinite(lat)
    ix = np.floor((lon[ok] - LON0) / res).astype(np.int64)
    iy = np.floor((lat[ok] - LAT0) / res).astype(np.int64)
    ids, dv = iy * nx_g + ix, d[ok]
    uniq, inv = np.unique(ids, return_inverse=True)
    valid = np.isfinite(dv)
    s = np.bincount(inv, weights=np.where(valid, dv, 0.0), minlength=len(uniq))
    c = np.bincount(inv, weights=valid, minlength=len(uniq))
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(c > 0, s / c, np.nan).astype(np.float32)
    stats = dict(lon_min=float(lon[ok].min()), lon_max=float(lon[ok].max()),
                 lat_min=float(lat[ok].min()), lat_max=float(lat[ok].max()),
                 n_pixels=int(ok.sum()), frac_masked=float(masked.mean()),
                 frac_nan_unmasked=float(np.isnan(d[~masked]).mean()) if (~masked).any() else np.nan,
                 frac_valid=float(valid.sum() / max(ok.sum(), 1)))
    return uniq, val, stats


def group_merge(ids, d, fid, span_all, merge):
    """Merge per-frame cell values (one per frame per cell) into one value per cell."""
    order = np.argsort(ids, kind='stable')
    ids, d, sp = ids[order], d[order].astype(np.float64), span_all[fid[order]]
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
    return dict(ids=uniq, cumulative_mm=cum, rate_mm_yr=rate, std_mm=std, n_frames=cnt,
                n_valid=k, span_mean_yr=span_mean, span_range_yr=span_rng)


class SparseWriter:
    DTYPES = dict(ix='i4', iy='i4', cumulative_mm='f4', rate_mm_yr='f4', std_mm='f4',
                  n_frames='i2', n_valid='i2', span_mean_yr='f4', span_range_yr='f4')

    def __init__(self, grp, nx_g):
        self.nx_g, self.n = nx_g, 0
        self.ds = {k: grp.create_dataset(k, shape=(0,), maxshape=(None,), dtype=t, chunks=(65536,),
                                         compression='gzip', compression_opts=4, shuffle=True)
                   for k, t in self.DTYPES.items()}

    def append(self, r):
        r = dict(r)
        ids = r.pop('ids')
        r['iy'], r['ix'] = ids // self.nx_g, ids % self.nx_g
        m = len(ids)
        for k, ds in self.ds.items():
            ds.resize(self.n + m, axis=0)
            ds[self.n:self.n + m] = np.asarray(r[k]).astype(self.DTYPES[k])
        self.n += m


def merge_orbit(ks, scratch, span_all, nx_g, band_rows, merge, writer):
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
                                      span_all, merge))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', required=True, help='folder with the denoised .h5 files (searched recursively)')
    ap.add_argument('--out', default='disp_mosaic.h5')
    ap.add_argument('--merge', choices=['mean', 'median'], default='mean', help='how overlapping frames are combined')
    ap.add_argument('--res', type=float, default=None, help='cell size in degrees (default: native pixel size)')
    ap.add_argument('--n-avg', type=int, default=1, help='epochs averaged at each end (1 = literal last - first)')
    ap.add_argument('--span-flag-days', type=float, default=90., help='flag frames whose span differs from the median by more than this')
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
                                 first_date=int(d0), last_date=int(d1), span_yr=float(span),
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

    band_rows = max(1, int(round(a.band_deg / res)))
    with h5py.File(a.out, 'w') as out:
        for orbit in ORBITS:
            ks = [i for i, m in enumerate(meta) if m['orbit'] == orbit]
            if not ks:
                continue
            w = SparseWriter(out.create_group(orbit), nx_g)
            merge_orbit(ks, scratch, span_all, nx_g, band_rows, a.merge, w)
            out[orbit].attrs['n_cells'] = w.n
            out[orbit].attrs['n_frames'] = len(ks)
            log(f'{orbit}: {w.n} occupied cells written')
        fr = out.create_group('frames')
        for key in meta[0]:
            vals = [m[key] for m in meta]
            if isinstance(vals[0], str):
                fr.create_dataset(key, data=np.array(vals, dtype=object), dtype=h5py.string_dtype())
            else:
                fr.create_dataset(key, data=np.array(vals))
        out.attrs.update(dict(
            res_deg=res, lon0=LON0, lat0=LAT0, nx_global=nx_g, merge=a.merge, n_avg=a.n_avg,
            span_flag_days=a.span_flag_days, units_cumulative='mm', units_rate='mm/yr',
            convention='LOS displacement, positive away from the satellite, last epoch minus first epoch',
            nan_policy='cells covered by a frame but without valid pixels are kept with NaN values (n_valid = 0)',
            cell_definition='cell (ix, iy) spans lon [lon0+ix*res, lon0+(ix+1)*res), lat [lat0+iy*res, lat0+(iy+1)*res)',
            source_dir=os.path.abspath(a.results_dir), created=datetime.now().isoformat()))
    if not a.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    log(f'saved {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
