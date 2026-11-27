#!/usr/bin/env python
"""
Write the RAW FLATSIM series (no denoising) in the same HDF5 layout as the denoised results, so that every analysis
script (solve_frame_offsets.py, overlap_diff_series.py, build_disp_mosaic.py ...) can be run on them unchanged.

Same loader, thresholds and mask as the denoising run (RMS and absolute-value screening, empty pixels masked, units
mm, positive away from the satellite). NaNs stay NaN: nothing is filled, nothing is reconstructed, no velocity
correction. 'displacement' is stored as loaded (FLATSIM sets each pixel to zero at its first defined date).

Layout: <out-dir>/<region>/<product>.h5 with displacement, dates, lon, lat, mask, topography, los, attrs (orbit, key,
region, ...) and attrs complete=1 written last (atomic rename), like the denoised files.

Example:
  python export_raw_h5.py --out-dir results_raw --match TURQUIE LEVANT --exclude NSBAS_DAUX
Run it from the folder of the denoising code (needs utils.flatsim_utils).
"""
import argparse
import os
import time
import traceback

import h5py
import numpy as np

from utils.flatsim_utils import load_flatsim

WVL, RMS_THRESH, ABS_THRESH, POSITIVE_AWAY = 55.5, 10., 300., True      # same as flatsim_denoising.py


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def is_complete(path):
    try:
        with h5py.File(path, 'r') as f:
            return bool(f.attrs.get('complete', 0))
    except OSError:
        return False


def export_one(line, a):
    region, product = os.path.split(line.strip())
    h5path = os.path.join(a.out_dir, region, product + '.h5')
    if not a.overwrite and is_complete(h5path):
        return 'skipped (exists)'
    t0 = time.time()
    d = load_flatsim(a.base_dir, line, rms_thresh=a.rms_thresh, abs_thresh_mm=a.abs_thresh,
                     wavelength_mm=WVL, positive_away=POSITIVE_AWAY, with_velocity=False)
    cube = d['cube']
    T, ny, nx = cube.shape
    os.makedirs(os.path.dirname(h5path), exist_ok=True)
    tmp = h5path + '.tmp'
    kw = dict(compression='gzip', compression_opts=4, shuffle=True)
    c2 = (min(ny, 256), min(nx, 256))
    with h5py.File(tmp, 'w') as f:
        f.create_dataset('displacement', data=cube, chunks=(min(T, 32), min(ny, 64), min(nx, 64)), **kw)
        f.create_dataset('dates', data=d['dates'])
        for name in ('lon', 'lat'):
            f.create_dataset(name, data=d[name], chunks=c2, **kw)
        f.create_dataset('topography', data=d['topo'], chunks=c2, **kw)
        f.create_dataset('mask', data=d['mask'].astype(np.uint8), chunks=c2, **kw)
        f.create_dataset('los', data=d['los'], chunks=(min(ny, 128), min(nx, 128), 3), **kw)
        f['displacement'].attrs['units'] = 'mm'
        f['displacement'].attrs['reference'] = 'raw FLATSIM, zero at the first defined date of each pixel'
        for k, v in d['meta'].items():
            f.attrs[k] = v
        f.attrs['source'] = 'raw FLATSIM (no denoising, no filling)'
        f.attrs['rms_thresh'], f.attrs['abs_thresh_mm'] = a.rms_thresh, a.abs_thresh
        f.attrs['complete'] = 1
    os.replace(tmp, h5path)
    return f'done in {time.time() - t0:.0f}s, {os.path.getsize(h5path) / 1e9:.2f} GB'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', default=os.path.expanduser('~/Desktop/flatsim_product_names_folders.txt'))
    ap.add_argument('--base-dir', default='/Users/giuseppe/Documents/DATA/InSAR/FLATSIM')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--match', nargs='+', default=None, help='keep lines containing any of these strings')
    ap.add_argument('--exclude', nargs='+', default=['NSBAS_DAUX-PKG'], help='drop lines containing any of these')
    ap.add_argument('--rms-thresh', type=float, default=RMS_THRESH)
    ap.add_argument('--abs-thresh', type=float, default=ABS_THRESH)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    lines = [l.strip() for l in open(a.list) if l.strip()]
    lines = [l for l in dict.fromkeys(lines)
             if (not a.match or any(m in l for m in a.match)) and not any(x in l for x in a.exclude)]
    lines = lines[:a.limit]
    log(f'{len(lines)} series to export')
    os.makedirs(a.out_dir, exist_ok=True)
    for k, line in enumerate(lines, 1):
        try:
            log(f'[{k}/{len(lines)}] {line} -> {export_one(line, a)}')
        except Exception:
            with open(os.path.join(a.out_dir, 'export_errors.log'), 'a') as ef:
                ef.write(f'{line}\n{traceback.format_exc()}\n')
            log(f'[{k}/{len(lines)}] {line} FAILED (see export_errors.log)')


if __name__ == '__main__':
    main()
