#!/usr/bin/env python
"""Does the reconstruction (denoiser + stitching) introduce the planar ramps? Compare, frame by frame, the plane
fitted to the RAW FLATSIM series with the plane fitted to the DENOISED series, and to their difference.

For every selected denoised .h5 it reloads the raw series with your utils.flatsim_utils.load_flatsim (same
thresholds as flatsim_denoising.py), builds  d = median(last K epochs) - median(first K epochs)  for the raw cube and
for the denoised cube on exactly the same valid pixels, and fits d = a + gE*E + gN*N for
  raw, denoised, and denoised - raw
(gradients in the series' unit per 100 km, split into RANGE = look direction and AZIMUTH = flight direction).
If the denoised-minus-raw plane is large, the reconstruction adds the ramp; if raw already has it, it comes from
the data. Run it from your repository (or pass --repo) on the machine that has both the raw and denoised files.

Example:
  python compare_raw_ramps.py --results-dir results/<tag> --base-dir /data/giuseppe/FLATSIM --repo . --match A026 --png-dir cmp
"""
import argparse
import glob
import os
import re
import sys
import time
from datetime import datetime

import h5py
import numpy as np

WVL, RMS_THRESH, ABS_THRESH, POSITIVE_AWAY = 55.5, 10., 300., True     # as in flatsim_denoising.py


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def attr_str(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def track_label(key):
    m = re.search(r'S1_(.+?)-VV', key)
    return m.group(1) if m else key.split('/')[-1]


def fit_plane(x, y, d, max_samples=2_000_000, seed=0):
    if len(d) > max_samples:
        sel = np.random.default_rng(seed).choice(len(d), max_samples, replace=False)
        x, y, d = x[sel], y[sel], d[sel]
    keep = np.ones(len(d), bool)
    for _ in range(3):
        A = np.column_stack([np.ones(keep.sum()), x[keep], y[keep]])
        coef = np.linalg.lstsq(A, d[keep], rcond=None)[0]
        res = d - (coef[0] + coef[1] * x + coef[2] * y)
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res) < 3 * max(s, 1e-6)
    return coef, float(np.std(d[keep])), float(np.std(res[keep]))


def endpoints(cube, k):
    """median of the last k epochs minus median of the first k epochs (NaN-aware), per pixel."""
    with np.errstate(all='ignore'):
        return np.nanmedian(cube[-k:], axis=0) - np.nanmedian(cube[:k], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', required=True, help='folder with the denoised .h5 files')
    ap.add_argument('--base-dir', required=True, help='root of the raw FLATSIM data (as --base-dir of flatsim_denoising.py)')
    ap.add_argument('--repo', default='.', help='folder containing the utils package (added to sys.path)')
    ap.add_argument('--match', nargs='*', default=None, help='substrings of the frame key to select (e.g. A026 D004)')
    ap.add_argument('--orbit', choices=['ascending', 'descending', 'both'], default='both')
    ap.add_argument('--k-epochs', type=int, default=10, help='epochs in each end median (raw is noisy)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--png-dir', default=None, help='save a raw / denoised / difference map per frame here')
    ap.add_argument('--stride', type=int, default=8, help='pixel stride of the maps')
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.repo))
    from utils.flatsim_utils import load_flatsim

    files = sorted(p for p in glob.glob(os.path.join(a.results_dir, '**', '*.h5'), recursive=True)
                   if not os.path.basename(p).startswith('master'))
    sel = []
    for p in files:
        with h5py.File(p, 'r') as f:
            if not bool(f.attrs.get('complete', 0)):
                continue
            key, orbit = attr_str(f.attrs['key']), attr_str(f.attrs.get('orbit', '')).lower()
        if (a.match is None or any(m in key for m in a.match)) and (a.orbit == 'both' or orbit == a.orbit):
            sel.append((p, key, orbit))
    sel = sel[:a.limit]
    log(f'{len(sel)} frames selected')
    rows = []
    for i, (p, key, orbit) in enumerate(sel, 1):
        t0 = time.time()
        with h5py.File(p, 'r') as f:
            heading = float(f.attrs.get('heading_deg', np.nan))
            den = endpoints(f['displacement'][...], a.k_epochs)
            lon, lat = f['lon'][...].astype(np.float64), f['lat'][...].astype(np.float64)
            masked = f['mask'][...].astype(bool)
        data = load_flatsim(a.base_dir, key, rms_thresh=RMS_THRESH, abs_thresh_mm=ABS_THRESH,
                            wavelength_mm=WVL, positive_away=POSITIVE_AWAY)
        raw = endpoints(np.asarray(data['cube']), a.k_epochs)
        del data
        if raw.shape != den.shape:
            log(f'[{i}/{len(sel)}] {key}: shape mismatch raw {raw.shape} vs denoised {den.shape}, skipped')
            continue
        lon = (lon + 180.0) % 360.0 - 180.0
        ok = (np.isfinite(lon) & np.isfinite(lat) & ~((lon == 0) & (lat == 0)) & ~masked
              & np.isfinite(raw) & np.isfinite(den))
        if ok.sum() < 10000:
            log(f'[{i}/{len(sel)}] {key}: too few common valid pixels ({int(ok.sum())})')
            continue
        lo, la = lon[ok], lat[ok]
        lon_c, lat_c = float(np.median(lo)), float(np.median(la))
        x = (lo - lon_c) * np.cos(np.deg2rad(lat_c)) * 111.32
        y = (la - lat_c) * 110.57
        look, head = np.deg2rad(heading + 90.0), np.deg2rad(heading)
        r = dict(label=track_label(key), orbit=orbit, n_pix=int(ok.sum()), lon_c=lon_c, lat_c=lat_c)
        for name, field in (('raw', raw), ('den', den), ('diff', den - raw)):
            coef, rms0, rms1 = fit_plane(x, y, field[ok].astype(np.float64))
            gE, gN = coef[1] * 100, coef[2] * 100
            r.update({f'{name}_range': gE * np.sin(look) + gN * np.cos(look),
                      f'{name}_azim': gE * np.sin(head) + gN * np.cos(head),
                      f'{name}_rms': rms0, f'{name}_rms_plane': rms1})
        rows.append(r)
        log(f'[{i}/{len(sel)}] {r["orbit"][:4]} {r["label"]} done ({time.time() - t0:.0f}s)')
        if a.png_dir:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            os.makedirs(a.png_dir, exist_ok=True)
            s = a.stride
            fields = [raw[::s, ::s], den[::s, ::s], (den - raw)[::s, ::s]]
            for fld in fields:
                fld[~ok[::s, ::s]] = np.nan
            v = np.nanpercentile(np.abs(fields[1]), 98)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for ax, fld, t in zip(axs, fields, ('raw', 'denoised', 'denoised - raw')):
                im = ax.imshow(fld, cmap='RdBu_r', vmin=-v, vmax=v)
                ax.set_title(f'{r["label"]} {t} (last - first, mm)')
                plt.colorbar(im, ax=ax, shrink=0.7)
            fig.tight_layout()
            fig.savefig(os.path.join(a.png_dir, f'{r["label"]}.png'), dpi=120)
            plt.close(fig)

    print('\nplane gradients per 100 km (series unit), RANGE = look direction, AZIMUTH = flight direction; '
          'rms before -> after plane removal')
    print(f'  {"frame":<26}{"orb":>4} | {"raw range":>9}{"raw azim":>9} | {"den range":>9}{"den azim":>9} | '
          f'{"diff range":>10}{"diff azim":>9} | {"rms raw":>13}{"rms den":>13}{"rms diff":>13}')
    for r in rows:
        print(f'  {r["label"]:<26}{r["orbit"][:4]:>4} | {r["raw_range"]:9.2f}{r["raw_azim"]:9.2f} | '
              f'{r["den_range"]:9.2f}{r["den_azim"]:9.2f} | {r["diff_range"]:10.2f}{r["diff_azim"]:9.2f} | '
              f'{r["raw_rms"]:6.1f}->{r["raw_rms_plane"]:5.1f}{r["den_rms"]:6.1f}->{r["den_rms_plane"]:5.1f}'
              f'{r["diff_rms"]:6.1f}->{r["diff_rms_plane"]:5.1f}')
    if a.csv and rows:
        with open(a.csv, 'w') as fh:
            fh.write(','.join(rows[0].keys()) + '\n')
            for r in rows:
                fh.write(','.join(str(v) for v in r.values()) + '\n')
        log(f'saved {a.csv}')


if __name__ == '__main__':
    main()
