#!/usr/bin/env python
"""Per-frame diagnostics of ramps in the denoised FLATSIM series (run where the denoised .h5 files are).

For every complete series it computes the cumulative displacement d = last - first (or the rate d/span), then
 1. fits a plane d = a + gE*E + gN*N over the whole frame (robust, 2 passes) and reports the gradient size, its
    direction, and its components along RANGE (look direction) and AZIMUTH (flight direction);
 2. if --start/--end are given, extracts the frame's OWN profile along that line (no merging with other frames)
    and fits its slope, so a sawtooth of per-frame tilts can be seen directly.
A ramp with a fixed orientation and the same sign in all frames (large range component, small azimuth component)
points to a residual range-direction ramp; random orientations point elsewhere.

Examples:
  python frame_ramps.py --results-dir results/<tag> --csv ramps.csv
  python frame_ramps.py --results-dir results/<tag> --start 80 33 --end 100 33 --width 20 --out own_profiles.png
"""
import argparse
import glob
import os
import re
import time
from datetime import datetime

import h5py
import numpy as np

R_EARTH = 6371.0088


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def parse_dates(raw):
    out = []
    for x in np.asarray(raw).ravel():
        s = x.decode() if isinstance(x, (bytes, np.bytes_)) else (x if isinstance(x, str) else str(int(x)))
        out.append(datetime.strptime(s.strip()[:8], '%Y%m%d').toordinal())
    return np.array(out)


def attr_str(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def track_label(key):
    m = re.search(r'S1_(.+?)-VV', key)
    return m.group(1) if m else key.split('/')[-1]


def unit_vec(lon, lat):
    lam, phi = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)


class Profile:
    def __init__(self, start, end):
        self.a, b = unit_vec(*start), unit_vec(*end)
        n = np.cross(self.a, b)
        self.n = n / np.linalg.norm(n)
        self.t = np.cross(self.n, self.a)
        self.length = R_EARTH * float(np.arccos(np.clip(self.a @ b, -1, 1)))

    def project(self, lon, lat):
        p = unit_vec(lon, lat)
        return R_EARTH * np.arctan2(p @ self.t, p @ self.a), R_EARTH * np.arcsin(np.clip(p @ self.n, -1, 1))


def bin_median(bins, vals, nbin):
    med, cnt = np.full(nbin, np.nan), np.zeros(nbin, int)
    if not len(vals):
        return med, cnt
    order = np.lexsort((vals, bins))
    b, v = bins[order], vals[order]
    u, start, c = np.unique(b, return_index=True, return_counts=True)
    med[u] = 0.5 * (v[start + (c - 1) // 2] + v[start + c // 2])
    cnt[u] = c
    return med, cnt


def fit_plane(x, y, d, max_samples=2_000_000, seed=0):
    """Robust plane fit d = a + gx*x + gy*y (x, y in km). Returns a, gx, gy (per km), rms before/after, n."""
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
    return coef, float(np.std(d[keep])), float(np.std(res[keep])), int(keep.sum())


def process(path, a, prof):
    with h5py.File(path, 'r') as f:
        if not bool(f.attrs.get('complete', 0)):
            return None
        orbit = attr_str(f.attrs.get('orbit', 'unknown')).lower()
        if a.orbit != 'both' and orbit != a.orbit:
            return None
        key = attr_str(f.attrs.get('key', os.path.basename(path)))
        heading = float(f.attrs.get('heading_deg', np.nan))
        U = f['displacement']
        n = a.n_avg
        dates = parse_dates(f['dates'][...])
        span = (dates[-n:].mean() - dates[:n].mean()) / 365.25
        with np.errstate(all='ignore'):
            d = (np.nanmean(U[-n:], axis=0) - np.nanmean(U[:n], axis=0)).astype(np.float64)
        if a.field == 'rate':
            d = d / span
        lon, lat = f['lon'][...].astype(np.float64), f['lat'][...].astype(np.float64)
        masked = f['mask'][...].astype(bool)
    lon = (lon + 180.0) % 360.0 - 180.0
    ok = np.isfinite(lon) & np.isfinite(lat) & ~((lon == 0) & (lat == 0)) & ~masked & np.isfinite(d)
    if ok.sum() < a.min_pixels:
        return None
    lo, la, dv = lon[ok], lat[ok], d[ok]
    out = dict(key=key, label=track_label(key), orbit=orbit, heading=heading, n_pix=int(ok.sum()), span=span)
    lon_c, lat_c = float(np.median(lo)), float(np.median(la))
    x = (lo - lon_c) * np.cos(np.deg2rad(lat_c)) * 111.32
    y = (la - lat_c) * 110.57
    coef, rms0, rms1, nk = fit_plane(x, y, dv)
    gE, gN = coef[1] * 100.0, coef[2] * 100.0          # per 100 km
    look = np.deg2rad(heading + 90.0)                   # right-looking SAR
    head = np.deg2rad(heading)
    out.update(gE=gE, gN=gN, gmag=float(np.hypot(gE, gN)), gaz=float(np.rad2deg(np.arctan2(gE, gN)) % 360),
               g_range=float(gE * np.sin(look) + gN * np.cos(look)),
               g_azim=float(gE * np.sin(head) + gN * np.cos(head)), rms0=rms0, rms1=rms1,
               lon_c=lon_c, lat_c=lat_c)
    if prof is not None:
        along, cross = prof.project(lo, la)
        sel = (np.abs(cross) <= a.width / 2) & (along >= 0) & (along < prof.length)
        if sel.sum() >= a.min_pixels // 10:
            nbin = int(np.ceil(prof.length / a.step))
            med, cnt = bin_median(np.floor(along[sel] / a.step).astype(np.int64), dv[sel], nbin)
            med[cnt < a.min_bin_pixels] = np.nan
            xs = (np.arange(nbin) + 0.5) * a.step
            m = np.isfinite(med)
            if m.sum() >= 5 and np.ptp(xs[m]) > 20:
                out.update(prof_x=xs, prof_y=med, prof_slope=float(np.polyfit(xs[m], med[m], 1)[0] * 100),
                           prof_from=float(xs[m].min()), prof_to=float(xs[m].max()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--field', choices=['cumulative', 'rate'], default='cumulative')
    ap.add_argument('--orbit', choices=['ascending', 'descending', 'both'], default='both')
    ap.add_argument('--n-avg', type=int, default=1)
    ap.add_argument('--start', type=float, nargs=2, metavar=('LON', 'LAT'))
    ap.add_argument('--end', type=float, nargs=2, metavar=('LON', 'LAT'))
    ap.add_argument('--width', type=float, default=20.)
    ap.add_argument('--step', type=float, default=2.)
    ap.add_argument('--min-pixels', type=int, default=10000)
    ap.add_argument('--min-bin-pixels', type=int, default=20)
    ap.add_argument('--demean', action='store_true', help='subtract each frame\'s median profile value in the plot')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--out', default='own_profiles.png')
    ap.add_argument('--dpi', type=int, default=200)
    a = ap.parse_args()
    prof = Profile(a.start, a.end) if a.start and a.end else None
    unit = 'mm' if a.field == 'cumulative' else 'mm/yr'

    files = sorted(p for p in glob.glob(os.path.join(a.results_dir, '**', '*.h5'), recursive=True)
                   if not os.path.basename(p).startswith('master'))[:a.limit]
    res = []
    for i, p in enumerate(files, 1):
        t0 = time.time()
        try:
            r = process(p, a, prof)
        except Exception as e:
            log(f'[{i}/{len(files)}] FAILED {p}: {e}')
            continue
        if r is not None:
            res.append(r)
            log(f'[{i}/{len(files)}] {r["orbit"][:4]} {r["label"]} ({time.time() - t0:.0f}s)')

    for orbit in ('ascending', 'descending'):
        rr = [r for r in res if r['orbit'] == orbit]
        if not rr:
            continue
        print(f'\n{orbit}: whole-frame plane fit of {a.field} displacement ({unit} per 100 km); '
              f'look direction = heading + 90 deg')
        print(f'  {"frame":<28} {"|grad|":>7} {"dir (deg)":>9} {"range":>8} {"azimuth":>8} {"rms":>7} {"rms-plane":>9}'
              + (f' {"own-profile slope":>18}' if prof else ''))
        for r in sorted(rr, key=lambda r: r['lon_c']):
            ps = f' {r["prof_slope"]:>10.2f} ({r["prof_from"]:.0f}-{r["prof_to"]:.0f} km)' if 'prof_slope' in r else ''
            print(f'  {r["label"]:<28} {r["gmag"]:7.2f} {r["gaz"]:9.0f} {r["g_range"]:8.2f} {r["g_azim"]:8.2f} '
                  f'{r["rms0"]:7.1f} {r["rms1"]:9.1f}{ps}')
        gr, ga = np.array([r['g_range'] for r in rr]), np.array([r['g_azim'] for r in rr])
        print(f'  range component: mean {gr.mean():+.2f}, median {np.median(gr):+.2f}, std {gr.std():.2f}, '
              f'{(np.sign(gr) == np.sign(np.median(gr))).mean():.0%} same sign as the median')
        print(f'  azimuth component: mean {ga.mean():+.2f}, median {np.median(ga):+.2f}, std {ga.std():.2f}, '
              f'{(np.sign(ga) == np.sign(np.median(ga))).mean():.0%} same sign as the median')
        if prof:
            sl = np.array([r['prof_slope'] for r in rr if 'prof_slope' in r])
            if len(sl):
                print(f'  own-profile slopes: mean {sl.mean():+.2f}, median {np.median(sl):+.2f} ({unit}/100 km, {len(sl)} frames)')

    if a.csv:
        cols = ['orbit', 'label', 'key', 'n_pix', 'span', 'heading', 'lon_c', 'lat_c', 'gE', 'gN', 'gmag', 'gaz',
                'g_range', 'g_azim', 'rms0', 'rms1', 'prof_slope']
        with open(a.csv, 'w') as fh:
            fh.write(','.join(cols) + '\n')
            for r in res:
                fh.write(','.join(str(r.get(c, '')) for c in cols) + '\n')
        log(f'saved {a.csv}')

    if prof is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        orbs = [o for o in ('ascending', 'descending') if any(r['orbit'] == o and 'prof_y' in r for r in res)]
        if orbs:
            fig, axs = plt.subplots(len(orbs), 1, figsize=(12, 4.5 * len(orbs)), squeeze=False)
            for ax, o in zip(axs[:, 0], orbs):
                for k, r in enumerate([r for r in res if r['orbit'] == o and 'prof_y' in r]):
                    y = r['prof_y'] - (np.nanmedian(r['prof_y']) if a.demean else 0)
                    ax.plot(r['prof_x'], y, lw=1.1, color=plt.cm.tab20(k % 20))
                    m = np.isfinite(y)
                    ax.text(r['prof_x'][m][0], y[m][0], r['label'].replace('TIBET-', '').replace('HIM-', ''), fontsize=5)
                ax.set_title(f'{o}: each frame\'s own profile ({a.field}, {a.width:.0f} km swath, median per {a.step:g} km bin)')
                ax.set_ylabel(f'{a.field} displacement ({unit})')
                ax.grid(alpha=0.3)
            axs[-1, 0].set_xlabel('Distance along profile (km)')
            fig.tight_layout()
            fig.savefig(a.out, dpi=a.dpi)
            log(f'saved {a.out}')


if __name__ == '__main__':
    main()
