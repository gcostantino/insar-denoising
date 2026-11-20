#!/usr/bin/env python
"""Long-wavelength check of every frame: raw FLATSIM, reconstruction, re-aligned frame (last - first, mm).

For each frame, a 2x3 figure:
  row 1: maps   [1] raw FLATSIM cube (same masks / thresholds as the denoiser input)
                [2] reconstruction (denoised series, should be velocity_mode = none, the 'rawest' version)
                [3] re-aligned frame = [2] - offset_cum_mm of the inter-frame offset solve in the mosaic file
  row 2: profiles of each map along four lines of the image (the "cross" and the "x"): vertical and horizontal lines
         through the center of the valid-data bounding box, and its two diagonals. Each profile is the median over a
         swath of +- --swath pixels perpendicular to the line. Axes in km (pixel size from the lon/lat of the frame);
         the three profile panels share the y axis.
By default ALL frames of the mosaic file are processed (--frames to select labels, --orbit / --filter to restrict).
A frame that fails (no reconstruction file, grid mismatch, ...) is reported and skipped; the run continues.
Outputs in --out-dir: frame_check_<label>.png per frame, summary.csv (one row per frame, written as the run goes) and
failed.txt. Printed per frame: offset applied, plane (range, azimuth) gradients of the maps and of reconstruction - raw
(mm per 100 km), slope / peak-to-peak of every profile.
Note: [3] is [2] minus one constant, so profile shapes in [2] and [3] are identical; the panel shows how large the offset
is compared with the variation inside the frame.

Orientation: images are drawn as stored (rows = azimuth, downward; columns = range).

python frame_longwave_check.py --results-dir results/tag_none --mosaic slip_mosaic_none.h5 --base-dir /data/FLATSIM
python frame_longwave_check.py --frames TIBET-D004SUD TIBET-D004NORD --results-dir ... --mosaic ... --base-dir ...
python frame_longwave_check.py --orbit descending --filter 'TIBET-D0(04|33)' --skip-existing --results-dir ...
Run it from the repository root (it imports utils.flatsim_utils.load_flatsim).
"""
import argparse
import csv
import glob
import os
import re
import time
import traceback
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import map_coordinates

LINE_STYLE = {'vertical': 'tab:blue', 'horizontal': 'tab:green', 'diag \\ (top-left > bottom-right)': 'tab:red',
              'diag / (bottom-left > top-right)': 'tab:orange'}
LINE_SHORT = {'vertical': 'v', 'horizontal': 'h', 'diag \\ (top-left > bottom-right)': 'd1',
              'diag / (bottom-left > top-right)': 'd2'}


def track_label(key):
    m = re.search(r'S1_(.+?)-VV', key)
    return m.group(1) if m else key.split('/')[-1]


def attr_str(v):
    return v.decode() if isinstance(v, bytes) else str(v)


# ----------------------------------------------------------------------------- locating the inputs
def index_results(results_dir):
    """{key: path} of the complete series found under results_dir (one scan)."""
    idx = {}
    for p in sorted(glob.glob(os.path.join(results_dir, '**', '*.h5'), recursive=True)):
        if os.path.basename(p).startswith('master'):
            continue
        try:
            with h5py.File(p, 'r') as f:
                if bool(f.attrs.get('complete', 0)) and 'key' in f.attrs:
                    idx.setdefault(attr_str(f.attrs['key']), p)
        except OSError:
            continue
    return idx


def read_mosaic_frames(mosaic):
    """{key: dict(offsets, orbit, ...)} from the /frames table, plus the mosaic's global attributes."""
    with h5py.File(mosaic, 'r') as f:
        if 'frames' not in f or 'offset_cum_mm' not in f['frames']:
            raise SystemExit(f'{mosaic} has no offsets: build it with --offset-mode rate or full')
        fr = f['frames']
        keys = [attr_str(x) for x in fr['key'][...]]
        cols = [n for n in ('offset_cum_mm', 'offset_rate_mm_yr', 'offset_det_mm', 'offset_group',
                            'offset_is_reference', 'offset_n_pairs', 'span_yr', 'orbit') if n in fr]
        data = {n: fr[n][...] for n in cols}
        glob_attrs = dict(n_avg=f.attrs.get('n_avg', np.nan), offset_mode=attr_str(f.attrs.get('offset_mode', '?')))
    table = {}
    for i, k in enumerate(keys):
        row = {n: (attr_str(data[n][i]) if n == 'orbit' else data[n][i]) for n in cols}
        row.update(glob_attrs)
        table[k] = row
    return table


def end_minus_start(series, n_avg):
    """mean of the last n_avg epochs - mean of the first n_avg epochs; series (T, ny, nx)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return (np.nanmean(series[-n_avg:], axis=0) - np.nanmean(series[:n_avg], axis=0)).astype(np.float32)


def pixel_size_km(lon, lat, win=200):
    """(dx, dy) in km: pixel spacing along columns (range) and along rows (azimuth)."""
    ny, nx = lon.shape
    y0, x0 = max(0, ny // 2 - win // 2), max(0, nx // 2 - win // 2)
    lo = lon[y0:y0 + win, x0:x0 + win].astype(float)
    la = lat[y0:y0 + win, x0:x0 + win].astype(float)
    bad = ~(np.isfinite(lo) & np.isfinite(la) & ~((lo == 0) & (la == 0)))
    lo[bad], la[bad] = np.nan, np.nan
    c = np.cos(np.deg2rad(np.nanmean(la)))
    dx = np.nanmedian(np.hypot(np.diff(lo, axis=1) * c, np.diff(la, axis=1))) * 111.195
    dy = np.nanmedian(np.hypot(np.diff(lo, axis=0) * c, np.diff(la, axis=0))) * 111.195
    return float(dx), float(dy)


# ----------------------------------------------------------------------------- profiles and plane fits
def valid_bbox(m):
    r = np.flatnonzero(np.any(m, axis=1))
    c = np.flatnonzero(np.any(m, axis=0))
    return int(r[0]), int(r[-1]), int(c[0]), int(c[-1])


def make_lines(bbox):
    r0, r1, c0, c1 = bbox
    rc, cc = 0.5 * (r0 + r1), 0.5 * (c0 + c1)
    return {'vertical': ((r0, cc), (r1, cc)),
            'horizontal': ((rc, c0), (rc, c1)),
            'diag \\ (top-left > bottom-right)': ((r0, c0), (r1, c1)),
            'diag / (bottom-left > top-right)': ((r1, c0), (r0, c1))}


def swath_profile(img, a, b, half_width, dx, dy, min_frac=0.3):
    """Median over +- half_width pixels perpendicular to the line a->b; returns distance (km) and profile."""
    (ra, ca), (rb, cb) = a, b
    dr, dc = rb - ra, cb - ca
    n = int(np.ceil(max(abs(dr), abs(dc)))) + 1
    s = np.linspace(0, 1, n)
    rows, cols = ra + dr * s, ca + dc * s
    nrm = np.hypot(dr, dc)
    pr, pc = -dc / nrm, dr / nrm                             # unit normal in pixel space
    ks = np.arange(-half_width, half_width + 1)
    R = rows[None, :] + ks[:, None] * pr
    C = cols[None, :] + ks[:, None] * pc
    vals = map_coordinates(img.astype(np.float64), [R, C], order=0, mode='constant', cval=np.nan)
    ok = np.isfinite(vals)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        prof = np.nanmedian(vals, axis=0)
    prof[ok.mean(axis=0) < min_frac] = np.nan
    return s * np.hypot(dr * dy, dc * dx), prof


def plane_gradients(img, dx, dy, max_pts=300_000, seed=0):
    """Least-squares plane z = a + gr*x + ga*y (x = range km, y = azimuth km); gradients in mm per 100 km."""
    rr, cc = np.nonzero(np.isfinite(img))
    if len(rr) < 100:
        return np.nan, np.nan
    if len(rr) > max_pts:
        sel = np.random.default_rng(seed).choice(len(rr), max_pts, replace=False)
        rr, cc = rr[sel], cc[sel]
    z = img[rr, cc].astype(np.float64)
    A = np.stack([np.ones(len(z)), cc * dx, rr * dy], axis=1)
    coef = np.linalg.lstsq(A, z, rcond=None)[0]
    return float(coef[1] * 100), float(coef[2] * 100)


def line_slope(x, y):
    m = np.isfinite(y)
    if m.sum() < 5 or (x[m].max() - x[m].min()) < 5:
        return np.nan, np.nan
    return float(np.polyfit(x[m], y[m], 1)[0] * 100), float(np.ptp(y[m]))


# ----------------------------------------------------------------------------- figure and report
def make_figure(label, maps, titles, dx, dy, bbox, half_width, vlim, demean, header):
    ny, nx = maps[0].shape
    shown = [m - np.nanmedian(m) for m in maps] if demean else maps
    fig = plt.figure(figsize=(19, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.22, wspace=0.18)
    norm, cmap = Normalize(-vlim, vlim), plt.get_cmap('RdBu_r')
    lines = make_lines(bbox)
    axm = [fig.add_subplot(gs[0, j]) for j in range(3)]
    axp = [fig.add_subplot(gs[1, j]) for j in range(3)]
    ylo, yhi = np.inf, -np.inf
    profiles = {}
    for j in range(3):
        axm[j].imshow(shown[j], extent=[0, nx * dx, ny * dy, 0], cmap=cmap, norm=norm, interpolation='nearest',
                      aspect='equal')
        axm[j].set_title(titles[j], fontsize=10)
        axm[j].set_xlabel('range (km)')
        if j == 0:
            axm[j].set_ylabel('azimuth (km)')
        for name, (a, b) in lines.items():
            axm[j].plot([a[1] * dx, b[1] * dx], [a[0] * dy, b[0] * dy], color=LINE_STYLE[name], lw=1.2)
            x, p = swath_profile(shown[j], a, b, half_width, dx, dy)
            profiles[(j, name)] = (x, p)
            axp[j].plot(x, p, color=LINE_STYLE[name], lw=1.3, label=name)
            if np.isfinite(p).any():
                ylo, yhi = min(ylo, np.nanmin(p)), max(yhi, np.nanmax(p))
        axp[j].axhline(0, color='k', lw=0.5)
        axp[j].grid(alpha=0.3)
        axp[j].set_xlabel('distance along line (km)')
        if j == 0:
            axp[j].set_ylabel('median over swath (mm)' + (' - map median' if demean else ''))
            axp[j].legend(fontsize=7, loc='best')
    if np.isfinite(ylo):
        pad = 0.05 * (yhi - ylo + 1e-9)
        for ax in axp:
            ax.set_ylim(ylo - pad, yhi + pad)
    fig.colorbar(ScalarMappable(norm, cmap), ax=axm, orientation='horizontal', fraction=0.03, pad=0.08,
                 label='cumulative LOS displacement, last - first (mm; positive = away from satellite)')
    fig.suptitle(f'{label} | {header}', fontsize=11)
    return fig, profiles


def report(label, maps, names, dx, dy, profiles, off):
    print(f'\n=== {label} ===')
    print(f' pixel size: {dx * 1000:.0f} m (range) x {dy * 1000:.0f} m (azimuth)')
    print(f' offset applied in [3]: {off["offset_cum_mm"]:+.2f} mm'
          + (f' (rate {off["offset_rate_mm_yr"]:+.3f} mm/yr, level {off["offset_det_mm"]:+.2f} mm)'
             if 'offset_rate_mm_yr' in off else '')
          + f', group {int(off.get("offset_group", -1))}, reference {int(off.get("offset_is_reference", 0))}, '
            f'{int(off.get("offset_n_pairs", 0))} pairs')
    print(' plane gradients (mm per 100 km)       range   azimuth')
    for nme, m in zip(names, maps):
        gr, ga = plane_gradients(m, dx, dy)
        print(f'   {nme:<34} {gr:+8.2f} {ga:+8.2f}')
    diff = maps[1] - maps[0]
    gr, ga = plane_gradients(diff, dx, dy)
    print(f'   {"reconstruction - raw":<34} {gr:+8.2f} {ga:+8.2f}   (mean {np.nanmean(diff):+.2f} mm)')
    print(' profiles: slope (mm per 100 km) / peak-to-peak (mm)')
    print(f'   {"line":<36}' + ''.join(f'{n:>20}' for n in ('raw', 'reconstruction', 're-aligned')))
    for name in LINE_STYLE:
        cells = []
        for j in range(3):
            sl, pp = line_slope(*profiles[(j, name)])
            cells.append(f'{sl:+8.2f} / {pp:6.1f}')
        print(f'   {name:<36}' + ''.join(f'{c:>20}' for c in cells))


# ----------------------------------------------------------------------------- one frame
def process_frame(key, path, off, a, load_flatsim):
    label = track_label(key)
    with h5py.File(path, 'r') as f:
        mode = attr_str(f.attrs.get('cfg_velocity_mode', 'none'))
        if mode != 'none':
            print(f'WARNING {label}: reconstruction was run with velocity_mode={mode}, not the rawest version')
        recon = end_minus_start(f['displacement'], a.n_avg)
        lon, lat = f['lon'][...], f['lat'][...]
        kw = dict(rms_thresh=float(f.attrs.get('rms_thresh', 10.)), abs_thresh_mm=float(f.attrs.get('abs_thresh', 300.)),
                  wavelength_mm=float(f.attrs.get('wvl_mm', 55.5)), positive_away=bool(f.attrs.get('positive_away', True)))
    data = load_flatsim(a.base_dir, key, **kw)
    raw = end_minus_start(data['cube'], a.n_avg)
    del data
    if raw.shape != recon.shape:
        raise ValueError(f'raw {raw.shape} and reconstruction {recon.shape} grids differ')
    if np.isfinite(off['n_avg']) and int(off['n_avg']) != a.n_avg:
        print(f'WARNING {label}: the mosaic was built with n_avg={int(off["n_avg"])}, this run uses {a.n_avg}')
    if a.mask_recon:
        recon[~np.isfinite(raw)] = np.nan
    aligned = recon - np.float32(off['offset_cum_mm'])
    maps = [raw, recon, aligned]
    valid = np.isfinite(raw) if np.isfinite(raw).any() else np.isfinite(recon)
    if not valid.any():
        raise ValueError('no valid pixel')
    bbox = valid_bbox(valid)
    dx, dy = pixel_size_km(lon, lat)
    vlim = a.vlim or float(np.nanpercentile(np.abs(recon[np.isfinite(recon)]), 98))
    titles = ['[1] raw FLATSIM', f'[2] reconstruction (velocity {mode})',
              f'[3] re-aligned ({-off["offset_cum_mm"]:+.1f} mm, {off["offset_mode"]} offsets)']
    header = (f'last - first (n_avg={a.n_avg}) | span {float(off.get("span_yr", np.nan)):.2f} yr | '
              f'valid bbox rows {bbox[0]}-{bbox[1]}, cols {bbox[2]}-{bbox[3]}')
    fig, profiles = make_figure(label, maps, titles, dx, dy, bbox, a.swath, vlim, a.demean, header)
    report(label, maps, ['[1] raw', '[2] reconstruction', '[3] re-aligned'], dx, dy, profiles, off)
    out = os.path.join(a.out_dir, f'frame_check_{label}.png')
    fig.savefig(out, dpi=a.dpi, bbox_inches='tight')
    print('saved', out)
    if a.show:
        plt.show()
    plt.close(fig)

    row = dict(label=label, key=key, orbit=off.get('orbit', ''), offset_cum_mm=float(off['offset_cum_mm']),
               offset_rate_mm_yr=float(off.get('offset_rate_mm_yr', np.nan)),
               offset_det_mm=float(off.get('offset_det_mm', np.nan)), group=int(off.get('offset_group', -1)),
               is_reference=int(off.get('offset_is_reference', 0)), n_pairs=int(off.get('offset_n_pairs', 0)),
               span_yr=float(off.get('span_yr', np.nan)), dx_km=dx, dy_km=dy)
    for nme, m in (('raw', raw), ('recon', recon)):
        gr, ga = plane_gradients(m, dx, dy)
        row[f'grad_range_{nme}'], row[f'grad_az_{nme}'] = gr, ga
    gr, ga = plane_gradients(recon - raw, dx, dy)
    row.update(grad_range_diff=gr, grad_az_diff=ga, mean_diff_mm=float(np.nanmean(recon - raw)))
    for name, short in LINE_SHORT.items():
        row[f'slope_{short}_raw'] = line_slope(*profiles[(0, name)])[0]
        row[f'slope_{short}_recon'] = line_slope(*profiles[(1, name)])[0]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', nargs='+', default=['all'],
                    help='track labels (e.g. TIBET-D004SUD, exact or substring) or "all" (default)')
    ap.add_argument('--orbit', choices=['ascending', 'descending', 'both'], default='both')
    ap.add_argument('--filter', default=None, help='regular expression the track label must match')
    ap.add_argument('--results-dir', required=True, help='folder with the reconstructions (velocity_mode none)')
    ap.add_argument('--mosaic', required=True, help='mosaic .h5 built from those reconstructions (offset-mode rate|full)')
    ap.add_argument('--base-dir', required=True, help='FLATSIM base directory (for the raw cube)')
    ap.add_argument('--n-avg', type=int, default=1, help='epochs averaged at each end (1 = literal last - first)')
    ap.add_argument('--swath', type=int, default=10, help='half-width of the profile swath (pixels)')
    ap.add_argument('--vlim', type=float, default=None, help='symmetric colour limit (default: 98th percentile of |[2]|)')
    ap.add_argument('--mask-recon', action='store_true', help='hide the reconstruction where the raw data are NaN')
    ap.add_argument('--demean', action='store_true', help='subtract the median of each map (maps and profiles)')
    ap.add_argument('--skip-existing', action='store_true', help='skip frames whose PNG already exists')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--dpi', type=int, default=150)
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()
    try:
        from utils.flatsim_utils import load_flatsim
    except ImportError as e:
        raise SystemExit(f'cannot import utils.flatsim_utils.load_flatsim ({e}); run from the repository root')
    os.makedirs(a.out_dir, exist_ok=True)

    table = read_mosaic_frames(a.mosaic)
    keys = list(table)
    if a.frames != ['all']:
        sel = []
        for lab in a.frames:
            exact = [k for k in keys if track_label(k) == lab]
            part = [k for k in keys if lab in track_label(k)]
            found = exact or part
            if not found:
                print(f'no frame matches "{lab}" in {a.mosaic}')
            sel += [k for k in found if k not in sel]
        keys = sel
    if a.orbit != 'both':
        keys = [k for k in keys if table[k].get('orbit', '') == a.orbit]
    if a.filter:
        keys = [k for k in keys if re.search(a.filter, track_label(k))]
    if not keys:
        raise SystemExit('no frame selected')
    print(f'{len(keys)} frame(s) selected; scanning {a.results_dir} ...')
    index = index_results(a.results_dir)

    csv_path = os.path.join(a.out_dir, 'summary.csv')
    writer, fh = None, None
    failed, n_ok, t0 = [], 0, time.time()
    for n, key in enumerate(keys, 1):
        label = track_label(key)
        print(f'\n[{n}/{len(keys)}] {label}', flush=True)
        if a.skip_existing and os.path.exists(os.path.join(a.out_dir, f'frame_check_{label}.png')):
            print(' skipped (PNG exists)')
            continue
        try:
            if key not in index:
                raise FileNotFoundError(f'no complete reconstruction for {key} under {a.results_dir}')
            row = process_frame(key, index[key], table[key], a, load_flatsim)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failed.append((label, f'{type(e).__name__}: {e}'))
            print(f' FAILED: {type(e).__name__}: {e}')
            if not isinstance(e, (FileNotFoundError, ValueError)):
                traceback.print_exc()
            continue
        n_ok += 1
        if writer is None:
            fh = open(csv_path, 'w', newline='')
            writer = csv.DictWriter(fh, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        el = time.time() - t0
        print(f' done ({el / 60:.1f} min elapsed, ETA {el / n * (len(keys) - n) / 60:.1f} min)')
    if fh:
        fh.close()
    if failed:
        with open(os.path.join(a.out_dir, 'failed.txt'), 'w') as ff:
            for lab, msg in failed:
                ff.write(f'{lab}\t{msg}\n')
    print(f'\n{n_ok} frame(s) done, {len(failed)} failed'
          + (f' (see {os.path.join(a.out_dir, "failed.txt")})' if failed else '')
          + (f'; summary in {csv_path}' if n_ok else ''))


if __name__ == '__main__':
    main()
