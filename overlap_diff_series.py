#!/usr/bin/env python
"""
Overlap difference series between reconstructed frames (same orbit): is the disagreement a constant level, a rate,
a step, a seasonal wiggle, or a spatial ramp?

For every pair of series that overlap in space and time:
  * cells = the mosaic grid of build_disp_mosaic.py (same binning, masked pixels excluded); a random subset of the
    overlap cells (--max-cells) is used;
  * both series are brought to the same dates (the denser series sets the dates, the other one is interpolated
    linearly in time) over the COMMON time window;
  * D(t, cell) = u_i - u_j, then referenced per cell to its mean over the first --ref-days of the window, so only the
    time-varying part of the disagreement is left (a constant level cancels in any window displacement);
  * per epoch: median over cells, 16-84 % spread over cells, and a plane fit (gradient E-W / N-S, mm per 100 km).
Fits on the median series: robust linear rate (Theil-Sen), linear + annual, best single step. No smoothing anywhere.
Output per pair: PNG + CSV in --out-dir.

Examples:
  python overlap_diff_series.py --results-dir results/<tag> --match TURQUIE-A087 --out-dir ovdiff
  python overlap_diff_series.py results/<tag>/turkey/<A087-2014-2021>.h5 results/<tag>/turkey/<A087-2020-2025>.h5
  python overlap_diff_series.py --results-dir results/<tag> --match MEXIQUE-A049 MEXIQUE-A005 --ref-days 120
Needs build_disp_mosaic.py in the same folder (same cell definition).
"""
import argparse
import glob
import itertools
import os
import re
import time
import warnings
from datetime import datetime

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from build_disp_mosaic import LON0, LAT0, attr_str, native_spacing_deg, parse_dates, pixel_cells, track_label

warnings.filterwarnings('ignore', category=RuntimeWarning)
Y2000 = datetime(2000, 1, 1).toordinal()


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def label(g):
    d0, d1 = (datetime.fromordinal(int(x)).strftime('%Y-%m') for x in (g['dates'][0], g['dates'][-1]))
    return f'{track_label(g["key"])} {d0}..{d1}'


# ----------------------------------------------------------------------------- geometry and reading
def frame_geometry(path, res, nx_g):
    """Unmasked valid pixels of a series and their mosaic cell ids."""
    with h5py.File(path, 'r') as f:
        uniq, inv, ok, masked, _ = pixel_cells(f, res, nx_g)
        dates = parse_dates(f['dates'][...])
        orbit = attr_str(f.attrs.get('orbit', 'unknown')).lower()
        key = attr_str(f.attrs.get('key', os.path.basename(path)))
    flat = np.flatnonzero(ok.ravel())
    keep = ~masked.ravel()[flat]
    return dict(path=path, orbit=orbit, key=key, dates=dates, shape=ok.shape,
                pix=flat[keep], cid=uniq[inv][keep], ids=np.unique(uniq[inv][keep]))


def read_cell_series(g, common, k0, k1, block):
    """Mean displacement of the valid pixels of every cell in `common` (sorted ids), epochs k0..k1 -> (n_ep, n_cells)."""
    pos = np.searchsorted(common, g['cid'])
    pos = np.minimum(pos, len(common) - 1)
    hit = common[pos] == g['cid']
    pix, cp = g['pix'][hit], pos[hit]
    ny, nx = g['shape']
    r, c = np.divmod(pix, nx)
    r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
    loc = (r - r0) * (c1 - c0) + (c - c0)
    n = len(common)
    out = np.full((k1 - k0 + 1, n), np.nan, np.float32)
    with h5py.File(g['path'], 'r') as f:
        U = f['displacement']
        for a in range(k0, k1 + 1, block):
            b = min(a + block, k1 + 1)
            blk = U[a:b, r0:r1, c0:c1].reshape(b - a, -1)
            for q in range(b - a):
                v = blk[q, loc].astype(np.float64)
                fin = np.isfinite(v)
                s = np.bincount(cp, weights=np.where(fin, v, 0.0), minlength=n)
                cnt = np.bincount(cp, weights=fin.astype(float), minlength=n)
                out[a - k0 + q] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return out


def at_dates(A, dates_all, k0, t):
    """Linear interpolation in time of A (rows = epochs k0, k0+1, ...) at ordinal days t."""
    da = dates_all[k0:k0 + len(A)]
    k = np.clip(np.searchsorted(da, t, 'right') - 1, 0, len(da) - 2)
    den = (da[k + 1] - da[k]).astype(float)
    fr = np.where(den > 0, np.clip((t - da[k]) / den, 0, 1), 0.0)
    out = np.empty((len(t), A.shape[1]), np.float32)
    for n, (kk, ff) in enumerate(zip(k, fr)):
        out[n] = A[kk] if ff == 0 else (A[kk + 1] if ff == 1 else (1 - ff) * A[kk] + ff * A[kk + 1])
    return out


# ----------------------------------------------------------------------------- statistics
def plane_gradients(Dref, lon, lat, max_cells=20000, seed=0):
    """Per-epoch plane fit (E-W, N-S gradient in mm / 100 km) of the cell-wise disagreement."""
    n = Dref.shape[1]
    sel = np.arange(n) if n <= max_cells else np.sort(np.random.default_rng(seed).choice(n, max_cells, replace=False))
    lo, la = lon[sel], lat[sel]
    x = (lo - lo.mean()) * 111.32 * np.cos(np.deg2rad(la.mean()))
    y = (la - la.mean()) * 110.57
    X = np.column_stack([np.ones_like(x), x / 100.0, y / 100.0])
    gE, gN = np.full(len(Dref), np.nan), np.full(len(Dref), np.nan)
    for k in range(len(Dref)):
        v = Dref[k, sel]
        m = np.isfinite(v)
        if m.sum() > 50:
            c = np.linalg.lstsq(X[m], v[m].astype(float), rcond=None)[0]
            gE[k], gN[k] = c[1], c[2]
    return gE, gN, float(np.ptp(x)), float(np.ptp(y))


def theil(x, y):
    from scipy.stats import theilslopes
    m = np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    s, b, _, _ = theilslopes(y[m], x[m])
    return float(s), float(b)


def lstsq_rss(X, y):
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ c
    return c, float(r @ r), r


def fit_summary(t, med):
    """t ordinal days, med median-series. Returns dict of fit results."""
    ty = (t - t[0]) / 365.25
    m = np.isfinite(med)
    out = dict(T=float(ty[-1]))
    s, b = theil(ty, med)
    out.update(rate=s, intercept=b)
    res = med - (b + s * ty)
    out['rms_lin'] = float(np.sqrt(np.nanmean(res ** 2)))
    out['mad_lin'] = float(1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res))))
    tau = (t - Y2000) / 365.25
    Xa = np.column_stack([np.ones(m.sum()), ty[m], np.cos(2 * np.pi * tau[m]), np.sin(2 * np.pi * tau[m])])
    c, rss_a, _ = lstsq_rss(Xa, med[m])
    out['rms_lin_annual'] = float(np.sqrt(rss_a / m.sum()))
    out['annual_amp'] = float(np.hypot(c[2], c[3]))
    out['annual_coef'] = c
    X1 = np.column_stack([np.ones(m.sum()), ty[m]])
    _, rss_1, _ = lstsq_rss(X1, med[m])
    best = (np.inf, None, None)
    idx = np.flatnonzero(m)
    for q in range(int(0.1 * len(idx)), int(0.9 * len(idx))):
        H = (np.arange(len(idx)) >= q).astype(float)
        c2, rss2, _ = lstsq_rss(np.column_stack([X1, H]), med[m])
        if rss2 < best[0]:
            best = (rss2, idx[q], c2[2])
    out['step_ratio'] = best[0] / rss_1 if rss_1 > 0 else np.nan
    out['step_date'] = datetime.fromordinal(int(t[best[1]])).strftime('%Y-%m-%d') if best[1] is not None else ''
    out['step_mm'] = float(best[2]) if best[2] is not None else np.nan
    out['rate_fit_end'] = out['intercept'] + out['rate'] * out['T']
    return out


# ----------------------------------------------------------------------------- one pair
def analyse_pair(gi, gj, res, nx_g, a):
    common = np.intersect1d(gi['ids'], gj['ids'])
    n_total = len(common)
    if n_total < a.min_cells:
        return None
    ta, tb = max(gi['dates'][0], gj['dates'][0]), min(gi['dates'][-1], gj['dates'][-1])
    if (tb - ta) / 365.25 < a.min_window_yr:
        return None
    if n_total > a.max_cells:
        common = np.sort(np.random.default_rng(a.seed).choice(common, a.max_cells, replace=False))
    rng_ = {}
    for tag, g in (('i', gi), ('j', gj)):
        inn = np.flatnonzero((g['dates'] >= ta) & (g['dates'] <= tb))
        if len(inn) < 3:
            return None
        rng_[tag] = (inn, max(inn[0] - 1, 0), min(inn[-1] + 1, len(g['dates']) - 1))
    base = gi if len(rng_['i'][0]) >= len(rng_['j'][0]) else gj
    t = base['dates'][rng_['i'][0] if base is gi else rng_['j'][0]]
    X = {}
    for tag, g in (('i', gi), ('j', gj)):
        _, k0, k1 = rng_[tag]
        A = read_cell_series(g, common, k0, k1, a.block)
        X[tag] = at_dates(A, g['dates'], k0, t)
        del A
    D = X['i'] - X['j']
    del X
    ref = (t - t[0]) <= a.ref_days
    ref[0] = True
    D0 = np.nanmean(D[ref], axis=0)
    Dref = D - D0[None, :]
    med = np.nanmedian(Dref, axis=1)
    p16, p84 = np.nanpercentile(Dref, [16, 84], axis=1)
    raw_med = np.nanmedian(D, axis=1)
    nval = np.isfinite(Dref).sum(1)
    ix, iy = common % nx_g, common // nx_g
    lon, lat = LON0 + (ix + 0.5) * res, LAT0 + (iy + 0.5) * res
    gE, gN, ex, ey = plane_gradients(Dref, lon, lat, seed=a.seed)
    fit = fit_summary(t, med)
    q = max(3, len(t) // 4)
    endmap = np.nanmean(Dref[-q:], axis=0)
    sE, _ = theil((t - t[0]) / 365.25, gE)
    sN, _ = theil((t - t[0]) / 365.25, gN)
    return dict(t=t, med=med, p16=p16, p84=p84, raw=raw_med, nval=nval, gE=gE, gN=gN, lon=lon, lat=lat,
                endmap=endmap, fit=fit, n_total=n_total, n_used=len(common), extent_km=(ex, ey),
                gradrate=(sE, sN), window=(ta, tb), label_i=label(gi), label_j=label(gj), base=label(base))


def report_lines(r):
    f = r['fit']
    sp = float(np.nanmedian(r['p84'] - r['p16']) / 2)
    ex, ey = r['extent_km']
    return [f'{r["label_i"]}  minus  {r["label_j"]}',
            f'window {datetime.fromordinal(int(r["window"][0])):%Y-%m-%d} .. {datetime.fromordinal(int(r["window"][1])):%Y-%m-%d}'
            f'  ({f["T"]:.2f} yr), {len(r["t"])} dates (base {r["base"]})',
            f'overlap cells {r["n_total"]} (used {r["n_used"]}), extent {ex:.0f} x {ey:.0f} km',
            f'linear rate (Theil-Sen): {f["rate"]:+.2f} mm/yr -> {f["rate"] * f["T"]:+.1f} mm over the window',
            f'residual about line: rms {f["rms_lin"]:.2f}, MAD {f["mad_lin"]:.2f} mm',
            f'+ annual term: rms {f["rms_lin_annual"]:.2f} mm, amplitude {f["annual_amp"]:.2f} mm',
            f'best single step: {f["step_mm"]:+.1f} mm at {f["step_date"]}, RSS ratio vs line {f["step_ratio"]:.2f}',
            f'last epoch value {r["med"][-1]:+.1f} mm vs line at end {f["rate_fit_end"]:+.1f} mm',
            f'spread across cells (half 16-84 %): {sp:.2f} mm',
            f'plane gradient at end: E-W {np.nanmean(r["gE"][-5:]):+.2f}, N-S {np.nanmean(r["gN"][-5:]):+.2f} mm/100km',
            f'gradient trend: E-W {r["gradrate"][0]:+.2f}, N-S {r["gradrate"][1]:+.2f} mm/100km/yr']


def plot_pair(r, out_png):
    dt = [datetime.fromordinal(int(x)) for x in r['t']]
    f = r['fit']
    ty = (r['t'] - r['t'][0]) / 365.25
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, width_ratios=[1.7, 1], hspace=0.28, wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(dt, r['p16'], r['p84'], color='0.85', label='16-84 % across cells')
    ax.plot(dt, r['med'], 'k.-', ms=3, lw=0.6, label='median over cells')
    ax.plot(dt, f['intercept'] + f['rate'] * ty, 'r-', lw=1.5, label=f'linear {f["rate"]:+.2f} mm/yr')
    tau = (r['t'] - Y2000) / 365.25
    c = f['annual_coef']
    ax.plot(dt, c[0] + c[1] * ty + c[2] * np.cos(2 * np.pi * tau) + c[3] * np.sin(2 * np.pi * tau), 'b-', lw=0.8,
            label='linear + annual')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('disagreement relative to the reference period (mm)')
    ax.set_title(f'{r["label_i"]}  -  {r["label_j"]}', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.plot(dt, r['gE'], 'tab:blue', lw=0.8, label='E-W gradient')
    ax2.plot(dt, r['gN'], 'tab:orange', lw=0.8, label='N-S gradient')
    ax2.axhline(0, color='k', lw=0.4)
    ax2.set_ylabel('plane fit (mm / 100 km)')
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(gs[0, 1])
    v = r['endmap'] - np.nanmedian(r['endmap'])
    lim = float(np.nanpercentile(np.abs(v), 98)) or 1.0
    sc = ax3.scatter(r['lon'], r['lat'], c=v, s=1, cmap='RdBu_r', vmin=-lim, vmax=lim, rasterized=True)
    ax3.set_aspect(1 / max(np.cos(np.deg2rad(np.nanmean(r['lat']))), 0.1))
    ax3.set_title('end-of-window disagreement, minus its median (mm)', fontsize=9)
    fig.colorbar(sc, ax=ax3, fraction=0.04)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.text(0, 1, '\n'.join(report_lines(r)[1:]), va='top', family='monospace', fontsize=7)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_csv(r, path):
    dates = [datetime.fromordinal(int(x)).strftime('%Y-%m-%d') for x in r['t']]
    with open(path, 'w') as fh:
        fh.write('date,n_valid,median_ref_mm,p16_mm,p84_mm,median_raw_mm,grad_EW_mm_100km,grad_NS_mm_100km\n')
        for k, d in enumerate(dates):
            fh.write(f'{d},{r["nval"][k]},{r["med"][k]:.3f},{r["p16"][k]:.3f},{r["p84"][k]:.3f},{r["raw"][k]:.3f},'
                     f'{r["gE"][k]:.3f},{r["gN"][k]:.3f}\n')


# ----------------------------------------------------------------------------- main
def collect_files(a):
    files = list(a.files)
    if a.results_dir:
        allf = sorted(glob.glob(os.path.join(a.results_dir, '**', '*.h5'), recursive=True))
        for p in allf:
            b = os.path.basename(p)
            if b.startswith('master') or 'DAUX' in p:
                continue
            if not a.match or any(m in p for m in a.match):
                files.append(p)
    out = []
    for p in dict.fromkeys(files):
        try:
            with h5py.File(p, 'r') as f:
                if bool(f.attrs.get('complete', 1)):
                    out.append(p)
        except OSError:
            log(f'cannot open {p}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help='reconstructed series (.h5)')
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--match', nargs='+', default=None, help='with --results-dir: keep paths containing any of these strings')
    ap.add_argument('--out-dir', default='ovdiff')
    ap.add_argument('--res', type=float, default=None, help='cell size in degrees (default: native pixel size)')
    ap.add_argument('--min-cells', type=int, default=500)
    ap.add_argument('--max-cells', type=int, default=100000, help='random subset of the overlap cells')
    ap.add_argument('--min-window-yr', type=float, default=0.5)
    ap.add_argument('--ref-days', type=float, default=90., help='reference period at the start of the common window')
    ap.add_argument('--max-pairs', type=int, default=20)
    ap.add_argument('--block', type=int, default=32, help='epochs read at once (HDF5 chunk size along time)')
    ap.add_argument('--seed', type=int, default=0)
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
    log(f'cell = {res} deg')
    geo = []
    for p in files:
        g = frame_geometry(p, res, nx_g)
        log(f'  {g["orbit"][:4]} {label(g)}: {len(g["ids"])} cells')
        geo.append(g)
    os.makedirs(a.out_dir, exist_ok=True)
    done = 0
    for gi, gj in itertools.combinations(geo, 2):
        if done >= a.max_pairs:
            break
        if gi['orbit'] != gj['orbit']:
            continue
        r = analyse_pair(gi, gj, res, nx_g, a)
        if r is None:
            continue
        done += 1
        stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{r["label_i"]}__{r["label_j"]}')
        plot_pair(r, os.path.join(a.out_dir, f'ovdiff_{stem}.png'))
        save_csv(r, os.path.join(a.out_dir, f'ovdiff_{stem}.csv'))
        log('\n  '.join([''] + report_lines(r)))
    log(f'{done} pair(s) written to {a.out_dir}')


if __name__ == '__main__':
    main()
