#!/usr/bin/env python
"""
plot_reconstruction.py  (plots from the reconstructions already written by flatsim_denoising.py)

For every time series of the product list (as in flatsim_denoising.py: NSBAS_DAUX-PKG lines discarded, duplicates
removed) it opens the finished reconstruction files  <denoised-dir>/<run>/<region>/<product>.h5  of every selected run
(a "run" = one flatsim_denoising.py output folder, i.e. one variant: velocity mode, --solve-tilt, --fit-plane, ...),
and makes the two plots of the longwave check:
  1. <product>_part*.png : raw FLATSIM + one column per run: map of last - first and four swath profiles
  2. <product>_overlay.png: the profiles of all runs on the same axes (raw dashed)
No network is run. The raw column comes from the original FLATSIM files (--base-dir); use --no-raw to skip it.

Runs
  --runs TAG [TAG ...]   run folders (names or globs) inside --denoised-dir; default: every folder that contains
                         reconstructions. Labels are built from the file attributes (velocity mode, tilt, plane).
  --exclude STR [...]    drop runs whose folder name contains STR, or whose label has that flag (e.g. tilt)
  --list-runs            print the runs found with their labels and stop
Series
  --list FILE | --product P | --only a b | --shard i/n | --limit N | --strip-region (flat local FLATSIM layout)
Selection (every series; default = full frame and full time span)
  --rows r0 r1  --cols c0 c1  --lonlat W E S N  --dates D0 D1 | --tidx i0 i1
  last - first is evaluated between the first and the last selected epoch (--n-avg epochs averaged at each end).

Outputs in --out-dir/<region>/: <product>_part*.png, <product>_overlay.png, <product>_summary.csv; in --out-dir:
summary_all_shard<i>.csv, skipped.txt, failed.txt.
"""
import argparse
import csv
import fnmatch
import gc
import glob
import os
import time
import traceback
import warnings

import h5py
import numpy as np

from utils.flatsim_utils import load_flatsim, reference_cube, flag_frozen

WVL, RMS_THRESH, ABS_THRESH, POSITIVE_AWAY = 55.5, 10., 300., True

PROFILES = [('vertical', 'tab:blue'), ('horizontal', 'tab:green'),
            ('diag \\ (top-left > bottom-right)', 'tab:red'), ('diag / (bottom-left > top-right)', 'tab:orange')]


# ----------------------------------------------------------------------------- product list / runs
def read_product_list(path, only=None, shard='0/1', limit=None):
    """As flatsim_denoising.py: DAUX entries discarded, duplicates removed, then shard i/n."""
    raw = [l.strip() for l in open(os.path.expanduser(path)) if l.strip() and not l.lstrip().startswith('#')]
    n_aux = sum('NSBAS_DAUX-PKG' in l for l in raw)
    ts = list(dict.fromkeys(l for l in raw if 'NSBAS_DAUX-PKG' not in l))
    if only:
        ts = [l for l in ts if any(o.lower() in l.lower() for o in only)]
    i, n = map(int, shard.split('/'))
    ts = ts[i::n]
    return (ts[:limit] if limit else ts), n_aux


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def label_from_attrs(at):
    parts = []
    vm = _s(at.get('cfg_velocity_mode', 'none'))
    if vm != 'none':
        vo = int(at.get('cfg_velocity_order', 1))
        parts.append(vm if vo == 1 else f'{vm}{vo}')
    if int(at.get('cfg_solve_tilt', 0)):
        pr = float(at.get('cfg_tilt_prior', 0.))
        parts.append('tilt' + (f'(prior={pr:g})' if pr > 0 else ''))
    if int(at.get('cfg_fit_plane', 0)):
        po = int(at.get('cfg_plane_order', 1))
        parts.append('plane' if po == 1 else f'plane{po}')
    return '+'.join(parts) if parts else 'base'


def discover_runs(root, patterns=None, exclude=()):
    """[{tag, label}] for every run folder under root that holds reconstructions."""
    tags = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if patterns:
        tags = [t for t in tags if any(fnmatch.fnmatch(t, p) or t == p for p in patterns)]
    runs = []
    for t in tags:
        files = glob.glob(os.path.join(root, t, '*', '*.h5')) + glob.glob(os.path.join(root, t, '*.h5'))
        if not files:
            continue
        at = {}
        for fpath in files[:20]:                                    # first complete file gives the attributes
            try:
                with h5py.File(fpath, 'r') as f:
                    if f.attrs.get('complete', 0):
                        at = dict(f.attrs)
                        break
            except OSError:
                continue
        runs.append(dict(tag=t, label=label_from_attrs(at), attrs=at))
    ex = [e.strip().lower() for e in exclude]
    runs = [r for r in runs if not any(e in r['tag'].lower() or e in r['label'].split('+') or e == r['label']
                                       or (e == 'plane' and any(p.startswith('plane') for p in r['label'].split('+')))
                                       for e in ex)]
    count = {}
    for r in runs:                                                    # make labels unique
        count[r['label']] = count.get(r['label'], 0) + 1
    for r in runs:
        if count[r['label']] > 1:
            at = r['attrs']
            r['label'] += f" [s{at.get('cfg_spatial_stride', '?')}/t{at.get('cfg_t_stride', '?')}]"
    count = {}
    for r in runs:
        count[r['label']] = count.get(r['label'], 0) + 1
    for r in runs:
        if count[r['label']] > 1:
            r['label'] += f" <{r['tag'][-30:]}>"
    return runs


def h5_path(a, run, line):
    region, product = os.path.split(line.strip())
    return os.path.join(a.denoised_dir, run['tag'], region, product + '.h5')


def out_stem(a, line):
    region, product = os.path.split(line.strip())
    d = os.path.join(a.out_dir, region) if region else a.out_dir
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, product)


# ----------------------------------------------------------------------------- selection / raw
def subset_slices(shape, lon, lat, rows, cols, lonlat):
    ny, nx = shape
    r0, r1, c0, c1 = 0, ny, 0, nx
    if lonlat:
        w, e, s, n = lonlat
        inside = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
        if not inside.any():
            raise ValueError('no pixel inside --lonlat box')
        rr, cc = np.flatnonzero(inside.any(1)), np.flatnonzero(inside.any(0))
        r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
    if rows:
        r0, r1 = max(0, rows[0]), min(ny, rows[1])
    if cols:
        c0, c1 = max(0, cols[0]), min(nx, cols[1])
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f'empty spatial subset rows {r0}:{r1}, cols {c0}:{c1}')
    return slice(int(r0), int(r1)), slice(int(c0), int(c1))


def time_slice(dates, date_range, tidx):
    T = len(dates)
    if date_range and tidx:
        raise ValueError('use either --dates or --tidx')
    if date_range:
        ok = np.flatnonzero((dates >= date_range[0]) & (dates <= date_range[1]))
        if len(ok) == 0:
            raise ValueError(f'no epoch between {date_range[0]} and {date_range[1]} ({dates[0]}..{dates[-1]})')
        return slice(int(ok[0]), int(ok[-1]) + 1)
    if tidx:
        return slice(max(0, tidx[0]), min(T, tidx[1]))
    return slice(0, T)


def km_per_pixel(lon, lat, default=(0.116, 0.110)):
    def hav(lo1, la1, lo2, la2):
        lo1, la1, lo2, la2 = map(np.deg2rad, (lo1, la1, lo2, la2))
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return 2 * 6371.0088 * np.arcsin(np.sqrt(a))

    out = []
    for axis, d in ((0, default[0]), (1, default[1])):
        try:
            a = (lon[:-1], lat[:-1], lon[1:], lat[1:]) if axis == 0 else (
                lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = float(np.nanmedian(hav(*a)))
            out.append(m if np.isfinite(m) and m > 0 else d)
        except Exception:
            out.append(d)
    return out[0], out[1]  # (km per row = azimuth, km per column = range)


def prepare_cube(cube_mm, mode):
    """Reference every pixel to its first finite epoch; mode 'mask' also NaNs the frozen (no-data) pixels."""
    c = cube_mm.copy()
    first = reference_cube(c)
    dead = np.zeros(c.shape[1:], bool)
    if mode != 'keep':
        dead, _ = flag_frozen(c, first)
        r, k = np.nonzero(dead & (first >= 0))
        c[first[r, k], r, k] = np.nan
    return c


def delta_cube(cube, n_avg):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return (np.nanmean(cube[-n_avg:], axis=0) - np.nanmean(cube[:n_avg], axis=0)).astype(np.float32)


def delta_h5(dset, st, sr, sc, n_avg):
    """last - first of the stored displacement between epochs st.start and st.stop-1 (n_avg epochs at each end)."""
    n = max(1, min(n_avg, (st.stop - st.start) // 2 or 1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.nanmean(np.asarray(dset[st.start:st.start + n, sr, sc], np.float32), axis=0)
        b = np.nanmean(np.asarray(dset[st.stop - n:st.stop, sr, sc], np.float32), axis=0)
    return (b - a).astype(np.float32)


# ----------------------------------------------------------------------------- profiles, statistics
def swath_profile(M, p0, p1, half_width, kx, ky, min_frac=0.3):
    (y0, x0), (y1, x1) = p0, p1
    L = float(np.hypot(y1 - y0, x1 - x0))
    n = int(np.ceil(L)) + 1
    s = np.linspace(0, L, n)
    dy, dx = (y1 - y0) / L, (x1 - x0) / L
    off = np.arange(-half_width, half_width + 1)
    yy = np.rint(y0 + s[None] * dy + off[:, None] * (-dx)).astype(int)
    xx = np.rint(x0 + s[None] * dx + off[:, None] * dy).astype(int)
    ok = (yy >= 0) & (yy < M.shape[0]) & (xx >= 0) & (xx < M.shape[1])
    vals = np.full(yy.shape, np.nan, np.float32)
    vals[ok] = M[yy[ok], xx[ok]]
    cnt = np.isfinite(vals).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        med = np.nanmedian(vals, axis=0)
    med[cnt < min_frac * len(off)] = np.nan
    return s * np.hypot(dy * ky, dx * kx), med


def profile_lines(bbox):
    r0, r1, c0, c1 = bbox
    rm, cm = (r0 + r1) / 2, (c0 + c1) / 2
    return [((r0, cm), (r1, cm)), ((rm, c0), (rm, c1)), ((r0, c0), (r1, c1)), ((r1, c0), (r0, c1))]


def all_profiles(M, bbox, half_width, kx, ky):
    return [swath_profile(M, p0, p1, half_width, kx, ky) for p0, p1 in profile_lines(bbox)]


def fit_stats(M, kx, ky, sub=2):
    """Plane (mm/100 km) and quadratic (mm at half extent) fits of a map; robust (3-sigma, 2 passes)."""
    ys, xs = np.mgrid[0:M.shape[0]:sub, 0:M.shape[1]:sub]
    z = M[ys, xs].astype(np.float64)
    ok = np.isfinite(z)
    out = dict(plane_az=np.nan, plane_rg=np.nan, q_vv=np.nan, q_uv=np.nan, q_uu=np.nan, rms_plane=np.nan,
               rms_quad=np.nan, p2p=np.nan)
    if ok.sum() < 50:
        return out
    ys, xs, z = ys[ok].astype(float), xs[ok].astype(float), z[ok]
    xk, yk = xs * kx, ys * ky
    u = (xs - M.shape[1] / 2) / (M.shape[1] / 2)
    v = (ys - M.shape[0] / 2) / (M.shape[0] / 2)

    def fit(B):
        use = np.ones(len(z), bool)
        for _ in range(3):
            c = np.linalg.lstsq(B[use], z[use], rcond=None)[0]
            r = z - B @ c
            s = 1.4826 * np.median(np.abs(r[use] - np.median(r[use]))) + 1e-9
            use = np.abs(r) < 3 * s
        return c, float(np.sqrt(np.mean(r[use] ** 2)))

    c1, r1 = fit(np.stack([np.ones_like(z), xk, yk], 1))
    c2, r2 = fit(np.stack([np.ones_like(z), u, v, u * u, u * v, v * v], 1))
    out.update(plane_rg=100 * c1[1], plane_az=100 * c1[2], rms_plane=r1, q_uu=c2[3], q_uv=c2[4], q_vv=c2[5],
               rms_quad=r2, p2p=float(np.nanpercentile(z, 99) - np.nanpercentile(z, 1)))
    return out


def profile_slopes(profs):
    out = []
    for d, p in profs:
        ok = np.isfinite(p)
        out.append(100 * np.polyfit(d[ok], p[ok], 1)[0] if ok.sum() > 5 else np.nan)
    return out


# ----------------------------------------------------------------------------- plots
def plot_page(path, cols, extent, bbox, kx, ky, vlim, ylim, title, half_width):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(max(4.2 * n, 7), 10.5), squeeze=False, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1.35, 1]))
    lines = profile_lines(bbox)
    for j, c in enumerate(cols):
        ax = axes[0, j]
        im = ax.imshow(c['map'], cmap='RdBu_r', vmin=-vlim, vmax=vlim, extent=extent, interpolation='nearest')
        for (p0, p1), (_, col) in zip(lines, PROFILES):
            ax.plot([p0[1] * kx, p1[1] * kx], [p0[0] * ky, p1[0] * ky], color=col, lw=1.2)
        ax.set_title(c['title'], fontsize=9)
        ax.set_xlabel('range (km)')
        if j == 0:
            ax.set_ylabel('azimuth (km)')
        bx = axes[1, j]
        for (d, p), (name, col) in zip(c['profiles'], PROFILES):
            bx.plot(d, p, color=col, lw=1.1, label=name)
        bx.axhline(0, color='k', lw=.5)
        bx.grid(alpha=.3)
        bx.set_ylim(*ylim)
        bx.set_xlabel('distance along line (km)')
        if j == 0:
            bx.set_ylabel(f'median over ±{half_width}px swath (mm)')
            bx.legend(fontsize=7)
        else:
            bx.set_yticklabels([])
    fig.colorbar(im, ax=axes[0, :].tolist(), orientation='horizontal', fraction=0.04, pad=0.02,
                 label='cumulative LOS displacement, last - first (mm; positive = away from satellite)')
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_overlay(path, cols, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    cmap = plt.get_cmap('tab20')
    k = 0
    for c in cols:
        for i, ax in enumerate(axes.ravel()):
            d, p = c['profiles'][i]
            if c['is_raw']:
                ax.plot(d, p, color='k', ls='--', lw=1.4, label='raw')
            else:
                ax.plot(d, p, color=cmap(k % 20), lw=1.2, label=c['label'])
        k += 0 if c['is_raw'] else 1
    for ax, (name, _) in zip(axes.ravel(), PROFILES):
        ax.set_title(name, fontsize=10)
        ax.axhline(0, color='k', lw=.5)
        ax.grid(alpha=.3)
        ax.set_xlabel('distance along line (km)')
        ax.set_ylabel('mm')
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- one series
def run_product(a, line, runs, stem):
    """Plots for one series from the finished reconstructions. Returns (status, summary rows)."""
    t0 = time.time()
    files = {}
    for r in runs:
        p = h5_path(a, r, line)
        if not os.path.exists(p):
            continue
        try:
            with h5py.File(p, 'r') as f:
                if not f.attrs.get('complete', 0):
                    print(f'  {r["label"]}: file not complete, skipped')
                    continue
        except OSError as e:
            print(f'  {r["label"]}: cannot open ({e})')
            continue
        files[r['label']] = p
    if not files:
        return 'skipped: no reconstruction found in the selected runs', []
    missing = [r['label'] for r in runs if r['label'] not in files]
    if missing:
        print('  no reconstruction for:', missing)

    ref_path = next(iter(files.values()))
    with h5py.File(ref_path, 'r') as f:
        dates_all = np.asarray(f['dates']).astype(int)
        lon_all, lat_all = np.asarray(f['lon']), np.asarray(f['lat'])
        shape = f['displacement'].shape[1:]
        orbit = _s(f.attrs.get('orbit', ''))
    sr, sc = subset_slices(shape, lon_all, lat_all, a.rows, a.cols, a.lonlat)
    st = time_slice(dates_all, a.dates, a.tidx)
    if st.stop - st.start < 2:
        return 'skipped: fewer than 2 epochs selected', []
    dates = dates_all[st]
    T = len(dates)
    H, W = sr.stop - sr.start, sc.stop - sc.start
    ky, kx = km_per_pixel(lon_all[sr, sc], lat_all[sr, sc])

    # raw FLATSIM
    raw_map = None
    if not a.no_raw:
        region, product = os.path.split(line.strip())
        load_line = os.path.basename(line.strip()) if a.strip_region else line.strip()
        lregion, lproduct = os.path.split(load_line)
        if os.path.isdir(os.path.join(a.base_dir, lregion, lproduct)):
            try:
                data = load_flatsim(a.base_dir, load_line, rms_thresh=a.rms_thresh, abs_thresh_mm=a.abs_thresh,
                                    wavelength_mm=WVL, positive_away=POSITIVE_AWAY, with_velocity=False,
                                    empty_mode='keep')
                if data['cube'].shape[0] != len(dates_all) or data['cube'].shape[1:] != tuple(shape):
                    print(f'  raw cube {data["cube"].shape} does not match the reconstruction '
                          f'({len(dates_all)},{shape}); raw column skipped')
                else:
                    raw_cube = prepare_cube(np.ascontiguousarray(data['cube'][st, sr, sc]).astype(np.float32),
                                            a.empty)
                    raw_map = delta_cube(raw_cube, max(1, min(a.n_avg, T // 2 or 1)))
                    del raw_cube
                del data
            except Exception as e:
                print(f'  raw FLATSIM could not be read ({e}); raw column skipped')
        else:
            print('  raw FLATSIM not downloaded; raw column skipped')

    # reconstructions
    maps, extra = {}, {}
    for lab, p in files.items():
        with h5py.File(p, 'r') as f:
            if f['displacement'].shape[1:] != tuple(shape) or len(f['dates']) != len(dates_all):
                print(f'  {lab}: shape/dates differ from the reference run, skipped')
                continue
            m = delta_h5(f['displacement'], st, sr, sc, a.n_avg)
            at = f.attrs
            ex = dict(rms_before=float(at.get('rms_before', np.nan)), rms_after=float(at.get('rms_after', np.nan)),
                      rms_after_constrained=float(at.get('rms_after_constrained', np.nan)),
                      tilt_rms=float(at.get('tilt_rms_mm_per_half_tile', np.nan)),
                      velocity_misfit=(np.asarray(at['velocity_misfit_final']).tolist()
                                       if 'velocity_misfit_final' in at else ''),
                      plane_removed_u_last=np.nan, plane_removed_v_last=np.nan)
            if 'plane_removed' in f:
                pr = np.asarray(f['plane_removed'])
                ex['plane_removed_u_last'], ex['plane_removed_v_last'] = float(pr[st.stop - 1, 0]), float(pr[st.stop - 1, 1])
        if a.mask_recon and raw_map is not None:
            m[~np.isfinite(raw_map)] = np.nan
        maps[lab], extra[lab] = m, ex
    if not maps:
        return 'skipped: no usable reconstruction', []
    order = [r['label'] for r in runs if r['label'] in maps]

    # common geometry, limits, profiles
    all_maps = ([raw_map] if raw_map is not None else []) + [maps[l] for l in order]
    fin = np.zeros((H, W), bool)
    for m in all_maps:
        fin |= np.isfinite(m)
    if not fin.any():
        return 'skipped: no valid pixel in the last-first maps', []
    rr, cc = np.flatnonzero(fin.any(1)), np.flatnonzero(fin.any(0))
    bbox = (int(rr[0]), int(rr[-1]), int(cc[0]), int(cc[-1]))
    allv = np.concatenate([m[np.isfinite(m)] for m in all_maps])
    vlim = a.vlim or float(np.percentile(np.abs(allv), 75))
    cols = []
    if raw_map is not None:
        cols.append(dict(label='raw FLATSIM', map=raw_map, is_raw=True))
    cols += [dict(label=l, map=maps[l], is_raw=False) for l in order]
    for c in cols:
        c['profiles'] = all_profiles(c['map'], bbox, a.swath, kx, ky)
    pv = np.concatenate([p[np.isfinite(p)] for c in cols for _, p in c['profiles']])
    ylim = tuple(a.ylim) if a.ylim else (float(pv.min() - 0.05 * np.ptp(pv)), float(pv.max() + 0.05 * np.ptp(pv)))

    def _d(x):
        x = str(int(x))
        return np.datetime64(f'{x[:4]}-{x[4:6]}-{x[6:8]}')
    years = (_d(dates[-1]) - _d(dates[0])) / np.timedelta64(1, 'D') / 365.25
    product_name = os.path.basename(line.strip())
    rows_out = []
    for c in cols:
        st_ = fit_stats(c['map'], kx, ky)
        ps = profile_slopes(c['profiles'])
        ex = extra.get(c['label'], {})
        c['title'] = f'{c["label"]}\nplane az {st_["plane_az"]:+.1f}, rg {st_["plane_rg"]:+.1f} mm/100km'
        rows_out.append(dict(product=product_name, orbit=orbit, n_epochs=int(T), span_years=round(float(years), 2),
                             variant=c['label'], plane_az_mm100km=st_['plane_az'], plane_rg_mm100km=st_['plane_rg'],
                             quad_vv_mm=st_['q_vv'], quad_uv_mm=st_['q_uv'], quad_uu_mm=st_['q_uu'],
                             rms_after_plane=st_['rms_plane'], rms_after_quad=st_['rms_quad'], p2p_1_99=st_['p2p'],
                             slope_vertical=ps[0], slope_horizontal=ps[1], slope_diag_bs=ps[2], slope_diag_s=ps[3],
                             rms_before=ex.get('rms_before', np.nan), rms_after=ex.get('rms_after', np.nan),
                             rms_after_constrained=ex.get('rms_after_constrained', np.nan),
                             tilt_rms=ex.get('tilt_rms', np.nan),
                             plane_removed_u_last=ex.get('plane_removed_u_last', np.nan),
                             plane_removed_v_last=ex.get('plane_removed_v_last', np.nan),
                             velocity_misfit=ex.get('velocity_misfit', '')))
    with open(stem + '_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    short = product_name.replace('NSBAS_TS-PKG_S1_', '')
    title = (f'{short[:60]} ({orbit}) | rows {sr.start}-{sr.stop}, cols {sc.start}-{sc.stop} | '
             f'{dates[0]}-{dates[-1]} ({T} epochs) | last - first (n_avg={a.n_avg}) | reconstructions from files')
    extent = [0, W * kx, H * ky, 0]
    raw_cols, var_cols = [c for c in cols if c['is_raw']], [c for c in cols if not c['is_raw']]
    per = max(a.max_cols - len(raw_cols), 1)
    for k in range(0, len(var_cols), per):
        plot_page(f'{stem}_part{k // per + 1}.png', raw_cols + var_cols[k:k + per], extent, bbox, kx, ky, vlim,
                  ylim, title, a.swath)
    plot_overlay(f'{stem}_overlay.png', cols, title)       # written last: marks the series as done
    return f'done in {time.time() - t0:.0f}s ({len(order)} runs{", raw" if raw_cols else ""})', rows_out


# ----------------------------------------------------------------------------- main
def get_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', default=os.path.expanduser('~/Desktop/flatsim_product_names_folders.txt'),
                    help='product list as in flatsim_denoising.py (DAUX entries are discarded)')
    ap.add_argument('--product', default=None, help="process only this series ('region/NSBAS_TS-PKG_...')")
    ap.add_argument('--only', nargs='+', default=None, help='keep list entries containing one of these strings')
    ap.add_argument('--shard', default='0/1', help='i/n: process entries i, i+n, ...')
    ap.add_argument('--limit', type=int, default=None, help='first N entries only (testing)')
    ap.add_argument('--strip-region', action='store_true', help='raw FLATSIM in a flat layout: base-dir/NSBAS_TS-PKG_...')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--base-dir', default='/Users/giuseppe/Documents/DATA/InSAR/FLATSIM', help='raw FLATSIM')
    ap.add_argument('--denoised-dir', default='/Users/giuseppe/Documents/DATA/InSAR/FLATSIM/DENOISED',
                    help='--out-dir used with flatsim_denoising.py (contains one folder per run)')
    ap.add_argument('--out-dir', default='/Users/giuseppe/Documents/DATA/InSAR/FLATSIM/DENOISED/local_check')
    ap.add_argument('--runs', nargs='+', default=None, help='run folders (names or globs); default: all')
    ap.add_argument('--exclude', nargs='+', default=[], help='drop runs by folder substring or label flag')
    ap.add_argument('--list-runs', action='store_true')
    ap.add_argument('--rows', nargs=2, type=int)
    ap.add_argument('--cols', nargs=2, type=int)
    ap.add_argument('--lonlat', nargs=4, type=float, metavar=('W', 'E', 'S', 'N'))
    ap.add_argument('--dates', nargs=2, type=int, metavar=('START', 'END'))
    ap.add_argument('--tidx', nargs=2, type=int, metavar=('I0', 'I1'))
    ap.add_argument('--empty', choices=['keep', 'mask'], default='mask', help='zero handling of the raw column')
    ap.add_argument('--no-raw', action='store_true', help='do not read the original FLATSIM files')
    ap.add_argument('--n-avg', type=int, default=1, help='epochs averaged at each end')
    ap.add_argument('--swath', type=int, default=10, help='half width of the median swath (pixels)')
    ap.add_argument('--max-cols', type=int, default=6, help='columns per figure (raw + runs)')
    ap.add_argument('--vlim', type=float, default=None, help='symmetric map limit (mm); default: 75th percentile')
    ap.add_argument('--ylim', nargs=2, type=float, default=None, help='profile y limits (mm)')
    ap.add_argument('--mask-recon', action='store_true',
                    help='show the reconstructions only where raw last-first is valid')
    ap.add_argument('--rms-thresh', type=float, default=RMS_THRESH)
    ap.add_argument('--abs-thresh', type=float, default=ABS_THRESH)
    ap.add_argument('--dry-run', action='store_true', help='list the series and which runs have a file, then stop')
    return ap


def main(argv=None):
    a = get_parser().parse_args(argv)
    runs = discover_runs(a.denoised_dir, a.runs, a.exclude)
    print('runs:')
    for r in runs:
        print(f'  {r["label"]:<28} {r["tag"]}')
    if a.list_runs:
        return
    if not runs:
        raise SystemExit(f'no run with reconstructions found in {a.denoised_dir}')
    if a.product:
        lines, n_aux = [a.product.strip()], 0
    else:
        lines, n_aux = read_product_list(a.list, a.only, a.shard, a.limit)
    os.makedirs(a.out_dir, exist_ok=True)
    shard_i = int(a.shard.split('/')[0])
    print(f'{len(lines)} time series ({n_aux} auxiliary DAUX entries discarded)')
    if a.dry_run:
        for line in lines:
            have = [r['label'] for r in runs if os.path.exists(h5_path(a, r, line))]
            print(f'  {len(have)}/{len(runs)} runs  {line}')
        return

    summary_path = os.path.join(a.out_dir, f'summary_all_shard{shard_i}.csv')
    for k, line in enumerate(lines):
        stem = out_stem(a, line)
        tag = f'[{k + 1}/{len(lines)}] {line}'
        if not a.overwrite and os.path.exists(f'{stem}_overlay.png'):
            print(tag, '-> skipped: already done')
            continue
        print(tag, flush=True)
        try:
            status, rows = run_product(a, line, runs, stem)
            if rows:
                new = not os.path.exists(summary_path)
                with open(summary_path, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    if new:
                        w.writeheader()
                    w.writerows(rows)
            else:
                open(os.path.join(a.out_dir, 'skipped.txt'), 'a').write(f'{line}\t{status}\n')
            print(tag, '->', status, flush=True)
        except Exception:
            with open(os.path.join(a.out_dir, 'failed.txt'), 'a') as ef:
                ef.write(f'{line}\n{traceback.format_exc()}\n')
            print(tag, '-> FAILED (see failed.txt)', flush=True)
        gc.collect()
    print('done; see', a.out_dir)


if __name__ == '__main__':
    main()
