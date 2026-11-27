#!/usr/bin/env python
"""
Compare two runs of solve_frame_offsets.py frame by frame, e.g. raw FLATSIM (A) against the denoised series (B).
Question answered: do the frame offsets e_i(t) (their drift, steps and end-of-record jumps) come from the data
(same in A and B) or are they created / changed by the reconstruction (different in A and B)?

Frames are matched by key and first/last date. Both runs should use the same --ref-track, --step-days and frames.
Per frame (the first/last --trim grid dates are excluded from the statistics):
  rate (Theil-Sen, mm/yr) in A and B, their difference, rms of e_B - e_A, explained fraction of e_B by e_A,
  range of e, end-of-record jump (mean of the last 3 dates minus the median of the 10 before),
  and for every --event date the step of e (median of the --event-days after minus before).
Output: a CSV, a scatter of rates, and one overlay PNG per (region, orbit) with A dashed and B solid.

Example:
  python compare_offsets.py off_levant_raw.h5 off_levant_den.h5 --names raw denoised --event 2021-05-22 --out cmp_levant
"""
import argparse
import os
from datetime import datetime

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import theilslopes


def load(path):
    with h5py.File(path, 'r') as f:
        fr = {k: np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f['frames'][k][...]])
              for k in ('key', 'orbit', 'region', 'label')}
        fr['first'], fr['last'] = f['frames']['first_ord'][...], f['frames']['last_ord'][...]
        return fr, f['time']['ordinal'][...].astype(np.int64), f['e'][...].astype(float)


def rate(t, y):
    m = np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    return float(theilslopes(y[m], (t[m] - t[m][0]) / 365.25)[0])


def step(t, y, ev, w):
    a = y[(t >= ev) & (t < ev + w)]
    b = y[(t >= ev - w) & (t < ev)]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(np.median(a) - np.median(b)) if len(a) >= 2 and len(b) >= 2 else np.nan


def end_jump(y):
    v = y[np.isfinite(y)]
    return float(np.mean(v[-3:]) - np.median(v[-13:-3])) if len(v) >= 14 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--names', nargs=2, default=['A', 'B'])
    ap.add_argument('--trim', type=int, default=2, help='grid dates excluded at both ends for the statistics')
    ap.add_argument('--event', nargs='+', default=[], help='YYYY-MM-DD dates for step estimates')
    ap.add_argument('--event-days', type=float, default=90.)
    ap.add_argument('--out', default='cmp_offsets')
    a = ap.parse_args()

    fa, ta, ea = load(a.a)
    fb, tb, eb = load(a.b)
    common_t = np.intersect1d(ta, tb)
    if len(common_t) < 10:
        raise SystemExit('the two runs have almost no common grid date')
    ia, ib = np.searchsorted(ta, common_t), np.searchsorted(tb, common_t)
    t = common_t
    ka = {f'{fa["key"][i]}|{fa["first"][i]}|{fa["last"][i]}': i for i in range(len(fa['key']))}
    kb = {f'{fb["key"][i]}|{fb["first"][i]}|{fb["last"][i]}': i for i in range(len(fb['key']))}
    both = [k for k in ka if k in kb]
    print(f'{len(both)} frames in both runs ({len(ka)} in {a.names[0]}, {len(kb)} in {a.names[1]})')
    sl = slice(a.trim, len(t) - a.trim)
    events = [datetime.strptime(s, '%Y-%m-%d').toordinal() for s in a.event]
    rows = []
    for k in both:
        i, j = ka[k], kb[k]
        ya, yb = ea[i, ia], eb[j, ib]
        m = np.isfinite(ya) & np.isfinite(yb)
        if m[sl].sum() < 5:
            continue
        d = yb - ya
        row = dict(label=fa['label'][i], region=fa['region'][i], orbit=fa['orbit'][i], i=i, j=j,
                   rate_a=rate(t[sl], ya[sl]), rate_b=rate(t[sl], yb[sl]),
                   rms_diff=float(np.sqrt(np.nanmean(d[sl][m[sl]] ** 2))),
                   expl=float(1 - np.nanvar(d[sl][m[sl]]) / max(np.nanvar(yb[sl][m[sl]]), 1e-12)),
                   range_a=float(np.ptp(ya[sl][m[sl]])), range_b=float(np.ptp(yb[sl][m[sl]])),
                   jump_a=end_jump(ya), jump_b=end_jump(yb))
        row['rate_diff'] = row['rate_b'] - row['rate_a']
        for s, ev in zip(a.event, events):
            row[f'step_a_{s}'] = step(t, ya, ev, a.event_days)
            row[f'step_b_{s}'] = step(t, yb, ev, a.event_days)
        rows.append(row)
    if not rows:
        raise SystemExit('no frame with enough common data')

    na, nb = a.names
    cols = ['label', 'region', 'orbit', 'rate_a', 'rate_b', 'rate_diff', 'rms_diff', 'expl', 'range_a', 'range_b',
            'jump_a', 'jump_b'] + [c for c in rows[0] if c.startswith('step_')]
    with open(a.out + '.csv', 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in rows:
            fh.write(','.join(str(round(r[c], 3)) if isinstance(r[c], float) else str(r[c]) for c in cols) + '\n')
    print(f'{"frame":<38}{"rate "+na:>11}{"rate "+nb:>11}{"diff":>8}{"rms diff":>10}{"expl.":>7}{"range "+na:>11}{"range "+nb:>11}')
    for r in rows:
        print(f'{r["label"]:<38}{r["rate_a"]:>11.2f}{r["rate_b"]:>11.2f}{r["rate_diff"]:>8.2f}{r["rms_diff"]:>10.2f}'
              f'{r["expl"]:>7.2f}{r["range_a"]:>11.1f}{r["range_b"]:>11.1f}')
    for s in a.event:
        sa = np.array([r[f'step_a_{s}'] for r in rows])
        sb = np.array([r[f'step_b_{s}'] for r in rows])
        print(f'step at {s}: median |step| {na} {np.nanmedian(np.abs(sa)):.1f} mm, {nb} {np.nanmedian(np.abs(sb)):.1f} mm')
    ra, rb = np.array([r['rate_a'] for r in rows]), np.array([r['rate_b'] for r in rows])
    print(f'median |rate| {na} {np.nanmedian(np.abs(ra)):.2f}, {nb} {np.nanmedian(np.abs(rb)):.2f} mm/yr; '
          f'median |difference| {np.nanmedian(np.abs(rb - ra)):.2f} mm/yr; correlation {np.corrcoef(ra[np.isfinite(ra) & np.isfinite(rb)], rb[np.isfinite(ra) & np.isfinite(rb)])[0, 1]:.2f}')

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for reg in sorted(set(r['region'] for r in rows)):
        sel = [r for r in rows if r['region'] == reg]
        ax.scatter([r['rate_a'] for r in sel], [r['rate_b'] for r in sel], s=14, label=reg)
    lim = np.nanmax(np.abs(np.r_[ra, rb])) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k-', lw=0.5)
    ax.set_xlabel(f'rate of e_i(t), {na} (mm/yr)')
    ax.set_ylabel(f'rate of e_i(t), {nb} (mm/yr)')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.savefig(a.out + '_rates.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    dt = [datetime.fromordinal(int(x)) for x in t]
    for (reg, orb) in sorted(set((r['region'], r['orbit']) for r in rows)):
        sel = [r for r in rows if r['region'] == reg and r['orbit'] == orb]
        fig, ax = plt.subplots(figsize=(12, 4))
        for n, r in enumerate(sel):
            c = f'C{n % 10}'
            ax.plot(dt, ea[r['i'], ia], '--', color=c, lw=0.8)
            ax.plot(dt, eb[r['j'], ib], '-', color=c, lw=0.9, label=r['label'])
        ax.set_title(f'{reg} | {orb}   (dashed: {na}, solid: {nb})', fontsize=9)
        ax.set_ylabel('offset e_i(t) (mm)')
        ax.legend(fontsize=5, ncol=3)
        ax.grid(alpha=0.3)
        fig.savefig(f'{a.out}_{reg}_{orb}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'saved {a.out}.csv, {a.out}_rates.png and one overlay per region/orbit')


if __name__ == '__main__':
    main()
