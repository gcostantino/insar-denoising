"""flatsim_velocity.py - read the FLATSIM mean LOS velocity and sanity-check it against the cube."""
import logging

import numpy as np

log = logging.getLogger(__name__).info
warn = logging.getLogger(__name__).warning


def raw_velocity(cube, dates_years, sub=8):
    """NaN-aware per-pixel linear regression slope of cube[:, ::sub, ::sub] (units/yr)."""
    c = cube[:, ::sub, ::sub].astype(np.float64)
    t = np.asarray(dates_years, float)[:, None, None]
    ok = np.isfinite(c)
    n = ok.sum(0)
    tb = np.where(ok, t, 0).sum(0) / np.maximum(n, 1)
    dt = np.where(ok, t - tb, 0)
    yb = np.where(ok, c, 0).sum(0) / np.maximum(n, 1)
    num = (dt * np.where(ok, c - yb, 0)).sum(0)
    den = (dt ** 2).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        v = num / den
    v[n < 5] = np.nan
    return v


def velocity_sanity(cube_mm, dates_years, vel_mm, sub=8, min_corr=0.8):
    """Compare the tiff velocity with the regression of the raw cube. Catches wrong units, sign, orientation."""
    vr = raw_velocity(cube_mm, dates_years, sub)
    vf = vel_mm[::sub, ::sub]
    ok = np.isfinite(vr) & np.isfinite(vf)
    if ok.sum() < 100:
        warn('velocity sanity: too few common pixels')
        return None
    a, b = vf[ok], vr[ok]
    if a.std() < 1e-6 or b.std() < 1e-6:  # e.g. all-zero frames: nothing to correlate
        res = dict(corr=np.nan, slope=np.nan, median_diff=float(np.median(b - a)), n=int(ok.sum()))
    else:
        res = dict(corr=float(np.corrcoef(a, b)[0, 1]), slope=float(np.polyfit(a, b, 1)[0]),
                   median_diff=float(np.median(b - a)), n=int(ok.sum()))
    log(f'velocity sanity: corr={res["corr"]:.3f} slope(raw/FLATSIM)={res["slope"]:.3f} '
        f'median diff={res["median_diff"]:.3f} mm/yr (n={res["n"]})')
    if np.isfinite(res['corr']) and (res['corr'] < min_corr or abs(res['slope'] - 1) > 0.3):
        warn('velocity sanity FAILED: check units (rad vs mm), sign, or orientation of the tiff')
    return res
