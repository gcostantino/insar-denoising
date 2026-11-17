
"""flatsim_velocity.py - read the FLATSIM mean LOS velocity and sanity-check it against the cube."""
import glob
import logging
import os

import numpy as np

log = logging.getLogger(__name__).info
warn = logging.getLogger(__name__).warning


def _read_band1(path):
    import tifffile
    a = tifffile.imread(path)
    if a.ndim == 3:                       # band axis = the smallest one
        a = np.take(a, 0, axis=int(np.argmin(a.shape)))
    v, nod = a.astype(np.float32), None
    if nod is not None:
        v[v == nod] = np.nan
    return v


def find_velocity_file(folder, shape=None, pattern='CNES_MV-LOS_radar_*rlks.tif*'):
    """MV-LOS radar file in the time-series folder (recursive fallback). If several, pick the one matching `shape`."""
    cands = sorted(glob.glob(os.path.join(folder, pattern))) or \
        sorted(glob.glob(os.path.join(folder, '**', pattern), recursive=True))
    if not cands:
        raise FileNotFoundError(f'no {pattern} in {folder}')
    if len(cands) > 1 and shape is not None:
        good = [c for c in cands if _read_band1(c).shape == tuple(shape)]
        cands = good or cands
    return cands[0]


def load_velocity(base_dir, line, shape, wavelength_mm, positive_away=True, mask=None):
    """Mean LOS velocity in mm/yr on the cube grid, same sign convention as the cube.

    FLATSIM MV-LOS band 1 is in rad/yr, positive away from the satellite.
    mm = rad * wavelength / (4 pi).  mask: True = invalid (as in denoise_series).
    """
    folder = os.path.join(base_dir, line.strip())
    path = find_velocity_file(folder, shape)
    v = _read_band1(path)
    if v.shape != tuple(shape):
        raise ValueError(f'velocity grid {v.shape} != cube grid {tuple(shape)} ({path}); '
                         'resample or crop it the same way as the cube')
    v = v * (wavelength_mm / (4 * np.pi))
    if not positive_away:
        v = -v
    v[~np.isfinite(v)] = np.nan
    if mask is not None:
        v[np.asarray(mask).astype(bool)] = np.nan
    log(f'velocity: {os.path.basename(path)}  median {np.nanmedian(v):.2f} mm/yr, '
        f'std {np.nanstd(v):.2f} mm/yr')
    return v


def raw_velocity(cube, dates_years, sub=8):
    """NaN-aware per-pixel linear regression slope of cube[::, ::sub, ::sub] (units/yr)."""
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
    """Compare the supplied velocity with the regression of the raw cube. Catches wrong units, sign, orientation."""
    vr = raw_velocity(cube_mm, dates_years, sub)
    vf = vel_mm[::sub, ::sub]
    ok = np.isfinite(vr) & np.isfinite(vf)
    if ok.sum() < 100:
        warn('velocity sanity: too few common pixels')
        return None
    a, b = vf[ok], vr[ok]
    if a.std() < 1e-6 or b.std() < 1e-6:                      # e.g. all-zero frames: nothing to correlate
        res = dict(corr=np.nan, slope=np.nan, median_diff=float(np.median(b - a)), n=int(ok.sum()))
    else:
        res = dict(corr=float(np.corrcoef(a, b)[0, 1]), slope=float(np.polyfit(a, b, 1)[0]),
                   median_diff=float(np.median(b - a)), n=int(ok.sum()))
    log(f'velocity sanity: corr={res["corr"]:.3f} slope(raw/FLATSIM)={res["slope"]:.3f} '
        f'median diff={res["median_diff"]:.3f} mm/yr (n={res["n"]})')
    if np.isfinite(res['corr']) and (res['corr'] < min_corr or abs(res['slope'] - 1) > 0.3):
        warn('velocity sanity FAILED: check units (rad vs mm), sign, or orientation of the tiff')
    return res
