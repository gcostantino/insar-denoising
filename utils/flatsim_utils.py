import glob
import os

import numpy as np
import tifffile


def _find(folder, pattern):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'Folder not found: {folder}')
    m = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(m) != 1:
        raise FileNotFoundError(f'{pattern} in {folder}: matches={m}; '
                                f'folder contains: {sorted(os.listdir(folder))[:30]}')
    return m[0]


def _find_any(folder, patterns):
    """Try each pattern in order; return the first that matches exactly one file."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'Folder not found: {folder}')
    tried = {}
    for pat in patterns:
        m = sorted(glob.glob(os.path.join(folder, pat)))
        tried[pat] = [os.path.basename(x) for x in m]
        if len(m) == 1:
            return m[0]
    raise FileNotFoundError(f'No unique match in {folder}. Tried: {tried}. '
                            f'Folder contains: {sorted(os.listdir(folder))[:30]}')


def _find_first(folders, pattern):
    """Exactly one match for `pattern`, searching the folders in order (time-series folder, then DAUX)."""
    tried = []
    for fo in folders:
        if os.path.isdir(fo):
            m = sorted(glob.glob(os.path.join(fo, pattern)))
            if len(m) == 1:
                return m[0]
            tried.append((fo, [os.path.basename(x) for x in m]))
    raise FileNotFoundError(f'{pattern}: no unique match. Searched: {tried}')


def load_mean_velocity(folders, shape, mask, wavelength_mm, sign):
    """FLATSIM mean LOS velocity (band 1 of CNES_MV-LOS_radar_8rlks.tiff) in mm/yr.

    The tiff is in rad/yr, positive away from the satellite; same conversion and sign as the cube.
    Returns (velocity, velfile). Pixels masked in the cube (all-NaN) are set to NaN.
    """
    velfile = _find_first(folders, 'CNES_MV-LOS_radar_8rlks.tiff')
    v = tifffile.imread(velfile)
    if v.ndim == 3:  # (ny, nx, bands), band 1 = velocity
        v = v[:, :, 0] if v.shape[-1] <= 4 else v[0]
    v = v.astype(np.float32)
    v[v == 9999.] = np.nan
    v = sign * (wavelength_mm * v) / (4 * np.pi)  # rad/yr -> mm/yr
    v[~np.isfinite(v)] = np.nan
    if v.shape != tuple(shape):
        raise ValueError(f'velocity grid {v.shape} != cube grid {tuple(shape)} ({velfile})')
    v[mask] = np.nan
    return v, velfile


def reference_cube(c):
    """Subtract each pixel's first finite value (in place); returns the index of that epoch (-1 if none)."""
    first = np.full(c.shape[1:], -1, np.int32)
    ref = np.zeros(c.shape[1:], np.float32)
    for t in range(c.shape[0]):
        m = (first < 0) & np.isfinite(c[t])
        first[m] = t
        ref[m] = c[t][m]
    c -= ref[None]
    return first


def flag_frozen(cube, first, per_epoch=True):
    """After referencing, epochs equal to the reference value (exact 0) carry no information."""
    T = cube.shape[0]
    dead = np.ones(cube.shape[1:], bool)
    frac = np.zeros(T)
    for t in range(T):
        z = (cube[t] == 0) & (first != t)
        frac[t] = z.mean()
        if per_epoch:
            cube[t][z] = np.nan
        dead &= z | np.isnan(cube[t]) | (first == t)
    return dead, frac


def load_flatsim(base_dir, line, rms_thresh=10., abs_thresh_mm=300., wavelength_mm=55.5, positive_away=True,
                 with_velocity=False, empty_mode='mask'):
    """line = 'region/NSBAS_TS-PKG_...'. Returns cube (T,ny,nx) in mm + auxiliary data."""
    region, product = os.path.split(line.strip())
    folder = os.path.join(base_dir, region, product)
    aux = os.path.join(base_dir, region, product.replace('NSBAS_TS-PKG', 'NSBAS_DAUX-PKG', 1))
    sign = 1 if positive_away else -1

    # tsfile = _find(folder, 'CNES_DTs_radar_*8rlks.tiff')
    tsfile = _find_any(folder, ['CNES_DTs_radar_8rlks.tiff', 'CNES_DTs_radar_*_8rlks.tiff'])
    rmsfile = _find(folder, 'CNES_Net_radar_8rlks.tiff')
    timefile = os.path.join(folder, 'RMSdate.txt')
    lonlatfile = _find(aux, 'CNES_Lut_radar_8rlks.tiff')
    topofile = _find(aux, 'CNES_DEM_radar_8rlks.tiff')
    losfile = _find(aux, 'CNES_CosENU_radar_8rlks.tiff')  # _find(aux, 'CNES_CosENU_geo_8rlks.tiff')

    rawts = tifffile.imread(tsfile).astype(np.float32)
    rawts[rawts == 9999.] = np.nan
    rawts = sign * (wavelength_mm * rawts) / (4 * np.pi)

    datestrings = np.loadtxt(timefile, usecols=1, dtype=str)
    dates = np.array([int(s) for s in datestrings], dtype=np.int32)  # YYYYMMDD
    T = len(dates)
    if rawts.shape[-1] == T:
        rawts = np.moveaxis(rawts, -1, 0)
    elif rawts.shape[0] != T:
        raise ValueError(f'No axis of length {T} in {rawts.shape}')
    cube = np.ascontiguousarray(rawts)

    lonlat = tifffile.imread(lonlatfile)
    lon, lat = lonlat[..., 0], lonlat[..., 1]
    topo = tifffile.imread(topofile)[:, :, 0].astype(np.float32)
    rms = (wavelength_mm * tifffile.imread(rmsfile)[:, :, 0].astype(np.float32)) / (4 * np.pi)
    los = tifffile.imread(losfile).astype(np.float32)
    los[los == 0.] = np.nan
    los = sign * los

    assert lon.shape == lat.shape == topo.shape == rms.shape == cube.shape[1:], \
        (lon.shape, topo.shape, rms.shape, cube.shape)

    with np.errstate(invalid='ignore'):
        cube[:, rms > rms_thresh] = np.nan
        cube[np.abs(cube) > abs_thresh_mm] = np.nan

    first = reference_cube(cube)  # subtract each pixel's first finite epoch
    empty = np.zeros(cube.shape[1:], bool)
    if empty_mode != 'keep':
        empty, _ = flag_frozen(cube, first)
        if empty_mode == 'mask':
            r, k = np.nonzero(empty & (first >= 0))
            cube[first[r, k], r, k] = np.nan  # reference epoch too: all-NaN, so masked
    mask = np.isnan(cube).all(axis=0)
    if empty_mode == 'noise':  # mask stays False; fill_cube_inplace adds the noise
        cube[:, empty & ~mask] = np.nan

    velocity, velfile = None, ''
    if with_velocity:
        velocity, velfile = load_mean_velocity([folder, aux], cube.shape[1:], mask, wavelength_mm, sign)

    se, sn, su = (float(np.nanmean(los[..., i])) for i in range(3))
    inc = float(np.rad2deg(np.arccos(-su)))
    look_az = float(np.rad2deg(np.arctan2(se, sn)) % 360)
    heading = (look_az - 90) % 360
    orbit = 'ascending' if np.cos(np.deg2rad(heading)) > 0 else 'descending'

    meta = dict(key=f'{region}/{product}', region=region, product=product, wvl_mm=wavelength_mm,
                positive_away=positive_away, rms_thresh=rms_thresh, abs_thresh=abs_thresh_mm,
                se=se, sn=sn, su=su, incidence_deg=inc, heading_deg=heading, orbit=orbit,
                tsfile=tsfile, velfile=velfile)
    return dict(cube=cube, mask=mask, topo=topo, lon=lon, lat=lat, los=los, rms=rms,
                dates=dates, meta=meta, velocity=velocity)
