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


def load_flatsim(base_dir, line, rms_thresh=10., abs_thresh_mm=300., wavelength_mm=55.5, positive_away=True):
    """line = 'region/NSBAS_TS-PKG_...'. Returns cube (T,ny,nx) in mm + auxiliary data."""
    region, product = os.path.split(line.strip())
    folder = os.path.join(base_dir, region, product)
    aux = os.path.join(base_dir, region, product.replace('NSBAS_TS-PKG', 'NSBAS_DAUX-PKG', 1))
    sign = 1 if positive_away else -1

    tsfile = _find(folder, 'CNES_DTs_radar_*_8rlks.tiff')
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
    mask = np.isnan(cube).all(axis=0)

    se, sn, su = (float(np.nanmean(los[..., i])) for i in range(3))
    inc = float(np.rad2deg(np.arccos(-su)))
    look_az = float(np.rad2deg(np.arctan2(se, sn)) % 360)
    heading = (look_az - 90) % 360
    orbit = 'ascending' if np.cos(np.deg2rad(heading)) > 0 else 'descending'

    meta = dict(key=f'{region}/{product}', region=region, product=product, wvl_mm=wavelength_mm,
                positive_away=positive_away, rms_thresh=rms_thresh, abs_thresh=abs_thresh_mm,
                se=se, sn=sn, su=su, incidence_deg=inc, heading_deg=heading, orbit=orbit,
                tsfile=tsfile)
    return dict(cube=cube, mask=mask, topo=topo, lon=lon, lat=lat, los=los, rms=rms,
                dates=dates, meta=meta)
