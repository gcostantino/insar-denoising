#!/usr/bin/env python
"""
Build the corrected time-series cube (one file per orbit) from the offsets of solve_frame_offsets.py.

For every frame: cell-mean displacement (the mosaic grid of build_disp_mosaic.py, cell = native pixel size),
linear interpolation in time onto the regular grid of the offsets file, minus e_i(t). Where several frames cover a
cell: median over the frames at each epoch. Optional per-cell re-zeroing over a reference period.

Output <prefix>_<orbit>.h5
  u          int16 (epochs, rows, cols): mm / scale_mm, -32768 = no data. Row r <-> cell iy = iy0 + r (south to north),
             column c <-> cell ix = ix0 + c; cell (ix, iy) spans lon [-180 + ix*res, ...), lat [-90 + iy*res, ...).
  n_frames   int8 (epochs, rows, cols): number of frames that contributed (composition changes = possible steps)
  time/ordinal, time/yyyymmdd, lon, lat (cell centres), frames/* (label, path, uncorrected dates), attrs.
Chunks hold all epochs of 32 x 32 cells (fast time series at a point); empty chunks take no space.

Processing is tile by tile (--tile cells), so memory is bounded. --extent restricts the cube to a box (test on a small
one first, and use --dry-run to see the size). Frames that you want to ignore: --exclude (substrings of the label).

Examples:
  python build_cube.py off_levant.h5 --out-prefix cube_levant --dry-run
  python build_cube.py off_levant.h5 --out-prefix cube_levant --extent 35.5 36.5 32 33
  python build_cube.py off_tibet.h5 --out-prefix cube_tibet --exclude D004 D135NORD --ref-date 2015-10-22
Needs build_disp_mosaic.py in the same folder.
"""
import argparse
import os
import time
import warnings
from datetime import datetime

import h5py
import numpy as np
import scipy.sparse as sp

from build_disp_mosaic import LON0, LAT0, parse_dates, pixel_cells

FILL = -32768


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def read_offsets(path):
    with h5py.File(path, 'r') as f:
        fr = {k: np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f['frames'][k][...]])
              for k in ('path', 'orbit', 'key', 'region', 'label')}
        fr['first'], fr['last'] = f['frames']['first_ord'][...], f['frames']['last_ord'][...]
        return fr, f['time']['ordinal'][...].astype(np.int64), f['e'][...].astype(np.float64), float(f.attrs['res_deg'])


def fill_offsets(e, first, last, grid, policy):
    """Offsets to subtract. hold: carry the nearest valid value inside the span of the frame (frames that never got
    an offset are left uncorrected); keep: NaN -> 0 (uncorrected); drop: NaN stays NaN (the epoch is discarded)."""
    out, n_unc = e.copy(), np.zeros(len(e), int)
    for k in range(len(e)):
        inside = (grid >= first[k]) & (grid <= last[k])
        v, ok = e[k], np.isfinite(e[k]) & inside
        n_unc[k] = int((inside & ~np.isfinite(v)).sum())
        if policy == 'keep':
            out[k] = np.where(np.isfinite(v), v, 0.0)
        elif policy == 'hold':
            if not ok.any():
                out[k] = np.where(inside, 0.0, np.nan)
                continue
            idx = np.where(ok, np.arange(len(v)), -1)
            np.maximum.accumulate(idx, out=idx)
            idx[idx < 0] = np.argmax(ok)
            w = v[idx]
            w[~inside] = np.nan
            out[k] = w
    return out, n_unc


def load_geometry(path, res, nx_g):
    with h5py.File(path, 'r') as f:
        uniq, inv, ok, masked, _ = pixel_cells(f, res, nx_g)
        dates = parse_dates(f['dates'][...])
    flat = np.flatnonzero(ok.ravel())
    keep = ~masked.ravel()[flat]
    if not keep.any():
        return None
    cid = uniq[inv][keep]
    ix, iy = (cid % nx_g).astype(np.int32), (cid // nx_g).astype(np.int32)
    return dict(path=path, dates=dates, shape=ok.shape, pix=flat[keep].astype(np.int32), ix=ix, iy=iy,
                box=(int(ix.min()), int(ix.max()), int(iy.min()), int(iy.max())))


def tile_cell_series(g, sel, tx0, ty0, tw, th):
    """Cell-mean displacement of the pixels of frame g that fall in the tile -> (epochs, th*tw), NaN where none."""
    pix = g['pix'][sel].astype(np.int64)
    r, c = np.divmod(pix, g['shape'][1])
    r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
    loc = (r - r0) * (c1 - c0) + (c - c0)
    cidx = (g['iy'][sel] - ty0).astype(np.int64) * tw + (g['ix'][sel] - tx0)
    with h5py.File(g['path'], 'r') as f:
        blk = f['displacement'][:, r0:r1, c0:c1]
    v = blk.reshape(blk.shape[0], -1)[:, loc]
    fin = np.isfinite(v)
    M = sp.csr_matrix((np.ones(len(loc), np.float32), (cidx, np.arange(len(loc)))), shape=(th * tw, len(loc)))
    S = M @ np.where(fin, v, 0.0).astype(np.float32).T
    C = M @ fin.astype(np.float32).T
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(C > 0, S / C, np.nan).T.astype(np.float32)


def to_grid(A, dates, grid, drop_ends):
    """Linear interpolation of A (epochs, cells) onto the grid dates inside the span (minus drop_ends epochs)."""
    T = len(dates)
    if T < 2 + 2 * drop_ends:
        return None, np.zeros(0, int)
    gi = np.flatnonzero((grid >= dates[drop_ends]) & (grid <= dates[T - 1 - drop_ends]))
    if not len(gi):
        return None, gi
    k = np.clip(np.searchsorted(dates, grid[gi], 'right') - 1, 0, T - 2)
    den = (dates[k + 1] - dates[k]).astype(float)
    fr = np.where(den > 0, np.clip((grid[gi] - dates[k]) / den, 0, 1), 0.0)
    out = A[k].copy()
    m = (fr > 0) & (fr < 1)
    if m.any():
        out[m] = (1 - fr[m])[:, None].astype(np.float32) * A[k[m]] + fr[m][:, None].astype(np.float32) * A[k[m] + 1]
    m1 = fr == 1
    if m1.any():
        out[m1] = A[k[m1] + 1]
    return out, gi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('offsets')
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--orbit', choices=['ascending', 'descending', 'both'], default='both')
    ap.add_argument('--extent', type=float, nargs=4, metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'))
    ap.add_argument('--exclude', nargs='+', default=[], help='drop frames whose label contains any of these')
    ap.add_argument('--tile', type=int, default=256, help='cells per tile side (rounded to a multiple of 32)')
    ap.add_argument('--scale', type=float, default=0.1, help='mm per integer step of u')
    ap.add_argument('--missing', choices=['hold', 'keep', 'drop'], default='hold',
                    help='epochs where a frame has no offset: hold the nearest one / leave uncorrected / discard')
    ap.add_argument('--drop-ends', type=int, default=0, help='discard the first/last N epochs of every frame')
    ap.add_argument('--ref-date', default=None, help='YYYY-MM-DD: re-zero every cell on its mean over --ref-days from here')
    ap.add_argument('--ref-days', type=float, default=60.)
    ap.add_argument('--keep-unreferenced', action='store_true', help='cells without data in the reference period stay as they are')
    ap.add_argument('--limit-tiles', type=int, default=None, help='testing')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    fr, grid, e, res = read_offsets(a.offsets)
    nx_g = int(np.ceil(360.0 / res))
    Tg = len(grid)
    tile = max(32, int(round(a.tile / 32)) * 32)
    e_eff, n_unc = fill_offsets(e, fr['first'], fr['last'], grid, a.missing)
    ref_idx = None
    if a.ref_date:
        t0 = datetime.strptime(a.ref_date, '%Y-%m-%d').toordinal()
        ref_idx = np.flatnonzero((grid >= t0) & (grid < t0 + a.ref_days))
        if not len(ref_idx):
            raise SystemExit('no grid date in the reference period')
    orbits = ['ascending', 'descending'] if a.orbit == 'both' else [a.orbit]
    for orbit in orbits:
        fidx = [k for k in range(len(fr['key'])) if fr['orbit'][k] == orbit
                and not any(s in fr['label'][k] for s in a.exclude)]
        if not fidx:
            continue
        log(f'{orbit}: {len(fidx)} frames, cell {res} deg, {Tg} epochs')
        geo = {}
        for k in fidx:
            g = load_geometry(fr['path'][k], res, nx_g)
            if g is None:
                log(f'  skipped (no unmasked cell): {fr["label"][k]}')
                continue
            geo[k] = g
            log(f'  {fr["label"][k]}: {len(g["pix"])} pixels in {len(np.unique(g["iy"].astype(np.int64) * nx_g + g["ix"]))} cells, '
                f'{int(n_unc[k])} epochs without offset')
        if not geo:
            continue
        ix0, ix1 = min(g['box'][0] for g in geo.values()), max(g['box'][1] for g in geo.values())
        iy0, iy1 = min(g['box'][2] for g in geo.values()), max(g['box'][3] for g in geo.values())
        if a.extent:
            ix0, ix1 = max(ix0, int(np.floor((a.extent[0] - LON0) / res))), min(ix1, int(np.floor((a.extent[1] - LON0) / res)))
            iy0, iy1 = max(iy0, int(np.floor((a.extent[2] - LAT0) / res))), min(iy1, int(np.floor((a.extent[3] - LAT0) / res)))
        nx, ny = ix1 - ix0 + 1, iy1 - iy0 + 1
        if nx <= 0 or ny <= 0:
            log(f'  {orbit}: empty extent')
            continue
        nty, ntx = int(np.ceil(ny / tile)), int(np.ceil(nx / tile))
        log(f'  raster {ny} x {nx} cells, {Tg} epochs: {Tg * ny * nx * 3 / 1e9:.1f} GB uncompressed upper bound, {nty * ntx} tiles')
        if a.dry_run:
            continue
        out_path = f'{a.out_prefix}_{orbit}.h5'
        tmp = out_path + '.tmp'
        chunk = (Tg, min(32, ny), min(32, nx))
        kw = dict(compression='gzip', compression_opts=4, shuffle=True)
        n_clip, t_start, done = 0, time.time(), 0
        with h5py.File(tmp, 'w') as out:
            U = out.create_dataset('u', shape=(Tg, ny, nx), dtype='i2', chunks=chunk, fillvalue=FILL, **kw)
            NF = out.create_dataset('n_frames', shape=(Tg, ny, nx), dtype='i1', chunks=chunk, fillvalue=0, **kw)
            for ty in range(nty):
                for tx in range(ntx):
                    if a.limit_tiles is not None and done >= a.limit_tiles:
                        break
                    r0, c0 = ty * tile, tx * tile
                    th, tw = min(tile, ny - r0), min(tile, nx - c0)
                    gx0, gy0 = ix0 + c0, iy0 + r0
                    members = []
                    for k, g in geo.items():
                        b = g['box']
                        if b[1] < gx0 or b[0] >= gx0 + tw or b[3] < gy0 or b[2] >= gy0 + th:
                            continue
                        sel = (g['ix'] >= gx0) & (g['ix'] < gx0 + tw) & (g['iy'] >= gy0) & (g['iy'] < gy0 + th)
                        if sel.any():
                            members.append((k, sel))
                    if not members:
                        continue
                    stack = np.full((len(members), Tg, th * tw), np.nan, np.float32)
                    for q, (k, sel) in enumerate(members):
                        A = tile_cell_series(geo[k], sel, gx0, gy0, tw, th)
                        o, gi = to_grid(A, geo[k]['dates'], grid, a.drop_ends)
                        if o is None:
                            continue
                        stack[q, gi] = o - e_eff[k, gi][:, None].astype(np.float32)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        med = stack[0] if len(members) == 1 else np.nanmedian(stack, axis=0)
                        cnt = np.isfinite(stack).sum(0)
                        if ref_idx is not None:
                            ref = np.nanmean(med[ref_idx], axis=0)
                            med = med - ref[None, :] if not a.keep_unreferenced else np.where(np.isfinite(ref)[None, :], med - ref[None, :], med)
                    qv = np.rint(med / a.scale)
                    over = np.isfinite(qv) & (np.abs(qv) > 32766)
                    n_clip += int(over.sum())
                    qv = np.clip(qv, -32766, 32766)
                    qi = np.where(np.isfinite(qv), qv, FILL).astype(np.int16)
                    if (qi != FILL).any():
                        U[:, r0:r0 + th, c0:c0 + tw] = qi.reshape(Tg, th, tw)
                        NF[:, r0:r0 + th, c0:c0 + tw] = cnt.astype(np.int8).reshape(Tg, th, tw)
                    done += 1
                    log(f'  tile {ty + 1}/{nty},{tx + 1}/{ntx}: {len(members)} frames, {(qi != FILL).mean():.0%} valid '
                        f'({time.time() - t_start:.0f}s)')
            tg = out.create_group('time')
            tg.create_dataset('ordinal', data=grid)
            tg.create_dataset('yyyymmdd', data=np.array([int(datetime.fromordinal(int(t)).strftime('%Y%m%d')) for t in grid]))
            out.create_dataset('lon', data=LON0 + (ix0 + np.arange(nx) + 0.5) * res)
            out.create_dataset('lat', data=LAT0 + (iy0 + np.arange(ny) + 0.5) * res)
            fg = out.create_group('frames')
            ks = list(geo)
            for key in ('label', 'path', 'key', 'region'):
                fg.create_dataset(key, data=np.array([fr[key][k] for k in ks], dtype=object), dtype=h5py.string_dtype())
            fg.create_dataset('epochs_without_offset', data=n_unc[ks])
            out.attrs.update(dict(
                res_deg=res, lon0=LON0, lat0=LAT0, nx_global=nx_g, ix0=ix0, iy0=iy0, scale_mm=a.scale, fill_value=FILL,
                units='mm', orbit=orbit, offsets_file=os.path.abspath(a.offsets), missing=a.missing,
                drop_ends=a.drop_ends, ref_date=a.ref_date or '', ref_days=a.ref_days, n_clipped=n_clip,
                convention='LOS displacement, positive away from the satellite; corrected = u_i(t) - e_i(t); median over frames',
                grid_note='row r <-> iy = iy0 + r (south to north), column c <-> ix = ix0 + c',
                created=datetime.now().isoformat(), complete=1))
        os.replace(tmp, out_path)
        log(f'saved {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB), {n_clip} values clipped')


if __name__ == '__main__':
    main()
