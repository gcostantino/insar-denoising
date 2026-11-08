import argparse
import datetime as dt
import glob
import json
import os
import time
import traceback

import h5py
import numpy as np

from config.local_denoiser_params import denoiser_configuration
from dataset.insarinferencedataset import InSARInferenceDataset
from kito import Engine
from model.insar_denoiser import InSARDenoiser
from utils.flatsim_utils import load_flatsim
from utils.space_time_reconstruction import denoise_series

WVL, RMS_THRESH, ABS_THRESH, POSITIVE_AWAY = 55.5, 10., 300., True  # abs thresh, though it should depend on the TS

UNIT_SCALE = 0.1   # mm -> cm (InSARDenoiser units)

DENOISE_CFG = dict(N=128, Nt=9, spatial_stride=64, t_stride=2, stat='median',
                   temporal_taper='hann', taper='hann', edge_trim=8,
                   min_valid_frac=0.0, anchor_weight=0.0, correct=True)


def _ds(f, name, arr, chunks=None):
    f.create_dataset(name, data=arr, chunks=chunks, compression='gzip',
                     compression_opts=4, shuffle=True)


def save_series(h5path, data, U, cfg, info, runtime_s):
    """Atomic write: tmp file, 'complete' attribute last, then rename."""
    os.makedirs(os.path.dirname(h5path), exist_ok=True)
    tmp = h5path + '.tmp'
    T, ny, nx = U.shape
    with h5py.File(tmp, 'w') as f:
        _ds(f, 'displacement', U, chunks=(min(T, 32), min(ny, 64), min(nx, 64)))
        f.create_dataset('dates', data=data['dates'])
        c2 = (min(ny, 256), min(nx, 256))
        for name in ('lon', 'lat', 'topography', 'rms'):
            src = {'topography': data['topo']}.get(name, data.get(name))
            _ds(f, name, src, chunks=c2)
        _ds(f, 'mask', data['mask'].astype(np.uint8), chunks=c2)
        _ds(f, 'los', data['los'], chunks=(min(ny, 128), min(nx, 128), 3))

        f['displacement'].attrs['units'] = 'mm'
        f['displacement'].attrs['reference'] = 'zero at dates[0]; positive away from satellite' \
            if data['meta']['positive_away'] else 'zero at dates[0]; positive toward satellite'
        f['dates'].attrs['format'] = 'YYYYMMDD'
        f['los'].attrs['bands'] = 'E, N, U components of the LOS unit vector (sign-adjusted)'
        for k, v in data['meta'].items():
            f.attrs[k] = v

        for k in ('N', 'Nt', 'spatial_stride', 't_stride', 'stat', 'taper', 'temporal_taper',
                  'edge_trim', 'min_valid_frac', 'anchor_weight', 'correct'):
            v = cfg[k]
            f.attrs[f'cfg_{k}'] = int(v) if isinstance(v, bool) else v
        f.attrs['weights'] = str(denoiser_configuration.model.weight_load_path)

        f.attrs['denoiser_cfg'] = json.dumps(cfg, default=str)
        f.attrs['denoiser_unit_scale'] = UNIT_SCALE
        f.attrs['n_windows_per_chunk'] = info['n_windows']
        f.attrs['n_chunks'] = len(info['t_starts'])
        f.attrs['runtime_s'] = runtime_s
        f.attrs['timestamp'] = dt.datetime.now().isoformat()
        f.attrs['complete'] = 1  # written last
    os.replace(tmp, h5path)


def is_complete(h5path):
    try:
        with h5py.File(h5path, 'r') as f:
            return bool(f.attrs.get('complete', 0))
    except OSError:
        return False


def process_series(line, base_dir, out_dir, engine, dataset_cls, cfg=DENOISE_CFG, overwrite=False):
    region, product = os.path.split(line.strip())
    h5path = os.path.join(out_dir, region, product + '.h5')
    if not overwrite and is_complete(h5path):
        return 'skipped'
    t0 = time.time()
    data = load_flatsim(base_dir, line, rms_thresh=RMS_THRESH, abs_thresh_mm=ABS_THRESH,
                        wavelength_mm=WVL, positive_away=POSITIVE_AWAY)

    if data['cube'].shape[0] < cfg['Nt']:
        return f'too short ({data["cube"].shape[0]} frames)'
    print(f'[{line.strip()}] cube {data["cube"].shape}, orbit {data["meta"]["orbit"]}')
    data['cube'] *= UNIT_SCALE  # mm to cm, as InSARDenoiser wants cm
    U, info = denoise_series(data['cube'], data['topo'], engine, dataset_cls,
                             mask=data['mask'], **cfg)
    U /= UNIT_SCALE  # back to mm, so the saved units attribute stays correct
    save_series(h5path, data, U, cfg, info, time.time() - t0)
    return f'done in {time.time() - t0:.0f}s'


def build_master(out_dir, master_path):
    """master.h5 exposing all complete series as region/product via external links."""
    with h5py.File(master_path, 'w') as m:
        for p in sorted(glob.glob(os.path.join(out_dir, '**', '*.h5'), recursive=True)):
            if os.path.abspath(p) == os.path.abspath(master_path) or not is_complete(p):
                continue
            rel = os.path.relpath(p, os.path.dirname(os.path.abspath(master_path)))
            m[os.path.relpath(p, out_dir)[:-3]] = h5py.ExternalLink(rel, '/')

# example with 2 devices:
# CUDA_VISIBLE_DEVICES=0 python flatsim_denoising.py --out-dir results --shard 0/2 &
# CUDA_VISIBLE_DEVICES=1 python flatsim_denoising.py --out-dir results --shard 1/2 &
# wait

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default=os.path.expanduser('~/Desktop/flatsim_product_names_folders.txt'))
    ap.add_argument('--base-dir', default='/Users/giuseppe/Documents/DATA/InSAR/FLATSIM')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--cuda-idx', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--shard', default='0/1', help='i/n: process lines i, i+n, ... (array jobs)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--spatial-stride', type=int, default=128)
    ap.add_argument('--temporal-stride', type=int, default=8)
    ap.add_argument('--stat', choices=['mean', 'median'], default='median')
    ap.add_argument('--spatial-taper', choices=['hann', 'none'], default='hann', help='spatial taper')
    ap.add_argument('--temporal-taper', choices=['hann', 'uniform'], default='hann')
    ap.add_argument('--batch-windows', type=int, default=2048, help='batch windows')
    ap.add_argument('--batch-size-gpu', type=int, default=128, help='batch size of GPU')

    a = ap.parse_args()

    cfg = dict(DENOISE_CFG)
    cfg.update(spatial_stride=a.spatial_stride, t_stride=a.temporal_stride, stat=a.stat,
               taper=a.spatial_taper, temporal_taper=a.temporal_taper, batch_windows=a.batch_windows)

    max_stride = int(0.75 * (cfg['N'] - 2 * cfg['edge_trim']))  # 84 with defaults
    if cfg['correct'] and a.spatial_stride > max_stride:
        ap.error(f'--spatial-stride must be <= {max_stride} for the offset correction to have overlaps')

    tag = f'sstride{a.spatial_stride}_tstride{a.temporal_stride}_aggreg_{a.stat}_staper_{a.spatial_taper}-ttaper_{a.temporal_taper}'
    out_dir = os.path.join(a.out_dir, tag)

    denoiser_configuration.training.train_mode = False
    denoiser_configuration.model.save_inference_to_disk = False
    denoiser_configuration.training.device_type = a.device
    denoiser_configuration.training.device_id = a.cuda_idx
    denoiser_configuration.training.batch_size = a.batch_size_gpu

    engine = Engine(InSARDenoiser(denoiser_configuration), denoiser_configuration)  # built once
    engine.module.build()
    engine.load_weights(denoiser_configuration.model.weight_load_path)

    i, n = map(int, a.shard.split('/'))
    lines = [l.strip() for l in open(a.list) if l.strip()][i::n]
    os.makedirs(out_dir, exist_ok=True)
    for line in lines:
        try:
            print(line, '->', process_series(line, a.base_dir, out_dir, engine,
                                             InSARInferenceDataset, cfg=cfg, overwrite=a.overwrite))
        except Exception:
            with open(os.path.join(out_dir, f'errors_shard{i}.log'), 'a') as ef:
                ef.write(f'{line}\n{traceback.format_exc()}\n')
            print(line, '-> FAILED (see log)')
