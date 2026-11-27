import sys
import h5py
import numpy as np

def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])

mosaic, flt = sys.argv[1], sys.argv[2:]          # optional substrings, all must appear in the two keys
with h5py.File(mosaic, 'r') as f:
    if 'pairs' not in f:
        raise SystemExit('no /pairs: rebuild with --offset-mode rate or cumulative')
    fr = {k: dec(f['frames'][k][...]) for k in ('key', 'path', 'orbit')}
    p = f['pairs']
    fi, fj, n, win = (p[k][...] for k in ('frame_i', 'frame_j', 'n_cells', 'window_yr'))

for n_, i, j, w in sorted(zip(n, fi, fj, win), key=lambda r: -r[0]):
    a, b = fr['key'][i], fr['key'][j]
    if all(s in a + ' ' + b for s in flt):
        print(f'{n_:9d} cells  {w:5.2f} yr  {fr["orbit"][i][:4]}  {a}  |  {b}')
        print(f'    python overlap_diff_series.py "{fr["path"][i]}" "{fr["path"][j]}"')
