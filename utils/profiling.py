import functools
import logging
import os
import sys
import time
from contextlib import contextmanager, nullcontext

try:
    import psutil

    _proc = psutil.Process()


    def rss_gb():
        return _proc.memory_info().rss / 1e9
except ImportError:
    def rss_gb():
        return float('nan')

logger = logging.getLogger('insar')
_NULL = nullcontext()


def setup_logging(level='INFO', logfile=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=getattr(logging, level.upper()), handlers=handlers, force=True,
                        format='%(asctime)s [pid %(process)d] %(message)s', datefmt='%H:%M:%S')


class Profiler:
    def __init__(self, enabled=False, track_memory=False):
        self.enabled, self.track_memory = enabled, track_memory
        self.reset()

    def reset(self):
        self.total, self.calls, self.last, self.mem = {}, {}, {}, {}
        self._stack = []

    def section(self, label):
        return self._section(label) if self.enabled else _NULL

    @contextmanager
    def _section(self, label):
        self._stack.append(label)
        path = '/'.join(self._stack)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            d = time.perf_counter() - t0
            self._stack.pop()
            self.total[path] = self.total.get(path, 0.) + d
            self.calls[path] = self.calls.get(path, 0) + 1
            self.last[label] = d
            if self.track_memory:
                self.mem[path] = max(self.mem.get(path, 0.), rss_gb())

    def profiled(self, label=None):
        def deco(fn):
            name = label or fn.__name__

            @functools.wraps(fn)
            def wrapper(*a, **k):
                if not self.enabled:
                    return fn(*a, **k)
                with self._section(name):
                    return fn(*a, **k)

            return wrapper

        return deco

    def get(self, label):
        """Duration of the last call of `label` (NaN if profiling is off)."""
        return self.last.get(label, float('nan'))

    def report(self):
        if not self.enabled or not self.total:
            return
        top = sum(v for p, v in self.total.items() if '/' not in p) or 1.
        lines = [f'{"section":<30s}{"calls":>6s}{"total s":>10s}{"mean s":>9s}{"% top":>7s}']
        for p in sorted(self.total):
            name = '  ' * p.count('/') + p.split('/')[-1]
            t, n = self.total[p], self.calls[p]
            mem = f'   rss {self.mem[p]:.1f} GB' if p in self.mem else ''
            lines.append(f'{name:<30s}{n:>6d}{t:>10.1f}{t / n:>9.2f}{100 * t / top:>7.1f}{mem}')
        logger.info('time breakdown:\n' + '\n'.join(lines))

    def as_dict(self):
        return {p: dict(total_s=round(t, 3), calls=self.calls[p]) for p, t in self.total.items()}


prof = Profiler(enabled=os.environ.get('INSAR_PROFILE', '0') == '1')


def configure(profile=None, track_memory=None):
    if profile is not None:
        prof.enabled = bool(profile)
    if track_memory is not None:
        prof.track_memory = bool(track_memory)


def log(msg):
    logger.info(f'[{rss_gb():.1f} GB] {msg}' if prof.track_memory else msg)
