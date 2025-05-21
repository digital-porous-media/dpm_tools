from time import perf_counter_ns
import importlib

__all__ = ["metrics", "io", "segmentation", "visualization"]
from . import metrics
from . import io
from . import segmentation
from . import visualization

def _get_version():
    from .__version__ import __version__
    suffix = "dev"
    if __version__.endswith(suffix):
        __version__ = __version__[:-len(suffix)]
    return __version__

__version__ = _get_version()
# print(f'Version {__version__}')


def timer(func):
    def wrapper(*args, **kwargs):
        tic = perf_counter_ns()
        result = func(*args, **kwargs)
        toc = perf_counter_ns()
        print(f'Function {func.__name__!r} executed in {(toc - tic) * 1e-9:.3f} s')
        return result

    return wrapper
