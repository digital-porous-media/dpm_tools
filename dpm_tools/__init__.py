import io
import visualization

def _get_version():
    from __version__ import __version__
    suffix = "dev"
    if __version__.endswith(suffix):
        __version__ = __version__[:-len(suffix)]
    return __version__

__version__ = _get_version()
print(__version__)
