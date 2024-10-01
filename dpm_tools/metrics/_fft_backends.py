import warnings
import multiprocessing
import numpy as np

def _get_backend(backend='auto'):

    try:
        import cupy as cp
        _gpu_available = cp.cuda.runtime.getDeviceCount() > 0
    except ImportError:
        warnings.warn("CuPy is not installed. Falling back to pyFFTW.", ImportWarning)
        cp = None
        _gpu_available = False
    except Exception as e:
        warnings.warn(f"CuPy import error: {e}. Falling back to pyFFTW.", ImportWarning)
        cp = None
        _gpu_available = False

    if backend == "cuda" and cp is not None and _gpu_available:
        print("Using CuPy for FFT")
        fftn, ifftn = cp.fft.fftn, cp.fft.ifftn
        to_numpy = cp.asnumpy
        get_array = cp.asarray
        arrlib = cp

    elif backend == "cpu" or backend == 'pyfftw' or cp is None or not _gpu_available:
        print("Using pyFFTW for FFT")
        import pyfftw
        pyfftw.config.NUM_THREADS = multiprocessing.cpu_count() - 1
        fftn, ifftn = pyfftw.interfaces.numpy_fft.fftn, pyfftw.interfaces.numpy_fft.ifftn
        to_numpy = np.asarray
        get_array = np.asarray
        arrlib = np
    elif backend == "auto":
        if cp is not None and _gpu_available:
            print("Using CuPy for FFT")
            fftn, ifftn = cp.fft.fftn, cp.fft.ifftn
            to_numpy = cp.asnumpy
            get_array = cp.asarray
            arrlib = cp
        else:
            print("Using pyFFTW for FFT")
            import pyfftw
            pyfftw.config.NUM_THREADS = multiprocessing.cpu_count() - 1
            fftn, ifftn = pyfftw.interfaces.numpy_fft.fftn, pyfftw.interfaces.numpy_fft.ifftn
            to_numpy = np.asarray
            get_array = np.asarray
            arrlib = np

    else:
        raise ValueError("Invalid FFT library choice.")

    return fftn, ifftn, to_numpy, get_array, arrlib