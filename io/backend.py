import importlib

_BACKENDS = {
    'volumescope': 'emalign.io.volumescope'
}

def get_io_backend(mode):
    if mode not in _BACKENDS:
        raise ValueError(f"mode must be one of {list(_BACKENDS.keys())}, got '{mode}'")
    return importlib.import_module(_BACKENDS[mode])