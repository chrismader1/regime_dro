# io.py

import gzip
import pickle


class _NPCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_out(path: str) -> dict:
    """
    Load dict saved by save_out (handles .pkl or .pkl.gz, regardless of extension).
    """
    with open(path, "rb") as raw:
        head = raw.read(2)
        raw.seek(0)

        fp = gzip.GzipFile(fileobj=raw) if head == b"\x1f\x8b" else raw

        try:
            return pickle.load(fp)
        except ModuleNotFoundError as e:
            if "numpy._core" in str(e):
                try:
                    fp.seek(0)
                except Exception:
                    raw.seek(0)
                    fp = gzip.GzipFile(fileobj=raw) if head == b"\x1f\x8b" else raw
                return _NPCompatUnpickler(fp).load()
            raise


def save_out(out: dict, path: str):
    """
    Save `out` dict to `path`. Use .pkl or .pkl.gz.
    """
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
