import importlib


_MODEL_MODULES = {
    "DAND_GTN",
}

__all__ = sorted(_MODEL_MODULES)


def __getattr__(name):
    if name in _MODEL_MODULES:
        module = importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
