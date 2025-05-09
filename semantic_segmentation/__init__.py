from importlib import import_module

__all__ = ["get_model"]

_MODEL_REGISTRY = {
    "hybrid_u": "segbench.models.hybrid_u:build",
    "deeplabv3": "segbench.models.deeplabv3:build",
    "fcn8s": "segbench.models.fcn8s:build",
    "segnet": "segbench.models.segnet:build",
}

def get_model(name: str, **kwargs):
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")
    module_path, func = _MODEL_REGISTRY[name].split(":")
    build_fn = getattr(import_module(module_path), func)
    return build_fn(**kwargs)
