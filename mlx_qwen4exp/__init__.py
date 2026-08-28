"""mlx_qwen4exp — MLX port of Qwen3.8-Flash-Next (qwen4_exp).

Public API::

    from mlx_qwen4exp import ModelArgs, Model

``ModelArgs`` is cheap (a dataclass in ``config``). ``Model`` pulls in mlx / mlx_lm and
the five sub-modules, so it is imported lazily via ``__getattr__`` to keep
``import mlx_qwen4exp`` (and just reading ``ModelArgs``) fast and side-effect-light.
"""

from .config import ModelArgs

__all__ = ["ModelArgs", "Model"]


def __getattr__(name: str):
    # PEP 562 lazy attribute access: only import the heavy model module on first use.
    if name == "Model":
        from .model import Model

        return Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["Model"])
