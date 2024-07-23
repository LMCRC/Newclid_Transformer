from __future__ import annotations
from typing import Any, Callable


def raise_if_called(missing_dep: str) -> Callable[..., Any]:
    def _raise_if_called(*args : Any, **kwargs : Any):
        raise ImportError("Missing optional dependency: %s", missing_dep)

    return _raise_if_called


def raise_if_instanciated(missing_dep: str) -> object:
    class RaiseIfInstanciated:
        def __init__(self, *args : Any, **kwargs : Any) -> None:
            raise ImportError("Missing optional dependency: %s", missing_dep)

    return RaiseIfInstanciated
