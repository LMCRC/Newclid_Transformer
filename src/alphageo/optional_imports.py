from typing import Callable


def raise_if_called(missing_import: str) -> Callable:
    def _raise_if_called(*args, **kwargs):
        raise ImportError("Missing optional dependency: %s", missing_import)

    return _raise_if_called


def raise_if_instanciated(missing_import: str) -> object:
    class RaiseIfInstanciated:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Missing optional dependency: %s", missing_import)

    return RaiseIfInstanciated
