"""
services/base.py
----------------
BaseService — base class for all services.
Services are callable and implement __call__().
"""


class BaseService:
    """
    Base class for all services.

    Each service implements __call__() with its own signature.
    Services are stateless by default — instantiated per call
    or once and reused depending on the resource.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement __call__()")