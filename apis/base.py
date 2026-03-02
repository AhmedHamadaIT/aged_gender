"""
apis/base.py
------------
BaseResource — base class for all API resources.
"""

from fastapi import HTTPException


class BaseResource:

    def __init__(self):
        self.class_instance = {}

    def get_service(self, action: str):
        service = self.class_instance.get(action)
        if service is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown action '{action}'. Available: {list(self.class_instance.keys())}"
            )
        return service

    def on_post(self, req):
        raise NotImplementedError

    def on_get(self):
        raise NotImplementedError