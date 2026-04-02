"""
apis/reid.py
-----------------
ReID resource — manages image-based identity search via Qdrant.

Routes registered in app.py:
    POST /reid/search   → upload an image to find top similar identities
"""

from fastapi import HTTPException, UploadFile, File, Form
from typing import Dict, Any

from apis.base import BaseResource
from services.reid import ReIDService
from logger.logger_config import Logger

log = Logger.get_logger(__name__)

# ─────────────────────────────────────────────
# ReID resource
# ─────────────────────────────────────────────
class ReidResource(BaseResource):
    def __init__(self):
        super().__init__()
        # Initialize service once so the model stays in memory
        log.info("[REID API] Initializing ReID Service...")
        self.reid_service = ReIDService()

    async def search(self, file: UploadFile = File(...), top_k: int = Form(10)) -> Dict[str, Any]:
        """
        Takes an uploaded image file, extracts bytes, and searches the vector DB.
        """
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        try:
            image_bytes = await file.read()
            
            # Call the search_by_image method added previously
            results = self.reid_service.search_by_image(image_bytes, top_k=top_k)
            
            return {
                "status": "success",
                "count": len(results),
                "results": results
            }
            
        except Exception as e:
            log.error(f"[REID API] Search failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ── Singleton ─────────────────────────────────
reid_api = ReidResource()