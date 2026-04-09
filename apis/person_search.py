"""
apis/person_search.py
-----------------
person_search resource — manages image-based identity search via Qdrant.

Routes registered in app.py:
    POST /person_search/search   → upload an image to find top similar identities
"""

from fastapi import HTTPException, UploadFile, File, Form
from typing import Dict, Any

from apis.base import BaseResource
from services.person_search import PersonSearchTask
from logger.logger_config import Logger

log = Logger.get_logger(__name__)

# ─────────────────────────────────────────────
# person_search resource
# ─────────────────────────────────────────────
class PersonSearchResource(BaseResource):
    def __init__(self):
        super().__init__()
        log.info("[PERSON_SEARCH API] Initializing Person Search Service...")
        
        # 1. Define the default task configuration
        default_config = {
            "taskId": 1,
            "taskName": "Main Camera Person Indexer",
            "algorithmType": "PERSON_SEARCH",
            "channelId": 101,
            "enable": True,
            "detailConfig": {
                "padding": 10
            },
            "validWeekday": [
                "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", 
                "FRIDAY", "SATURDAY", "SUNDAY"
            ],
            "validStartTime": 0,
            "validEndTime": 86400000
        }

        # 2. Pass the config to the task initializer
        self.person_search_task = PersonSearchTask(default_config)

    async def search(self, file: UploadFile = File(...), top_k: int = Form(10)) -> Dict[str, Any]:
        """
        Takes an uploaded image file, extracts bytes, and searches the vector DB.
        """
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        try:
            image_bytes = await file.read()
            
            # 3. Fixed reference (was self.reid_service)
            results = self.person_search_task.search_by_image(image_bytes, top_k=top_k)
            
            return {
                "status": "success",
                "count": len(results),
                "results": results
            }
            
        except Exception as e:
            log.error(f"[REID API] Search failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ── Singleton ─────────────────────────────────
person_search_api = PersonSearchResource()