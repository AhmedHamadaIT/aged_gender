"""
apis/semantic_search.py
-----------------
semantic_search resource — manages text-to-image semantic search via Qdrant.

Routes registered in app.py:
    POST /semantic_search/search_text   → submit text query to find top similar images
"""

from fastapi import HTTPException, Form, UploadFile, File
from typing import Dict, Any, Optional

from apis.base import BaseResource
from services.semantic_search import SemanticSearchService
from logger.logger_config import Logger

log = Logger.get_logger(__name__)

_SERVICE_UNAVAILABLE_HEADERS = {"Retry-After": "120"}

# ─────────────────────────────────────────────
# semantic_search resource
# ─────────────────────────────────────────────
class SemanticSearchResource(BaseResource):
    def __init__(self):
        super().__init__()
        log.info("[SEMANTIC_SEARCH API] Initializing Semantic Search Service...")
        self.semantic_search_service = SemanticSearchService()

    def _require_semantic_ready(self) -> None:
        if not getattr(self.semantic_search_service, "_ready", False):
            raise HTTPException(
                status_code=503,
                detail="Semantic search is unavailable — ONNX / open_clip models not loaded.",
                headers=_SERVICE_UNAVAILABLE_HEADERS,
            )

    async def search_text(self, text_query: str = Form(...), top_k: int = Form(10)) -> Dict[str, Any]:
        """
        Takes a text query and searches the vector DB for matching images.
        """
        self._require_semantic_ready()
        if not text_query or not text_query.strip():
            raise HTTPException(status_code=400, detail="Text query cannot be empty.")

        try:
            results = self.semantic_search_service.search_by_text(text_query=text_query, top_k=top_k)

            return {
                "status": "success",
                "count": len(results),
                "results": results
            }

        except RuntimeError as e:
            log.warning(f"[SEMANTIC_SEARCH API] Search unavailable: {e}")
            raise HTTPException(
                status_code=503,
                detail=str(e),
                headers=_SERVICE_UNAVAILABLE_HEADERS,
            ) from e
        except Exception as e:
            log.error(f"[SEMANTIC_SEARCH API] Text search failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e

    async def search(
        self,
        text_query: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        top_k: int = Form(10),
    ) -> Dict[str, Any]:
        """Search by text and/or by query image (at least one required)."""
        self._require_semantic_ready()
        tq = (text_query or "").strip()
        if not tq and file is None:
            raise HTTPException(
                status_code=400,
                detail="Provide text_query and/or an image file.",
            )
        if not tq and file is not None:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
            image_bytes = await file.read()
            try:
                results = self.semantic_search_service.search_by_image(image_bytes, top_k=top_k)
            except RuntimeError as e:
                raise HTTPException(
                    status_code=503,
                    detail=str(e),
                    headers=_SERVICE_UNAVAILABLE_HEADERS,
                ) from e
            return {"status": "success", "count": len(results), "results": results}
        return await self.search_text(text_query=tq, top_k=top_k)

# ── Singleton ─────────────────────────────────
semantic_search_api = SemanticSearchResource()