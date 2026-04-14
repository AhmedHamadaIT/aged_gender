"""
apis/semantic_search.py
-----------------
semantic_search resource — manages text-to-image semantic search via Qdrant.

Routes registered in app.py:
    POST /semantic_search/search_text   → submit text query to find top similar images
"""

from fastapi import HTTPException, Form
from typing import Dict, Any

from apis.base import BaseResource
from services.semantic_search import SemanticSearchService
from logger.logger_config import Logger

log = Logger.get_logger(__name__)

# ─────────────────────────────────────────────
# semantic_search resource
# ─────────────────────────────────────────────
class SemanticSearchResource(BaseResource):
    def __init__(self):
        super().__init__()
        log.info("[SEMANTIC_SEARCH API] Initializing Semantic Search Service...")
        self.semantic_search_service = SemanticSearchService()

    async def search_text(self, text_query: str = Form(...), top_k: int = Form(10)) -> Dict[str, Any]:
        """
        Takes a text query and searches the vector DB for matching images.
        """
        if not text_query or not text_query.strip():
            raise HTTPException(status_code=400, detail="Text query cannot be empty.")

        try:
            results = self.semantic_search_service.search_by_text(text_query=text_query, top_k=top_k)
            
            return {
                "status": "success",
                "count": len(results),
                "results": results
            }
            
        except Exception as e:
            log.error(f"[SEMANTIC_SEARCH API] Text search failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ── Singleton ─────────────────────────────────
semantic_search_api = SemanticSearchResource()