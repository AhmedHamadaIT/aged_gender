"""Placeholder for Qdrant embedding verification (extend when models are pinned)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Requires tuned ReID crops + Qdrant collections; run manually.")
def test_embedding_upsert_search():
    assert False
