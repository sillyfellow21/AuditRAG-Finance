from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Settings
from monitoring.observability import Observability


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        document_dir=tmp_path / "documents",
        chroma_dir=tmp_path / "chroma",
        response_cache_enabled=True,
        retrieval_cache_enabled=True,
    )
    settings.ensure_directories()
    return settings


@pytest.fixture
def observability(test_settings: Settings) -> Observability:
    return Observability(settings=test_settings)
