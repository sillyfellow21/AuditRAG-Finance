from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Financial Multimodal Assistant"
    app_env: str = "development"
    log_level: str = "INFO"
    finance_guardrails_enabled: bool = False

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    streamlit_backend_url: str = "http://localhost:8000"

    data_dir: Path = Field(default=Path("./data"))
    upload_dir: Path = Field(default=Path("./data/uploads"))
    document_dir: Path = Field(default=Path("./data/documents"))
    chroma_dir: Path = Field(default=Path("./data/chroma"))

    groq_api_key: Optional[SecretStr] = None
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    extraction_use_llm: bool = False
    max_upload_size_mb: int = 20
    docling_enabled: bool = True
    docling_for_pdf: bool = False

    tesseract_cmd: Optional[str] = None

    langchain_tracing_v2: bool = True
    langchain_api_key: Optional[SecretStr] = None
    langchain_project: str = "financial-doc-assistant"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    rag_top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 120
    max_chunks_per_document: int = 250

    response_cache_enabled: bool = True
    response_cache_ttl_seconds: int = 300
    response_cache_max_entries: int = 512

    retrieval_cache_enabled: bool = True
    retrieval_cache_ttl_seconds: int = 180
    retrieval_cache_max_entries: int = 1024

    repository_cache_ttl_seconds: int = 120
    repository_cache_max_entries: int = 1024

    anomaly_high_total_threshold: float = 5000.0
    anomaly_high_tax_ratio: float = 0.35
    anomaly_line_total_mismatch_ratio: float = 0.2
    anomaly_robust_z_threshold: float = 3.5
    anomaly_history_limit: int = 80

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def has_groq(self) -> bool:
        return self.groq_api_key is not None and bool(self.groq_api_key.get_secret_value())

    @property
    def has_langsmith(self) -> bool:
        return self.langchain_api_key is not None and bool(self.langchain_api_key.get_secret_value())

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.document_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
