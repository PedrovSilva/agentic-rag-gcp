from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic RAG API"
    app_env: str = "development"
    database_url: str = "postgresql://rag:rag@localhost:5432/ragdb"

    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    hf_token: str
    hf_embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    retrieval_top_k: int = 5
    max_upload_bytes: int = 10 * 1024 * 1024
    chunk_size: int = 900
    chunk_overlap: int = 150

    # RAG guardrails
    retrieval_max_distance: float = 0.45
    max_context_chars: int = 12000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
