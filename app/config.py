from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):

    # =========================
    # OpenAI
    # =========================

    openai_api_key: str

    chat_model: str = "gpt-5-mini"
    router_model: str = "gpt-5-mini"
    web_search_model: str = "gpt-5-mini"

    embedding_model: str = "text-embedding-3-small"

    # =========================
    # Pinecone
    # =========================

    pinecone_api_key: str

    pinecone_index: str = "rag-chatbot"
    pinecone_namespace: str = "documents"

    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    pinecone_dimension: int = 1536

    # =========================
    # Redis
    # =========================

    redis_url: str

    # =========================
    # RAG
    # =========================

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5

    # =========================
    # Uploads
    # =========================

    max_upload_mb: int = 20

    # =========================
    # Chat
    # =========================

    chat_context_messages: int = 8

    # =========================
    # Server
    # =========================

    host: str = "127.0.0.1"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()