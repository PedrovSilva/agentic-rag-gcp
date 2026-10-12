from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from app.core.config import get_settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        yield conn


def init_db() -> None:
    settings = get_settings()
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                content_sha256 CHAR(64) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (content_sha256)
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector({settings.embedding_dimensions}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (document_id, chunk_index)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx "
            "ON document_chunks (document_id)"
        )
