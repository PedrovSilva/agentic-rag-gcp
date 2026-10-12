import hashlib
from typing import Any

from pgvector import Vector

from app.core.config import get_settings
from app.db import get_connection
from app.rag.chunking import split_text
from app.services.ai import embed_text, generate_answer


def ingest_document(filename: str, content: str) -> dict[str, Any]:
    settings = get_settings()
    normalized = content.strip()
    if not normalized:
        raise ValueError("The document is empty")

    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    chunks = split_text(normalized, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("No text could be extracted from the document")

    # Compute embeddings before opening the transaction, avoiding a long DB lock
    # while making remote API calls.
    embedded_chunks = [(chunk, embed_text(chunk)) for chunk in chunks]

    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id, filename FROM documents WHERE content_sha256 = %s",
            (digest,),
        ).fetchone()
        if existing:
            return {"document_id": existing[0], "filename": existing[1], "chunks": 0, "duplicate": True}

        row = conn.execute(
            "INSERT INTO documents (filename, content_sha256) VALUES (%s, %s) RETURNING id",
            (filename, digest),
        ).fetchone()
        document_id = row[0]
        conn.executemany(
            "INSERT INTO document_chunks (document_id, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s)",
            [
                (document_id, index, chunk, Vector(vector))
                for index, (chunk, vector) in enumerate(embedded_chunks)
            ],
        )
    return {"document_id": document_id, "filename": filename, "chunks": len(chunks), "duplicate": False}


def retrieve_context(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    query_vector = Vector(embed_text(question))
    limit = top_k or settings.retrieval_top_k
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT d.id AS document_id, d.filename, c.chunk_index, c.content,
                   (c.embedding <=> %s) AS distance
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (query_vector, query_vector, limit),
        ).fetchall()
    return [
        {
            "document_id": row[0],
            "filename": row[1],
            "chunk_index": row[2],
            "content": row[3],
            "distance": float(row[4]),
        }
        for row in rows
    ]


def answer_question(question: str, top_k: int | None = None) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    sources = retrieve_context(question, top_k)
    if not sources:
        return {
            "answer": "Ainda não encontrei documentos indexados para responder à pergunta.",
            "sources": [],
        }
    context = "\n\n---\n\n".join(
        f"Fonte: {source['filename']} (trecho {source['chunk_index'] + 1})\n{source['content']}"
        for source in sources
    )
    answer = generate_answer(question, context)
    return {
        "answer": answer,
        "sources": [
            {
                "document_id": source["document_id"],
                "filename": source["filename"],
                "chunk_index": source["chunk_index"],
                "content": source["content"],
                "distance": source["distance"],
            }
            for source in sources
        ],
    }
