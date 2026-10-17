"""Hybrid retrieval: pgvector + PostgreSQL full-text search + RRF/cross-encoder."""
from functools import lru_cache
from typing import Any

from pgvector import Vector
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.db import get_connection
from app.services.ai import embed_text


@lru_cache(maxsize=1)
def _get_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def retrieve_hybrid_context(
    question: str, top_k: int | None = None, *, rerank: bool | None = None
) -> list[dict[str, Any]]:
    settings = get_settings()
    limit = top_k if top_k is not None else settings.retrieval_top_k
    if limit < 1:
        raise ValueError("top_k must be at least 1")
    candidate_limit = max(limit, limit * settings.hybrid_candidate_multiplier)
    query_vector = Vector(embed_text(question))

    with get_connection() as conn:
        vector_rows = conn.execute(
            """SELECT d.id, d.filename, c.chunk_index, c.content,
                      (c.embedding <=> %s) AS distance
               FROM document_chunks c JOIN documents d ON d.id = c.document_id
               ORDER BY c.embedding <=> %s LIMIT %s""",
            (query_vector, query_vector, candidate_limit),
        ).fetchall()
        lexical_rows = conn.execute(
            """SELECT d.id, d.filename, c.chunk_index, c.content,
                      ts_rank_cd(to_tsvector('simple', c.content),
                                 websearch_to_tsquery('simple', %s)) AS lexical_score
               FROM document_chunks c JOIN documents d ON d.id = c.document_id
               WHERE to_tsvector('simple', c.content) @@ websearch_to_tsquery('simple', %s)
               ORDER BY lexical_score DESC, c.id ASC LIMIT %s""",
            (question, question, candidate_limit),
        ).fetchall()

    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    fused_scores: dict[tuple[int, int], float] = {}
    constant = settings.rrf_constant
    for rank, row in enumerate(vector_rows, 1):
        key = (int(row[0]), int(row[2]))
        candidates.setdefault(key, {
            "document_id": row[0], "filename": row[1], "chunk_index": row[2],
            "content": row[3], "distance": float(row[4]), "lexical_score": None,
        })
        candidates[key]["distance"] = float(row[4])
        fused_scores[key] = fused_scores.get(key, 0.0) + 1 / (constant + rank)

    for rank, row in enumerate(lexical_rows, 1):
        key = (int(row[0]), int(row[2]))
        candidates.setdefault(key, {
            "document_id": row[0], "filename": row[1], "chunk_index": row[2],
            "content": row[3], "distance": None, "lexical_score": float(row[4]),
        })
        candidates[key]["lexical_score"] = float(row[4])
        fused_scores[key] = fused_scores.get(key, 0.0) + 1 / (constant + rank)

    ordered = sorted(candidates.items(), key=lambda item: fused_scores[item[0]], reverse=True)
    results = [candidate for _, candidate in ordered[:candidate_limit]]
    should_rerank = settings.reranker_enabled if rerank is None else rerank
    if should_rerank and results:
        scores = _get_reranker(settings.reranker_model).predict(
            [(question, item["content"]) for item in results]
        )
        for item, score in zip(results, scores):
            item["rerank_score"] = float(score)
        results.sort(key=lambda item: item["rerank_score"], reverse=True)
    return results[:limit]
