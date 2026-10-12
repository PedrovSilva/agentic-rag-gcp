from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.service import answer_question, ingest_document

import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceResponse(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    content: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@router.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    settings = get_settings()
    filename = file.filename or "uploaded-document.txt"
    if not filename.lower().endswith((".md", ".markdown", ".txt")):
        raise HTTPException(status_code=415, detail="Only .md, .markdown and .txt files are supported")

    content_bytes = await file.read(settings.max_upload_bytes + 1)
    if len(content_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB upload limit")
    try:
        content = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded") from exc
    if not content.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        return ingest_document(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=502,
            detail=f"Document ingestion failed: {type(exc).__name__}",
        ) from exc

@router.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest):
    try:
        return answer_question(payload.question, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=502,
            detail=f"RAG query failed: {type(exc).__name__}",
        ) from exc