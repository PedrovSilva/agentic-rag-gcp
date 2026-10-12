from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.db import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    description="RAG API using Groq for generation, Hugging Face for embeddings, and pgvector for retrieval.",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}
