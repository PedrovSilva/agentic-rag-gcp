# Agentic RAG — Backend MVP

FastAPI backend for document ingestion and retrieval-augmented generation (RAG).
Uses **Groq** for answer generation, **Hugging Face Inference Providers** for embeddings, and **PostgreSQL + pgvector** for similarity search.

## Requirements

- Python 3.12+ (Docker image uses Python 3.12)
- PostgreSQL with pgvector, or the included Compose service
- Groq API key and Hugging Face token with inference permissions

## Run locally

From the backend directory:

```bash
cp .env.example .env
# Fill GROQ_API_KEY and HF_TOKEN in .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Set `DATABASE_URL` to your PostgreSQL instance. The API creates the required tables and the `vector` extension on startup. Your database user needs permission to create the extension (or have an administrator enable it once).

## Run tests

```bash
pytest -q
```

## Endpoints

- `GET /health` — process health check
- `POST /api/documents` — multipart upload of `.md`, `.markdown` or `.txt`
- `POST /api/query` — JSON question, returns answer and source chunks
- `GET /docs` — interactive Swagger UI

Example upload:

```bash
curl -X POST http://localhost:8000/api/documents \
  -F 'file=@../data/architecture.md'
```

Example query:

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Como funciona a arquitetura do projeto?","top_k":5}'
```

## Important notes

- `BAAI/bge-small-en-v1.5` produces 384-dimensional embeddings. `EMBEDDING_DIMENSIONS` must match the PostgreSQL vector column. If you change embedding models/dimensions, recreate or migrate the `document_chunks` table and re-ingest every document.
- Do not commit `.env` or API tokens.
- Uploaded content is treated as untrusted context by the generation prompt. This is a baseline guardrail, not a complete prompt-injection defense.
- Duplicate normalized document content is detected by SHA-256 and is not inserted twice.
