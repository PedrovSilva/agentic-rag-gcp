# Agentic RAG GCP — MVP

Technical documentation assistant built with **FastAPI, PostgreSQL + pgvector and Google Gemini/Vertex AI**.

## MVP flow

1. Upload Markdown/TXT documentation.
2. Split it into chunks.
3. Generate embeddings.
4. Store vectors in PostgreSQL/pgvector.
5. Retrieve the most relevant chunks for a question.
6. Generate an answer grounded in the retrieved context.

## Run

```bash
cp .env.example .env

docker compose up --build
```

API: http://localhost:8000/docs

Upload:

```bash
curl -X POST http://localhost:8000/api/documents -F "file=@data/architecture.md"
```

Query:

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Como funciona a arquitetura do projeto?"}'
```

Tests:

```bash
cd backend
python -m pytest
```

## Roadmap

- [ ] Vue chat UI
- [ ] Hybrid search + reranking
- [ ] Agent tools
- [ ] MCP server
- [ ] Guardrails
- [ ] RAG evaluation dataset and metrics
- [ ] OpenTelemetry
- [ ] GitHub Actions
- [ ] Terraform + Cloud Run
