# Agentic RAG GCP

The MVP ingests technical documentation, creates vector embeddings, stores chunks in PostgreSQL with pgvector, retrieves the most relevant chunks, and asks a Gemini model to answer using only retrieved context.

## Architecture

React -> FastAPI -> RAG service -> pgvector -> Gemini embeddings/LLM

The next iteration will add tool calling, MCP, guardrails, OpenTelemetry and evaluation.
