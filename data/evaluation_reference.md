# Referência para avaliação do Agentic RAG

## Objetivo
O Agentic RAG é um assistente de documentação técnica que usa Retrieval-Augmented Generation (RAG) para responder perguntas com base em documentos recuperados.

## Tecnologias e arquitetura
O backend usa FastAPI. PostgreSQL com a extensão pgvector armazena chunks de documentos e seus embeddings. O fluxo de arquitetura descrito para o MVP é React -> FastAPI -> serviço RAG -> pgvector -> modelo Gemini para embeddings e geração.

O pipeline ingere documentos, divide o texto em chunks, gera embeddings, armazena os vetores, recupera trechos relevantes e solicita ao modelo de linguagem uma resposta fundamentada no contexto. Embeddings representam semanticamente textos em vetores; a busca vetorial usa esses vetores para encontrar trechos semelhantes à pergunta.

## Documentos e endpoints
O MVP aceita documentos Markdown e TXT. O endpoint POST /api/documents recebe o arquivo no campo multipart chamado `file`. O endpoint POST /api/query recebe um JSON com o campo `question` e retorna uma resposta com as fontes recuperadas. A documentação interativa da API fica em http://localhost:8000/docs durante a execução local.

## Grounding e comportamento esperado
As respostas devem usar evidências presentes nos trechos recuperados e não inventar informações sem suporte. Se não houver evidências suficientes nos documentos disponíveis, a aplicação deve informar que não encontrou evidências suficientes para responder.

## Execução e testes
Para iniciar localmente, copie `.env.example` para `.env` e execute `docker compose up --build` na raiz do repositório. Para executar os testes, entre na pasta `backend` e rode `python -m pytest`.

Uma consulta de exemplo pode ser enviada com `curl` usando POST /api/query, o cabeçalho `Content-Type: application/json` e um corpo como `{"question":"Como funciona a arquitetura do projeto?"}`.

## Roadmap
As melhorias planejadas incluem interface de chat em Vue com TypeScript, busca híbrida com reranking, ferramentas de agente, servidor MCP, dataset e métricas de avaliação RAG, OpenTelemetry e Terraform com Cloud Run.
