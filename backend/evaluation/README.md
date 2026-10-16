# RAG retrieval evaluation

Este diretório contém um dataset versionado com 20 perguntas e um script para medir a qualidade da recuperação. Cada registro JSONL inclui uma resposta de referência para futura avaliação das respostas e nomes de arquivos esperados para avaliar a recuperação.

## Pré-requisitos

1. Suba a aplicação e o PostgreSQL/pgvector com o fluxo normal do Docker Compose.
2. Faça upload de `data/evaluation_reference.md` para `POST /api/documents`. A avaliação não ingere documentos automaticamente.
3. Configure as variáveis de ambiente do backend no `.env`.

## Executar

Na pasta `backend/`:

```bash
python -m evaluation.evaluate_retrieval --top-k 5
python -m evaluation.evaluate_retrieval --top-k 5 --output evaluation/reports/latest.json
```

## Métricas

- **Recall@K**: fração média dos documentos-fonte esperados encontrados entre os K primeiros resultados.
- **MRR (Mean Reciprocal Rank)**: média do inverso da posição do primeiro documento esperado; zero quando nenhum é recuperado.
- **Hit Rate@K**: proporção de perguntas em que ao menos uma fonte esperada aparece nos K primeiros resultados.

## Evolução do dataset

- Mantenha um objeto JSON por linha em `dataset.jsonl`.
- `expected_sources` deve corresponder aos valores de `filename` armazenados no banco.
- Inclua perguntas fáceis e difíceis, documentos diferentes e casos sem resposta conforme o corpus crescer.
- `expected_answer` é uma referência escrita por uma pessoa, não uma resposta automaticamente verificada. Revise e expanda o dataset antes de usar métricas como afirmação de qualidade.
- Compare configurações com o mesmo dataset e corpus. Não divulgue resultados como medidos até executar o script contra um corpus indexado.
