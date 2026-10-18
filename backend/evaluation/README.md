# Avaliação de recuperação: corpus diversificado + reranking

O dataset contém 32 perguntas distribuídas entre quatro documentos de referência. Ele foi projetado para avaliar fontes distintas, consultas com termos exatos e perguntas formuladas de maneiras diferentes.

## Corpus

Os quatro arquivos em `data/evaluation_corpus/` devem ser ingeridos na API antes da avaliação:

- `architecture-and-ingestion.md`
- `retrieval-and-ranking.md`
- `evaluation-and-testing.md`
- `operations-and-security.md`

Os nomes dos arquivos ingeridos precisam corresponder exatamente aos valores de `expected_sources` no JSONL. Faça upload de cada documento pela API de ingestão já existente. Não é necessário excluir documentos anteriores; eles podem aparecer como candidatos, mas não contam como fontes esperadas.

## Executar comparação

No terminal, a partir de `backend/`:

```bash
python -m evaluation.compare_retrieval \
  --top-k 5 \
  --output evaluation/reports/retrieval-comparison.json
```

O script compara:

1. `vector_only`: recuperação vetorial atual.
2. `hybrid_rrf`: recuperação vetorial + lexical combinada por Reciprocal Rank Fusion, sem reranking.
3. `hybrid_rrf_crossencoder`: mesma recuperação híbrida, seguida pelo CrossEncoder.

A primeira execução com CrossEncoder pode baixar pesos e levar mais tempo. Para medir apenas os dois primeiros métodos:

```bash
python -m evaluation.compare_retrieval --top-k 5 --skip-crossencoder
```

## Métricas e limites

- **Recall@K:** proporção das fontes esperadas recuperadas entre os K resultados.
- **MRR:** inverso da posição do primeiro resultado relevante, calculado como média entre perguntas.
- **Hit Rate@K:** proporção de perguntas que recuperam pelo menos uma fonte esperada.

O relatório contém métricas agregadas e resultados por pergunta para cada método. Não espere necessariamente uma melhoria: o resultado depende do corpus, dos embeddings, da configuração e do modelo CrossEncoder. Essas métricas medem recuperação de fontes, não fidelidade da resposta gerada. Não publique resultados como benchmark de produção sem uma avaliação maior e representativa.
