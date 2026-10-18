# Recuperação de contexto e ranking

## Busca vetorial
Na busca vetorial, a pergunta é transformada em embedding e comparada aos embeddings dos chunks armazenados no pgvector. A distância cosseno é usada para ordenar os candidatos; valores menores representam maior proximidade vetorial. O parâmetro RETRIEVAL_TOP_K controla quantos resultados finais são devolvidos.

## Busca lexical
A busca lexical usa recursos de texto completo do PostgreSQL. to_tsvector transforma o conteúdo em uma representação pesquisável, e websearch_to_tsquery interpreta a consulta do usuário. Essa abordagem pode encontrar nomes de configuração, siglas e termos exatos que não ficam necessariamente próximos no espaço vetorial.

## Busca híbrida e RRF
A recuperação híbrida combina candidatos da busca vetorial e da busca lexical. Reciprocal Rank Fusion (RRF) combina posições, somando para cada candidato contribuições da forma 1 / (c + rank), em que c é uma constante que reduz a diferença entre posições. RRF não exige que as pontuações vetoriais e lexicais estejam na mesma escala.

## CrossEncoder
Um CrossEncoder recebe pares formados pela pergunta e pelo conteúdo de cada candidato e estima a relevância do par. Ele reordena os candidatos produzidos pela busca inicial; não substitui a etapa de recuperação. Como exige inferência adicional e pode baixar pesos na primeira execução, deve ser medido quanto a qualidade melhora e quanto aumenta a latência.

## Parâmetros
HYBRID_CANDIDATE_MULTIPLIER amplia o conjunto de candidatos antes do corte final. RRF_CONSTANT define a constante da fusão. RERANKER_ENABLED habilita ou desabilita o CrossEncoder. RERANKER_MODEL seleciona o modelo de reranking.
