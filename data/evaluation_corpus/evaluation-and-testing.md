# Avaliação e testes do RAG

## Dataset
O dataset de avaliação é um arquivo JSONL, com um objeto por linha. Cada caso inclui id, question, expected_answer e expected_sources. As fontes esperadas devem corresponder aos nomes dos arquivos que foram realmente ingeridos no banco. Perguntas devem cobrir diferentes documentos, consultas com termos exatos, paráfrases e perguntas que exigem localizar conceitos semelhantes.

## Métricas de recuperação
Recall@K mede a proporção das fontes relevantes esperadas encontradas entre os K primeiros resultados. Mean Reciprocal Rank (MRR) calcula a média do inverso da posição do primeiro resultado relevante. Hit Rate@K mede a proporção de perguntas que recuperam ao menos uma fonte esperada no Top K. As métricas dependem da qualidade e diversidade dos rótulos de referência.

## Comparação justa
Compare busca vetorial, busca híbrida com RRF e busca híbrida com RRF mais CrossEncoder usando exatamente as mesmas perguntas, documentos e valor de K. Registre métricas agregadas e resultados por pergunta. Não conclua que um método é melhor se todos os casos apontam para um único documento ou se o dataset não representa consultas reais.

## Testes automatizados
Testes unitários verificam validação, fusão de rankings e reordenação de candidatos sem depender do banco. Testes de integração validam a recuperação real usando PostgreSQL e pgvector. Testes destrutivos devem usar somente um banco de testes dedicado. Resultados da avaliação devem ser salvos em JSON para permitir comparação entre commits.

## Limitações
Recall e MRR avaliam recuperação de fontes, não a correção da resposta gerada. A avaliação de geração exige critérios adicionais, como fidelidade ao contexto, cobertura da resposta e taxa de afirmações sem suporte.
