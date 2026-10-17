# Operação, segurança e observabilidade

## Execução local
Docker Compose pode iniciar a API e o PostgreSQL com pgvector. Antes de executar avaliações, confirme que o container do banco está saudável, que a URL de conexão aponta para o banco correto e que o corpus de avaliação foi ingerido. O processo de avaliação executado no host normalmente usa localhost e a porta publicada; um container deve usar o nome do serviço Docker na rede interna.

## Variáveis e segredos
O arquivo .env guarda configuração local e não deve ser versionado. O repositório deve incluir um .env.example sem valores secretos. Chaves de API devem ser configuradas pelo ambiente e nunca incluídas em código, relatórios, screenshots ou logs.

## Observabilidade
Logs estruturados podem registrar duração da ingestão, quantidade de chunks, duração da recuperação, número de candidatos e duração da geração. Métricas úteis incluem latência p50/p95, erros por endpoint, tamanho do contexto e custo estimado por consulta. Não registre conteúdo sensível de documentos nem credenciais.

## Desempenho e custo
A busca vetorial e a busca lexical têm custos diferentes. O reranking aumenta o custo computacional porque pontua pares pergunta-documento. Meça latência com o mesmo hardware e conjunto de consultas; compare qualidade e tempo em vez de habilitar o reranker apenas por intuição. Um modelo pode ser carregado na primeira chamada, tornando a primeira execução mais lenta que as seguintes.

## Checklist antes de publicar
Execute os testes unitários e de integração, verifique que os relatórios não contêm segredos, documente como reproduzir a avaliação e informe claramente as limitações. Não publique métricas como resultado de produção se foram medidas apenas em um pequeno corpus local.
