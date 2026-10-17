# Arquitetura e ingestão de documentos

## Componentes
O backend da aplicação é desenvolvido com FastAPI e Python. O PostgreSQL armazena metadados dos documentos e os trechos de texto. A extensão pgvector guarda embeddings e permite busca por similaridade vetorial. O modelo de linguagem é acessado por meio da API da Groq; embeddings são produzidos localmente pelo Sentence Transformers.

## Fluxo de ingestão
O cliente envia um arquivo para o endpoint de documentos. A API valida o tipo e o tamanho do arquivo antes de processá-lo. O conteúdo extraído é dividido em chunks com sobreposição para preservar contexto entre trechos vizinhos. Cada chunk recebe um embedding e é persistido junto ao identificador do documento, nome do arquivo e índice do chunk.

## Configuração e limites
A variável DATABASE_URL define a conexão PostgreSQL. GROQ_API_KEY autentica chamadas ao serviço de geração. HF_TOKEN pode ser necessário conforme a configuração do provedor/modelo de embeddings. MAX_UPLOAD_BYTES limita o tamanho aceito no upload. CHUNK_SIZE controla o tamanho aproximado dos trechos e CHUNK_OVERLAP determina quantos caracteres são compartilhados entre trechos consecutivos.

## Falhas de ingestão
Falhas de extração, geração de embeddings ou gravação devem produzir respostas de erro controladas. O sistema não deve registrar chaves secretas nos logs. Para diagnosticar uma falha, verifique a disponibilidade do banco, as variáveis de ambiente e os logs do serviço sem expor credenciais.
