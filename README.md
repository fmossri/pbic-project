# Pipeline RAG de Ingestão de PDFs

Sistema para processamento de documentos PDF, geração de embeddings, busca semântica e obtenção de respostas contextuais através de modelos de linguagem.

## Visão Geral

Este sistema processa documentos PDF, extrai texto, divide em chunks, gera embeddings e responde perguntas utilizando busca por similaridade e recuperação de documentos. O sistema é projetado para ser escalável, confiável e fácil de usar, constituindo uma solução sequencial simples "naive RAG"

### Funcionalidades Principais

- Processamento de documentos PDF
- Detecção de duplicatas via hash MD5
- Divisão de texto em chunks
- Geração de embeddings
- Normalização de texto
- Armazenamento de embeddings com FAISS
- Armazenamento de chunks e metadados com SQLite
- Busca por similaridade e recuperação de chunks relevantes
- Integração com modelos de linguagem via API da Hugging Face

## Estrutura do Projeto

```
/root
├── components/
│   ├── data_ingestion/
│   │   ├── document_processor.py
│   │   ├── text_chunker.py
│   │   └── data_ingestion_orchestrator.py
│   ├── query_processing/
│   │   ├── query_orchestrator.py
│   │   └── hugging_face_manager.py
│   ├── shared/
│   │   ├── embedding_generator.py
│   │   ├── text_normalizer.py
│   │   ├── sqlite_manager.py
│   │   └── faiss_manager.py
│   └── models/
│       ├── document_file.py
│       ├── chunk.py
│       └── embedding.py
├── databases/
│   ├── public/**Diretório padrão dos arquivos .db
│   └── schemas/
│       └── schema.sql
├── indices/
│   └──public/**Diretório padrão dos arquivos .faiss
├── tests/
│   ├── data_ingestion/
│   │   ├── test_document_processor.py
│   │   ├── test_data_ingestion_orchestrator.py
│   │   ├── test_text_chunker.py
│   │   └── test_docs/
│   │       └── generate_test_pdfs.py
│   ├── query_processing/
│   │   ├── test_query_orchestrator.py
│   │   └── test_hugging_face_manager.py
│   └── shared/
│       ├── test_embedding_generator.py
│       ├── test_text_normalizer.py
│       ├── test_sqlite_manager.py
│       └── test_faiss_manager.py
├── main.py
└── README.md
```

## Componentes Principais

### Pipeline de Ingestão de Dados

#### 1. DataIngestionOrchestrator
- Coordena todo o fluxo de processamento de ingestão
- Efetua controle de duplicatas via hash MD5
- Controla transações para garantir a integridade dos dados
- Processa diretórios inteiros de PDFs
- Principais métodos:
  * `process_directory`: Executa a pipeline de ingestão
  * `list_pdf_files`: Lista os pdfs de um diretório

#### 2. DocumentProcessor
- Processa arquivos PDF e extrai texto
- Calcula hash MD5 para detecção de duplicatas
- Trata erros de PDF inválido com propagação adequada
- Principais métodos:
  * `extract_text`: Extração de texto do PDF
  * `calculate_hash`: Cálculo de hash MD5

#### 3. TextChunker
- Divide texto em chunks recursivos mantendo coerência
- Utiliza RecursiveCharacterTextSplitter para divisão por marcadores de quebra natural.
- Preserva estrutura do documento original nos chunks
- Principais métodos:
  * `chunk_text`: Divisão principal do texto

### Pipeline de Consulta

#### 1. QueryOrchestrator
- Coordena todo o processo de consulta
- Converte consultas em embeddings
- Recupera chunks relevantes via busca vetorial
- Prepara contexto para modelos de linguagem
- Principais métodos:
  * `query_llm`: Executa a pipeline de recuperação e consulta

#### 2. HuggingFaceManager
- Interface com a API de Inferência do Hugging Face
- Suporta diferentes modelos de linguagem (no momento zephyr-7b-beta)
- Parâmetros de geração configuráveis (temperatura, top_p, etc.)
- Principais métodos:
  * `generate_answer`: Consulta o LLM e retorna a resposta

### Componentes Compartilhados

#### 1. TextNormalizer
- Realiza normalização Unicode (NFKC)
- Normaliza espaços preservando estrutura
- Normaliza case para minúsculas
- Principais métodos:
  * `normalize`: Pipeline completo de normalização

#### 2. EmbeddingGenerator
- Gera representações vetoriais para texto
- Realiza processamento em lote para melhor performance
- Integração com sentence-transformers
- Principais métodos:
  * `generate_embeddings`: Gera embeddings a partir do texto fornecido

#### 3. SQLiteManager
- Gerencia todas as operações de banco de dados
- Inicialização de schema e controle de versão
- Controle de transações (commit/rollback)
- Armazenamento de documentos, chunks e metadados de embeddings
- Principais métodos:
  * `insert_document_file`: Insere os metadados dos documentos no banco de dados
  * `insert_chunks`: Insere os chunks de texto no banco de dados
  * `insert_embeddings`: Insere os metadados dos embeddings
  * `get_embeddings_chunks`: Recupera os chunks pelos índices FAISS de seus embeddings

#### 4. FaissManager
- Gerencia índices FAISS para busca por similaridade
- Cria, carrega e consulta índices
- Adiciona os embeddings ao índice FAISS.
- Realiza busca por similaridade
- Principais métodos:
  * `add_embeddings`: Adiciona embeddings ao índice
  * `search_faiss_index`: Realiza busca por similaridade

### Modelos de Dados

#### 1. DocumentFile
- Representa um documento PDF com metadados
- Rastreia caminho, hash e contagem de páginas

#### 2. Chunk
- Representa um chunk de texto com informações de posição
- Referência para o documento pai

#### 3. Embedding
- Representa um embedding vetorial
- Metadados para rastreamento no índice FAISS
- Referência para o chunk pai

## Requisitos

- Python 3.10+
- Dependências principais:
  * pypdf 5.4.0
  * sentence-transformers 3.4.1
  * faiss-cpu 1.10.0
  * huggingface-hub 0.29.3
  * langchain 0.3.21
  * langchain-text-splitters 0.3.7
  * SQLAlchemy 2.0.39
  * pytest 8.3.5
  * python-dotenv 1.0.1

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/rag-project.git
cd rag-project
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Processando PDFs e Gerando Embeddings

```bash
python main.py -i caminho/do/diretório
```

### Realizando Consultas

```bash
python main.py -q "Sua pergunta aqui"
```

### Executando Testes

```bash
python -m pytest -vv
```

## Estado Atual do Desenvolvimento

### Componentes Concluídos ✅

- Pipeline completo de ingestão de PDF
- Processamento de texto e chunking semântico
- Geração de embeddings e normalização de texto
- Armazenamento SQLite com gerenciamento de transações
- Armazenamento FAISS para vetores
- Sistema de consulta e recuperação
- Integração com Hugging Face para geração de respostas
- Testes unitários e de integração

### Em Desenvolvimento 🔄

- Sistema de logging estruturado
- Sistema de configuração
- Interface de usuário
- **API RESTful
- **Funcionalidades avançadas de busca

## Próximos Passos Possíveis

1. **Aprimoramento do Sistema de Consulta**
   - Otimização de prompts
   - Expansão de consultas usando sinônimos
   - Re-ranqueamento dos chunks recuperados
   - Atribuição de fontes para fundamentar as respostas

2. **Desenvolvimento de API**
   - API RESTful com FastAPI
   - Processamento assíncrono
   - Validação de entrada com pydantic
   - Documentação com OpenAPI/Swagger

3. **Funcionalidades Avançadas**
   - Busca híbrida com grafos de conhecimento
   - Processamento multi-modal (imagens, tabelas)

4. **Monitoramento e Logging**
   - Logging estruturado
   - Métricas de performance
   - Verificações de saúde da aplicação

5. **Interface Web**
  - Criação de página da web para interagir com o sistema
  - Funcionalidade de ingestão, com inserção de domínio, palavras-chave e diretório alvo
  - Sistema de configuração personalizada
  - Funcionalidade de consulta

