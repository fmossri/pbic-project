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
├── src/
│   ├── data_ingestion/
│   │   ├── document_processor.py
│   │   ├── text_chunker.py
│   │   └── data_ingestion_orchestrator.py
│   ├── query_processing/
│   │   ├── query_orchestrator.py
│   │   └── hugging_face_manager.py
│   ├── utils/
│   │   ├── embedding_generator.py
│   │   ├── text_normalizer.py
│   │   ├── sqlite_manager.py
│   │   └── faiss_manager.py
│   └── models/
│       ├── document_file.py
│       ├── chunk.py
│       └── embedding.py
├── storage/
│   └── domains/
│       ├── public/
│       │   ├── public.db
│       │   └── vector_store/
│       │       └── index.faiss
│       └── test_domain/
│           ├── test.db
│           └── vector_store/
│               └── test.faiss
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
│   └── utils/
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
- Interfaces:
  * `process_directory`: Executa a pipeline de ingestão
  * `list_pdf_files`: Lista os pdfs de um diretório

#### 2. DocumentProcessor
- Processa arquivos PDF e extrai texto
- Calcula hash MD5 para detecção de duplicatas
- Trata erros de PDF inválido com propagação adequada
- Interfaces:
  * `process_document`: Extrai o texto e calcula o hash de um documento PDF

#### 3. TextChunker
- Divide texto em chunks recursivos mantendo coerência
- Utiliza RecursiveCharacterTextSplitter para divisão por marcadores de quebra natural
- Preserva estrutura do documento original nos chunks
- Interfaces:
  * `chunk_text`: Divisão principal do texto

### Pipeline de Consulta

#### 1. QueryOrchestrator
- Coordena todo o processo de consulta
- Converte consultas em embeddings
- Recupera chunks relevantes via busca vetorial
- Prepara contexto para modelos de linguagem
- Interfaces:
  * `query_llm`: Executa a pipeline de recuperação e consulta

#### 2. HuggingFaceManager
- Interface com a API de Inferência do Hugging Face
- Suporta diferentes modelos de linguagem (no momento zephyr-7b-beta)
- Parâmetros de geração configuráveis (temperatura, top_p, etc.)
- Interfaces:
  * `generate_answer`: Consulta o LLM e retorna a resposta

### Componentes Compartilhados

#### 1. TextNormalizer
- Realiza normalização Unicode (NFKC)
- Normaliza espaços preservando estrutura
- Normaliza case para minúsculas
- Interfaces:
  * `normalize`: Pipeline completo de normalização

#### 2. EmbeddingGenerator
- Gera representações vetoriais para texto
- Realiza processamento em lote para melhor performance
- Integração com sentence-transformers
- Interfaces:
  * `generate_embeddings`: Gera embeddings a partir do texto fornecido

#### 3. SQLiteManager
- Gerencia todas as operações de banco de dados
- Inicialização de schema e controle de versão
- Controle de transações (commit/rollback)
- Armazenamento de documentos, chunks e metadados de embeddings
- Armazena os arquivos .db em `storage/domains/{domain_name}/`
- Interfaces:
  * `insert_document_file`: Insere os metadados dos documentos no banco de dados
  * `insert_chunks`: Insere os chunks de texto no banco de dados
  * `insert_embeddings`: Insere os metadados dos embeddings
  * `get_embeddings_chunks`: Recupera os chunks pelos índices FAISS de seus embeddings

#### 4. FaissManager
- Gerencia índices FAISS para busca por similaridade
- Cria, carrega e consulta índices
- Adiciona os embeddings ao índice FAISS
- Armazena os arquivos .faiss em `storage/domains/{domain_name}/vector_store/`
- Realiza busca por similaridade
- Interfaces:
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
git clone https://github.com/fmossri/pbic-project.git
cd pbic-project
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

4. Crie um token de acesso com permissões de leitura na Hugging Face;

5. Crie um arquivo .env e adicione a seguinte variável de ambiente:
```
HUGGINGFACE_API_TOKEN="seu token de acesso"
```

## Uso

** main.py --help

** Opcional: adicionar --debug ao final traz mais informações ao console e aos arquivos de log.

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
- Logger

### Em Desenvolvimento 🔄

- Lógica de criação de domínios
- Separação de documentos ingeridos por domínio (.db e .faiss independentes)
- Sistema de seleção de domínios para geração de respostas usando busca por similaridade
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

## Logging System

The system includes a robust logging system with the following features:

- JSON-formatted logs for machine readability
- Domain-based logging for different components
- Context tracking and correlation
- Error handling with stack traces
- Log file rotation and management
- Library log suppression
- Different formats for console and file output

### Log Levels

The system uses standard Python log levels:
- CRITICAL (50): Critical errors that may cause system failure
- ERROR (40): Errors that need attention but don't stop the system
- WARNING (30): Warning messages for potential issues
- INFO (20): General information about system operation
- DEBUG (10): Detailed information for debugging

### Log Format

Console output format:
```
2025-04-13T20:17:02.282109 - src.utils.logger - INFO - Sistema de registro de logs configurado
```

File output format (JSON):
```json
{
    "timestamp": "2025-04-13T20:17:02.282109",
    "level": "INFO",
    "name": "src.utils.logger",
    "message": "Sistema de registro de logs configurado",
    "function": "setup_logging",
    "context": {}
}
```

