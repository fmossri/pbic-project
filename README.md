# Pipeline de Ingestão de PDFs

Sistema para processamento de documentos PDF, geração de embeddings e busca semântica.

## Visão Geral

Esse sistema processa documentos PDF, extrai texto, divide em chunks semânticos e gera embeddings para busca semântica. O sistema é projetado para ser escalável, confiável e fácil de usar.

### Funcionalidades Principais

- Processamento de documentos PDF
- Detecção de duplicatas via hash MD5
- Divisão de texto em chunks semânticos
- Geração de embeddings para busca semântica
- Normalização de texto consistente
- Sistema de logging estruturado
- Armazenamento vetorial com FAISS
- Armazenamento de metadados com SQLite

## Estrutura do Projeto

```
/root
├── components/
│   ├── data_ingestion/
│   │   ├── document_processor.py
│   │   ├── text_chunker.py
│   │   ├── text_normalizer.py
│   │   └── data_ingestion_component.py
│   └── embedding_generator/
│       ├── embedding_generator.py
│       └── __init__.py
├── tests/
│   ├── data_ingestion/
│   │   ├── test_document_processor.py
│   │   ├── test_data_ingestion_component.py
│   │   ├── test_text_chunker.py
│   │   ├── test_text_normalizer.py
│   │   └── test_docs/
│   │       └── generate_test_pdfs.py
│   └── embedding_generator/
│       └── test_embedding_generator.py
├── main.py
└── README.md
```

## Componentes Principais

### 1. DataIngestionOrchestrator
- Coordena o fluxo de processamento
- Gerencia deduplicação de documentos
- Rastreia tamanhos e hashes
- Coordena interação entre componentes
- Principais métodos:
  * process_directory: Pipeline principal
  * list_pdf_files: Detecção de PDFs
  * _check_duplicate: Detecção de duplicatas
  * _find_original_document: Mapeamento de duplicatas

### 2. DocumentProcessor
- Processa arquivos PDF
- Extrai texto e calcula hash MD5
- Trata erros de PDF inválido
- Gerencia metadados do documento
- Principais métodos:
  * extract_text: Extração de texto do PDF
  * calculate_hash: Cálculo de hash MD5

### 3. TextChunker
- Divide texto em chunks semânticos
- Mantém coerência nas divisões
- Otimizado para ~790 caracteres por chunk
- Usa RecursiveCharacterTextSplitter
- Principais métodos:
  * chunk_text: Divisão principal do texto

!! Usando "RecursiveCharacterTextSplitter" - gera chunks e overlaps
!! usando separadores. Quebra o texto baseado nesses separadores
!! de forma hierárquica - mira um tamanho definido (1000 char, 200 char), mas
!! tenta quebrar em um separador principal (\n\n - parágrafos); se o tamanho do chunk
!! estiver muito distante do alvo, se chama recursivamente, tentando quebrar no próximo
!! separador (\n - linha), até o tamanho final do chunk estiver próximo o suficiente
!! do alvo. Dessa forma, os chunks não tem exatamente o mesmo tamanho. Considerar estratégias
!! diferentes, como semantic chunking

### 4. EmbeddingGenerator
- Gera representações vetoriais
- Processamento em lote
- Usa sentence-transformers
- Otimizado para performance
- Principais métodos:
  * generate_embeddings: Geração de embeddings

!! no momento, estamos usando o modelo "all-MiniLM-L6-v2", hardcoded.
!! Considerar a criação de uma lógica de configuração que permita 
!! selecionar o modelo a ser usado pelo sistema.

### 5. TextNormalizer
- Normalização Unicode (NFKC)
- Normalização de espaços
- Conversão para minúsculas
- Preserva estrutura do texto
- Principais métodos:
  * normalize: Pipeline principal de normalização
  * _normalize_unicode: Normalização Unicode
  * _normalize_whitespace: Normalização de espaços
  * _normalize_case: Normalização de case

### 6. VectorStore
- Busca ou cria um diretório para armazenar os índices
- Carrega o índice existente, ou cria um novo se não existir
- Adiciona os embeddings dos chunks ao índice

## Requisitos

- Python 3.10.12
- pypdf
- sentence-transformers
- pytest 8.3.5
- (Em desenvolvimento) faiss-cpu/gpu
- (Em desenvolvimento) sqlite3
- (Em desenvolvimento) structlog
- (Em desenvolvimento) pyyaml

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd pypdf
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

### Processamento de PDFs

```bash
python main.py caminho/do/diretório
```

### Executando Testes

```bash
python -m pytest -vv
```

## Métricas de Performance

- Processamento: 3.9 docs/segundo
- Média de chunks: 14.2 por documento
- Média de chunks por página: 2.5
- Tamanho médio de chunk: 790 caracteres

## Estado Atual

✅ Implementado:
- Sistema de ingestão de PDF
- Processamento e chunking de texto
- Geração de embeddings
- Sistema básico de métricas
- Normalização de texto

🔄 Em Desenvolvimento:
- Sistema de armazenamento
- Sistema de busca
- Sistema de logging
- Sistema de configuração

## Próximos Passos

1. Sistema de Armazenamento
   - FAISS para vetores
   - SQLite para metadados
   - Operações em lote

2. Sistema de Configuração
   - Arquivos YAML/JSON
   - Variáveis de ambiente
   - Valores padrão

3. Sistema de Logging
   - Logging estruturado JSON
   - Rotação de logs
   - Métricas de performance

5. Sistema de Busca
   - Busca por similaridade vetorial
   - Filtragem por metadados
   - Ordenação de resultados
