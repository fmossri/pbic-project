# Sistema de Ingestão e Processamento de PDFs

Este projeto implementa um sistema modular para ingestão, processamento e geração de embeddings de documentos PDF.

## Arquitetura do Sistema

### Componentes Principais

#### 1. Ingestão de Dados (`components/data_ingestion/`)
- **Componente Principal** (`data_ingestion_component.py`)
  - Responsável pela coordenação do processo de ingestão
  - Gerencia o fluxo de processamento dos documentos
  - Integra todos os subcomponentes de processamento
  - Detecta e gerencia documentos duplicados

- **Processador de Documentos** (`document_processor.py`)
  - Responsável pela leitura e extração de texto de PDFs
  - Implementa a lógica de processamento de documentos
  - Gerencia a conversão de PDF para texto
  - Calcula hashes para detecção de duplicatas

- **Chunking de Texto** (`text_chunker.py`)
  - Divide o texto em chunks menores e gerenciáveis
  - Implementa estratégias de divisão de texto
  - Mantém a coerência semântica dos chunks
  - Permite configuração de tamanho e sobreposição

#### 2. Geração de Embeddings (`components/embedding_generator/`)
- **Gerador de Embeddings** (`embedding_generator_component.py`)
  - Responsável pela geração de embeddings vetoriais
  - Utiliza o modelo sentence-transformers
  - Processa chunks de texto em lotes
  - Gera embeddings de dimensão configurável

### Fluxo de Processamento

1. **Ingestão de Documentos**
   - Leitura dos PDFs do diretório de entrada
   - Verificação de duplicatas via hash
   - Extração de texto dos documentos
   - Pré-processamento do texto extraído

2. **Processamento de Texto**
   - Divisão do texto em chunks
   - Aplicação de regras de processamento
   - Preparação para geração de embeddings

3. **Geração de Embeddings**
   - Conversão de chunks de texto em vetores
   - Processamento em lotes para eficiência
   - Armazenamento dos embeddings com metadados

## Uso do Sistema

Para executar o sistema:

```bash
# Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Executar o processamento
python main.py ../sample_docs
```

O sistema processará todos os PDFs no diretório especificado, gerando embeddings para cada chunk de texto e exibindo métricas detalhadas do processamento.

## Estrutura de Diretórios

```
.
├── components/
│   ├── data_ingestion/
│   │   ├── data_ingestion_component.py
│   │   ├── document_processor.py
│   │   └── text_chunker.py
│   └── embedding_generator/
│       └── embedding_generator.py

├── tests/
│   ├── data_ingestion/
│   │   ├── test_document_processor.py
│   │   ├── test_data_ingestion_component.py
│   │   ├── test_text_chunker.py
│   │   └── test_docs/
│   │       └── test_pdf_generator.py
│   └── embedding_generator/
│       └── test_embedding_generator.py
├── main.py
└── requirements.txt
```

## Métricas e Monitoramento

O sistema fornece métricas detalhadas sobre o processamento:
- Total de documentos processados
- Número de páginas e chunks gerados
- Estatísticas de embeddings gerados
- Tempo de processamento e performance

## Requisitos

- Python 3.x
- Dependências listadas em `requirements.txt`
- Memória suficiente para processamento de embeddings

## Status do Projeto

- ✅ Sistema de ingestão de PDFs implementado
- ✅ Processamento de texto e chunking funcionando
- ✅ Geração de embeddings implementada
- 🔄 Sistema de armazenamento em desenvolvimento
- 🔄 Sistema de busca em desenvolvimento 