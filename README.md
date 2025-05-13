# Sistema RAG de Ingestão de PDFs com GUI e testagem com consultas a LLMs

Sistema para processamento de documentos PDF em múltiplos domínios de conhecimento, com geração de embeddings, busca semântica, obtenção de respostas contextuais através de LLMs, e uma interface gráfica Streamlit para gerenciamento e consulta.

## Visão Geral

Este sistema implementa um pipeline RAG (Retrieval-Augmented Generation) para processar documentos PDF de um diretório dentro de "domínios" de conhecimento específicos. Para cada domínio ele cria um banco de dados e um índice FAISS. Ele extrai o texto de cada documento, divide em chunks, gera embeddings vetoriais, e armazena os dados em bancos de dados SQLite e índices vetoriais FAISS separados por domínio. Tanto um CLI quanto uma interface gráfica construída com Streamlit permitem ao usuário gerenciar esses domínios, iniciar a ingestão de documentos para um domínio selecionado, e realizar consultas que são respondidas por um LLM (através da API da Hugging Face) com base no contexto recuperado do domínio apropriado.

### Funcionalidades Principais

- **Gerenciamento de Domínios:** Criação, listagem, atualização e remoção de domínios de conhecimento isolados.
- **Ingestão de Documentos por Domínio:** Processamento de diretórios de PDFs para um domínio específico.
- **Pipeline RAG:**
    - Extração de texto de PDFs.
    - Detecção de duplicatas (intra-domínio) via hash MD5.
    - Divisão de texto em chunks semânticos.
    - Geração de embeddings (Sentence Transformers).
    - Normalização de texto.
    - Armazenamento de metadados e chunks (SQLite por domínio).
    - Armazenamento e busca vetorial de embeddings (FAISS por domínio).
- **Interface Gráfica (Streamlit):**
    - Gerenciamento de domínios.
    - Interface para ingestão de dados.
    - Interface de consulta para interagir com o LLM sobre domínios específicos.
- **Consulta Contextual:**
    - Busca por similaridade no índice FAISS do domínio selecionado.
    - Recuperação de chunks relevantes.
    - Integração com LLM (Hugging Face API) para geração de respostas baseadas no contexto recuperado.
- **Logging:** Sistema de log estruturado em JSON com rastreamento de contexto.
- **Testes:** Testes unitários e de integração (Pytest) para garantir a funcionalidade dos componentes.

## Estrutura do Projeto

```plaintext
/
├── .streamlit/
│   └── config.toml
├── gui/
│   └── streamlit_utils.toml       # Funções auxiliares do GUI
├── logs/
│   └── app.log           # Logs da aplicação em arquivo
├── pages/
│   ├── 1_Domain_Management.py
│   ├── 2_Data_Ingestion.py
│   ├── 3_Query_Interface.py
│   └── 4_Configuration.py # (Placeholder/Em desenvolvimento)
├── src/
│   ├── config/
│   │   ├── config_manager.py
│   │   ├── config_utils.py
│   │   ├── models.py       # Modelos das configurações
│   ├── data_ingestion/     # Lógica de ingestão de documentos
│   │   ├── __init__.py
│   │   ├── data_ingestion_orchestrator.py
│   │   ├── document_processor.py
│   │   └── chunking_strategy/  # Estratégias de chunk e seu gerenciador
│   │       ├── __init__.py
│   │       ├── chunking_manager.py
│   │       ├── chunking_strategy.py   # Abstract class
│   │       ├── recursive_strategy.py
│   │       └── semantic_cluster_strategy.py
│   ├── models/             # Modelos Pydantic para estruturas de dados
│   │   ├── __init__.py
│   │   ├── chunk.py
│   │   ├── document_file.py
│   │   ├── domain.py
│   │   └── domain_config.py
│   ├── query_processing/   # Lógica de consulta e interação com LLM
│   │   ├── __init__.py
│   │   ├── hugging_face_manager.py
│   │   └── query_orchestrator.py
│   ├── utils/              # Utilitários compartilhados
│   │   ├── __init__.py
│   │   ├── domain_manager.py
│   │   ├── embedding_generator.py
│   │   ├── faiss_manager.py
│   │   ├── logger.py
│   │   ├── sqlite_manager.py
│   │   └── text_normalizer.py
├── storage/
│   ├── control.db      # Banco de dados de controle (registros de domínio)
│   ├── domains/            # Armazenamento de dados por domínio
│   │   └── [domain_name]/  # Diretório para cada domínio criado
│   │       ├── [domain_name].db
│   │       └── vector_store/
│   │           └── [domain_name].faiss
│   └── schemas/            # Schemas SQL para bancos de dados
│       ├── control_schema.sql
│       └── schema.sql
├── tests/
│   ├── config/
│   │   └── test_config_manager.py/
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── test_data_ingestion_orchestrator.py
│   │   ├── test_document_processor.py
│   │   └── test_docs/
│   │       └── generate_test_pdfs.py # (Script auxiliar para testes)
│   ├── query_processing/
│   │   ├── __init__.py
│   │   ├── test_hugging_face_manager.py
│   │   └── test_query_orchestrator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── test_domain_manager.py
│   │   ├── test_embedding_generator.py
│   │   ├── test_faiss_manager.py
│   │   ├── test_logger.py
│   │   ├── test_sqlite_manager.py
│   │   └── test_text_normalizer.py
│   └── conftest.py         # Configurações e fixtures para Pytest
├── Admin.py                # Ponto de entrada principal da GUI Streamlit
├── main.py                 # Ponto de entrada principal da CLI (Incompleto)
├── README.md               
├── requirements.txt        
├── .env     
├── install.py             # Script de instalação do programa                    
└── .gitignore              
```

## Componentes Principais

### Orquestração e Gerenciamento

-   **`ConfigManager` (`src/config/config_manager.py`):** Gerencia o carregamento, validação, salvamento, backup e reset das configurações da aplicação (arquivo `config.toml`).
-   **`DomainManager` (`src/utils/domain_manager.py`):** Responsável por gerenciar os domínios de conhecimento (criar, listar, atualizar, deletar) e seus respectivos arquivos (banco de dados, índice vetorial).
-   **`ChunkingManager` (`src/data_ingestion/chunking_strategy/chunking_manager.py`):** Gerencia diferentes estratégias de chunking. Inicializa e delega para uma estratégia específica (ex. Recursive, SemanticCluster) de acordo com as definições de configuração.
-   **`DataIngestionOrchestrator` (`src/data_ingestion/data_ingestion_orchestrator.py`):** Coordena o pipeline completo de ingestão de documentos para um domínio específico, desde a leitura do PDF até o armazenamento dos chunks e embeddings.
-   **`QueryOrchestrator` (`src/query_processing/query_orchestrator.py`):** Gerencia o fluxo de consulta, incluindo a geração de embedding da query, busca no índice vetorial, recuperação de chunks, formatação do prompt e interação com o LLM.

### Gerenciamento de Dados e Vetores

-   **`SQLiteManager` (`src/utils/sqlite_manager.py`):** Interface para os bancos de dados SQLite. Gerencia um banco de controle (`control.db`) para os registros de domínio e bancos de dados específicos para cada domínio, armazenando metadados de documentos e chunks.
-   **`FaissManager` (`src/utils/faiss_manager.py`):** Interface para os índices vetoriais FAISS. Gerencia a criação, carregamento, adição de embeddings e busca por similaridade dentro do índice FAISS de cada domínio.

### Processamento de Documentos e Texto

-   **`DocumentProcessor` (`src/data_ingestion/document_processor.py`):** Extrai texto de arquivos PDF e calcula hashes para detecção de duplicatas.
-   **`Chunking Strategies` (`src/data_ingestion/chunking_strategy/`):**
    -   **`RecursiveStrategy` (`recursive_strategy.py`):** Realiza chunking por divisão recursiva de caracteres chave.
    -   **`SemanticClusterStrategy` (`semantic_cluster_strategy.py`):** Realiza chunking por clusterização semântica de segmentos de texto. contextually coherent chunks.
-   **`TextNormalizer` (`src/utils/text_normalizer.py`):** Aplica normalização ao texto (e.g., Unicode, lowercase) para consistência.
-   **`EmbeddingGenerator` (`src/utils/embedding_generator.py`):** Gera embeddings vetoriais para os chunks de texto usando modelos da biblioteca `sentence-transformers`.

### Interação com LLM

-   **`HuggingFaceManager` (`src/query_processing/hugging_face_manager.py`):** Interage com a API de Inferência da Hugging Face para enviar prompts formatados (incluindo o contexto recuperado) e obter respostas do modelo de linguagem.

### Interface Gráfica (GUI)

-   **`Admin.py` e `pages/`:** Aplicação Streamlit que fornece a interface para gerenciamento de domínios, ingestão de dados e consulta ao sistema RAG.

### Modelos de Dados

-   **`src/models/`:** Contém as definições (usando Pydantic) para as estruturas de dados `Domain`, `DomainConfig`, `DocumentFile`, e `Chunk`.

## Requisitos

- Python 3.10+
- Git (para clonar o repositório)
- Dependências principais:
  * streamlit
  * pydantic
  * pypdf 5.4.0
  * sentence-transformers 3.4.1
  * faiss-cpu 1.10.0
  * huggingface-hub 0.29.3
  * langchain 0.3.21 # Dependência pode ser indireta via langchain-text-splitters
  * langchain-text-splitters 0.3.7
  * SQLAlchemy 2.0.39
  * pytest 8.3.5
  * python-dotenv 1.0.1

## Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/fmossri/pbic-project.git
    cd pbic-project
    ```

2.  **Crie e ative um ambiente virtual:** (Recomendado)
    ```bash
    # Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # Windows (cmd/powershell)
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    python install * Substitui pip install -r requirements.txt, para garantir portabilidade através de sistemas com gpus diferentes, ou sem gpu
    ```

4.  **Configure o Token da Hugging Face:**
    *   Crie um token de acesso com permissões de leitura no site da [Hugging Face](https://huggingface.co/settings/tokens).
    *   Crie um arquivo chamado `.env` na raiz do projeto.
    *   Adicione a seguinte linha ao arquivo `.env`, substituindo `seu-token-aqui` pelo seu token:
        ```dotenv
        HUGGINGFACE_API_TOKEN="seu-token-aqui"
        ```
    ***As configurações de token podem mudar de acordo com o modelo escolhido. acessar página da Hugging Face para mais detalhes.

## Uso

### Via GUI

### Iniciando a Interface Gráfica

```bash
streamlit run Admin.py
```

### Via CLI (Temporariamente indisponível)

** main.py --help (Incompleto)

#### Adicionando Domínios
```bash
python main.py -d "nome do domínio", "breve descrição", "palavras-chave" [--debug]
```

#### Processando PDFs e Gerando Embeddings

```bash
python main.py -i caminho/do/diretório [--debug]
```

#### Realizando Consultas

```bash
python main.py -q "Sua pergunta aqui" [--debug]
```

#### Executando Testes

```bash
python -m pytest
```

## Estado Atual do Desenvolvimento

### Em Desenvolvimento 🔄

1. **Benchmarking**
    - Pesquisar estratégias de avaliação de sistemas RAG 
    - Implementar testagem e coleta de métricas relevantes no sistema.

2. **Novos Testes de Chunking**
    - Criar casos de teste para ChunkingManager, RecursiveStrategy e SemanticClusterStrategy

## Próximos Passos 🚀

1.  **Integração de OCR:**
    - Adicionar capacidade de OCR (Optical Character Recognition) à lógica de extração de conteúdo de PDFs, permitindo processar documentos baseados em imagem.

2.  **Avaliação de Modelos:**
    - Avaliar o desempenho e a adequação de diferentes modelos de embedding e LLMs para as tarefas específicas da aplicação.

3. **Remoção de Arquivos de um Domínio**
    - Implementar remoção de um arquivo do domínio. A lógica para a remoção do arquivo e seus chunks do banco de dados já existe; Desenvolver lógica de remoção dos embeddings associados ao documento do índice Faiss.

### Possíveis Melhorias 💡

   **Aprimoramento do Sistema de Configuração**
    - Criar sistema de configuração via CLI

   **Aprimoramento do Sistema de Consulta**
   - Otimização de prompts
   - Expansão de consultas usando sinônimos
   - Re-ranqueamento dos chunks recuperados
   - Atribuição de fontes para fundamentar as respostas

   **Expansão de Funcionalidades**
   - Adicionar suporte a outros tipos de documentos (e.g., .docx, .txt).
   - Implementar seleção de modelos de embedding e LLMs através da configuração.
   - Explorar outras estratégias avançadas de chunking e recuperação.
   - Implementar outras opções de normalização e tratamento de texto.

   **Funcionalidades Avançadas**
   - Busca híbrida com grafos de conhecimento
   - Processamento multi-modal (imagens, tabelas)
   - Uso de GPU (cuda)
   - Processamento paralelo

   **Saúde da Aplicação**
   - Otimizar performance e escalabilidade.
   - Implementar verificações de saúde da aplicação.
   - Limpar e padronizar logging e coleta de métricas.
   - Melhorar tratamento de erros e resiliência.

## Problemas Conhecidos

- **Erro do File Watcher do Streamlit com PyTorch:** Ao navegar para a página `Gerenciamento de Domínios`, um erro `RuntimeError: Tried to instantiate class '__path__._path'...` relacionado a `torch.classes` pode aparecer no console. Isso parece ser um problema com o file watcher do Streamlit tentando inspecionar a biblioteca `torch`. Tentativas de solucionar isso adicionando `torch` ou `.venv` à `folderWatchBlacklist` ou definindo `watchFileSystem = false` no arquivo `.streamlit/config.toml` não surtiram efeito. O erro parece ser apenas um ruído no console e não afeta a funcionalidade principal da GUI no momento. **Workaround: Silenciar Watcher em `.streamlit/config.toml` com `fileWatcherType = "none"`. Porém, ao modificarmos o código, necessitamos atualizar a página ou reiniciar o streamlit.

- **Visualização das configurações, em Configuration** A página `4_Configuration.py` mostra todas as propriedades da configuração atual, a partir da leitura do arquivo `config.toml`. Porém, após atualização do código, as definições de configuração relativas à Chunking, modelo de embedding, índice faiss e normalização de Embeddings foram passadas à área de criação de Domínio (Dado que essas configurações não podem ser mudadas após definidas, pois levam à inconsistência dos dados armazenados e potencialmente corrupção dos registros). Como a criação de um novo domínio não gera chamada à `ConfigManager.save_config()`, a exibição dos valores desses campos de configuração se torna incorreta. **Fix: Ou salvar o arquivo a cada novo domínio selecionado em DataIngestion (O que parece não ser desejável; o arquivo não deve ser alterado repetidamente); ou mover essas informações da seção `Configuração Atual` para a seção `Detalhes do Domínio`; ou fazer uma busca dessas informações no domínio armazenado na `session state`; ou dividir a seção em duas: Configurações gerais e configurações do domínio.

- **Caracteres especiais do PT-BR sendo passados para o JSON no logger** As mensagens de log com caractéres do Português, como â, ã, ç, etc., estão sendo passadas para o parsing JSON diretamente, sem tratamento. Como JSON não lida com esse tipo de caractére, as mensagens logadas se tornam defeituosas. **Fix: Criar uma função de tratamento de string que remova os caracteres especiais, e passar as mensagens através dela antes de enviar para o handler de arquivo.