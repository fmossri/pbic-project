from typing import List, Optional, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from src.models import Chunk, DocumentFile
from src.config.models import AppConfig

from .chunking_strategy import ChunkingStrategy

class RecursiveStrategy(ChunkingStrategy):
    """Gerencia a divisão de conteúdo de texto em chunks.
       Utiliza a estratégia e os parâmetros definidos na configuração.
    """
    
    def __init__(self, config: AppConfig, log_domain: str = "text-chunker"):
        """
        Inicializa o RecursiveStrategy com base na configuração fornecida.
        
        Args:
            config (AppConfig): Objeto de configuração contendo os parâmetros de ingestão.
            log_domain (str): Domínio para o logger.
        """
        super().__init__(config, log_domain)

        self.logger.info("Inicializando o RecursiveStrategy.", config_data=config.model_dump())
        self.splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.config.ingestion.chunk_size, 
                        chunk_overlap=self.config.ingestion.chunk_overlap,
                        separators=[
                            "\n\n",  # Parágrafos
                            "\n",    # Quebras de linha
                            ".",     # Pontos
                            "!",     # Exclamações
                            "?",     # Interrogações
                            ";",     # Ponto e vírgula
                            ":",     # Dois pontos
                            ",",     # Vírgulas
                            " ",     # Espaços
                            ""       # Fallback para divisão por caractere
                        ],
                        length_function=len,
                        add_start_index=True
                    )

    def _chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        Divide o texto em chunks (Langchain Document) com metadados.
        
        Args:
            text (str): Texto a ser dividido em chunks
            metadata (Optional[Dict]): Metadados adicionais para os chunks
            
        Returns:
            List[Document]: Lista de documentos Langchain resultantes da divisão.
        """
        self.logger.debug("Dividindo texto em chunks (Langchain Document).", text_length=len(text))
        if not text:
            self.logger.warning("Texto vazio fornecido para chunking.")
            return []
        try:
            docs = self.splitter.create_documents(
                texts=[text], 
                metadatas=[metadata] if metadata else None
            )
            self.logger.debug(f"Texto dividido em {len(docs)} chunks (Langchain Document) com sucesso.")
            return docs
        except Exception as e:
            self.logger.error(f"Erro ao dividir o texto em chunks (Langchain Document): {e}", exc_info=True)
            raise e
    
    def create_chunks(self, file: DocumentFile) -> List[Chunk]:
        """
        Divide o texto em objetos Chunk, enriquecidos com metadados.

        Args:
            text (str): Texto completo a ser dividido.
            metadata (Optional[Dict]): Metadados base (ex: document_id, page_number) 
                                      a serem adicionados a cada chunk.

        Returns:
            List[Chunk]: Lista de objetos Chunk criados.
        """
        document_chunks: List[Chunk] = []
        page_counter = 0
        metadata = {"document_id": file.id,
                    "page_number": -1
                    }

        self.logger.info(f"Iniciando chunking recursivo para o documento: {file.id} ({file.name}) com {len(file.pages)} páginas.")
        try:

            for page in file.pages:
                page_counter += 1
                self.logger.debug(f"Processando pagina: {page_counter}/{len(file.pages)}")
                
                if not page.page_content:
                    self.logger.warning(f"Página {page_counter} do documento {file.id} ({file.name}) está vazia. Pulando página.")
                    continue

                metadata["page_number"] = page.metadata.get("page")
                
                try:
                    page_docs = self._chunk_text(page.page_content, metadata)

                    if not page_docs:
                        self.logger.warning(f"Nenhum Chunk (Langchain Document) gerado por _chunk_text para a página {page_counter} do arquivo {file.name}. Pulando página.")
                        continue



                    page_chunks: List[Chunk] = []
                    for chunk_index, doc in enumerate(page_docs):
                        doc_metadata = doc.metadata if hasattr(doc, 'metadata') else metadata
                        
                        start_index = doc_metadata.get("start_index", -1)
                        if start_index == -1:
                            self.logger.warning(f"start_index nao encontrado nos metadados do chunk {chunk_index}. Usando -1.")

                        metadata = {"page_list": [doc_metadata.get("page_number", -1)]}
                        chunk = Chunk(
                            document_id=doc_metadata.get("document_id", -1),
                            metadata=metadata,
                            content=doc.page_content,
                        )
                        page_chunks.append(chunk)

                    if not page_chunks:
                        self.logger.warning(f"Nenhum objeto Chunk criado para a página {page_counter} do arquivo {file.name} (após conversão). Pulando página.")
                        continue

                    self.logger.info(f"Criados {len(page_chunks)} objetos Chunk recursivos para a página {page_counter} do arquivo {file.name}.")
                    document_chunks.extend(page_chunks)

                except Exception as e:
                    self.logger.error(f"Erro ao processar chunks para a página {page_counter} do arquivo {file.id} ({file.name}): {e}", exc_info=True)
                    continue
                
        except Exception as e:
            self.logger.error(f"Erro ao criar objetos Chunk para o arquivo {file.name}: {e}", exc_info=True)
            raise e
            
        self.logger.info(f"Chunking recursivo concluído para o arquivo {file.id}. Total de chunks criados: {len(document_chunks)}.")
        
        # Gera keywords para cada chunk
        keywords = self._generate_keywords(document_chunks)
        
        for i, chunk in enumerate(document_chunks):
            chunk.metadata["indices_list"] = [i]
            chunk.metadata["keywords"] = keywords[i]

        return document_chunks