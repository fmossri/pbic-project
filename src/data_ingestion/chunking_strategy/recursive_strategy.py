from typing import List, Optional, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.models import Chunk
from src.config.models import IngestionConfig

from .chunking_strategy import ChunkingStrategy

class RecursiveStrategy(ChunkingStrategy):
    """Gerencia a divisão de conteúdo de texto em chunks.
       Utiliza a estratégia e os parâmetros definidos na configuração.
    """
    
    def __init__(self, config: IngestionConfig, log_domain: str = "text-chunker"):
        """
        Inicializa o RecursiveStrategy com base na configuração fornecida.
        
        Args:
            config (IngestionConfig): Objeto de configuração contendo os parâmetros de ingestão.
            log_domain (str): Domínio para o logger.
        """
        super().__init__(config, log_domain)

        self.logger.info("Inicializando o RecursiveStrategy.", config_data=config.model_dump())
        self.model = RecursiveCharacterTextSplitter(
                        chunk_size=self.config.chunk_size, 
                        chunk_overlap=self.config.chunk_overlap,
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

    def update_config(self, new_config: IngestionConfig) -> None:
        """
        Atualiza a configuração do RecursiveStrategy com base na configuração fornecida.

        Args:
            new_config (IngestionConfig): A nova configuração a ser aplicada.
        """
        if new_config == self.config:
            self.logger.info("Nenhuma alteracao na configuracao detectada")
            return

        if new_config.chunk_size != self.config.chunk_size or new_config.chunk_overlap != self.config.chunk_overlap:
            self.model._chunk_size, self.model._chunk_overlap = new_config.chunk_size, new_config.chunk_overlap
            
            self.logger.info(f"Parametros de chunking recursivo alterados para: {new_config.model_dump()}")

        self.config = new_config
        self.logger.info("Configuracoes do RecursiveStrategy atualizadas com sucesso")


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
            docs = self.model.create_documents(
                texts=[text], 
                metadatas=[metadata] if metadata else None
            )
            self.logger.debug(f"Texto dividido em {len(docs)} chunks (Langchain Document) com sucesso.")
            return docs
        except Exception as e:
            self.logger.error(f"Erro ao dividir o texto em chunks (Langchain Document): {e}", exc_info=True)
            raise e
    
    def create_chunks(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Divide o texto em objetos Chunk, enriquecidos com metadados.

        Args:
            text (str): Texto completo a ser dividido.
            metadata (Optional[Dict]): Metadados base (ex: document_id, page_number) 
                                      a serem adicionados a cada chunk.

        Returns:
            List[Chunk]: Lista de objetos Chunk criados.
        """
        if not metadata or not isinstance(metadata, dict):
             metadata = {}
             self.logger.warning("Metadados nao fornecidos para create_chunks.")

        docs = self._chunk_text(text, metadata)

        chunks_list: List[Chunk] = []
        try:
            for chunk_index, doc in enumerate(docs):
                doc_metadata = doc.metadata if hasattr(doc, 'metadata') else metadata
                
                start_index = doc_metadata.get("start_index", -1)
                if start_index == -1:
                     self.logger.warning(f"start_index nao encontrado nos metadados do chunk {chunk_index}. Usando -1.")

                chunk = Chunk(
                    document_id=doc_metadata.get("document_id", 0),
                    page_number=doc_metadata.get("page_number", 0),
                    chunk_page_index=chunk_index,
                    chunk_start_char_position=start_index, 
                    content=doc.page_content,
                )
                chunks_list.append(chunk)
            
            self.logger.info(f"Criados {len(chunks_list)} objetos Chunk.")
            return chunks_list
        except Exception as e:
            self.logger.error(f"Erro ao criar objetos Chunk: {e}", exc_info=True)
            raise e