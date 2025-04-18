from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.models import Chunk
from src.utils.logger import get_logger

class TextChunker:
    """Gerencia a divisão de conteúdo de texto em chunks."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200, log_domain: str = "Ingestao de dados"):
        """
        Inicializa o TextChunker.
        
        Args:
            chunk_size (int): Tamanho de cada chunk em caracteres
            overlap (int): Número de caracteres para sobreposição entre chunks
        """
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info("Inicializando o TextChunker.", chunk_size=chunk_size, overlap=overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
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
            add_start_index=True  # Útil para debug
        )

    def _chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Divide o texto em chunks com metadados.
        
        Args:
            text (str): Texto a ser dividido em chunks
            metadata (Optional[Dict]): Metadados adicionais para os chunks
            
        Returns:
            List[Document]: Lista de chunks com seus metadados
        """
        self.logger.debug("Dividindo a pagina em chunks")
        if not text:
            self.logger.error("Texto nao encontrado")
            return []
        try:
            # Cria documentos com metadados
            docs = self.splitter.create_documents(
                texts=[text],
                metadatas=[metadata] if metadata else None
            )
            self.logger.debug(f"Pagina dividida em {len(docs)} chunks com sucesso")
            return docs
        except Exception as e:
            self.logger.error(f"Erro ao dividir a pagina em chunks: {e}")
            raise e
    
    def create_chunks(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Divide o texto em chunks com metadados.
        """
        docs =self._chunk_text(text, metadata)

        chunks = []
        try:
            for chunk_index, doc in enumerate(docs):
                chunk = Chunk(
                    document_id = doc.metadata.get("document_id", 0),
                    page_number = doc.metadata.get("page_number", 0),
                    chunk_page_index = chunk_index,
                    chunk_start_char_position = doc.metadata["start_index"],
                    content = doc.page_content,
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            self.logger.error(f"Erro ao criar objetos Chunk: {e}")
            raise e
