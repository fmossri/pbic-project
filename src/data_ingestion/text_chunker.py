from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.models import Chunk

class TextChunker:
    """Gerencia a divisão de conteúdo de texto em chunks."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Inicializa o TextChunker.
        
        Args:
            chunk_size (int): Tamanho de cada chunk em caracteres
            overlap (int): Número de caracteres para sobreposição entre chunks
        """
        print(f"Inicializando o TextChunker: chunk_size {chunk_size}; overlap {overlap}")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
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
        if not text:
            return []
            
        # Cria documentos com metadados
        docs = self.splitter.create_documents(
            texts=[text],
            metadatas=[metadata] if metadata else None
        )
 
        return docs
    
    def create_chunks(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Divide o texto em chunks com metadados.
        """
        docs =self._chunk_text(text, metadata)

        chunks = []
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