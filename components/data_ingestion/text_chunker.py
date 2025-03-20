from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    """Gerencia a divisão de conteúdo de texto em chunks."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Inicializa o TextChunker.
        
        Args:
            chunk_size (int): Tamanho de cada chunk em caracteres
            overlap (int): Número de caracteres para sobreposição entre chunks
        """
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

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, str]]:
        """
        Divide o texto em chunks sobrepostos.
        
        Args:
            text (str): Texto a ser dividido em chunks
            metadata (Optional[Dict]): Metadados adicionais para os chunks
            
        Returns:
            List[Dict[str, str]]: Lista de chunks com seus metadados
        """
        if not text:
            return []
            
        # Cria documentos com metadados
        docs = self.splitter.create_documents(
            texts=[text],
            metadatas=[metadata] if metadata else None
        )
        
        # Converte para formato mais simples
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "start_index": doc.metadata.get("start_index", 0)
            }
            for doc in docs
        ] 