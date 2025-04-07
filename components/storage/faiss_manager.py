# components/storage/vector_store.py

import faiss
import numpy as np
from typing import List
import os
from components.models import Embedding
class FaissManager:
    """Componente para armazenamento e indexação de vetores usando FAISS."""
    
    def __init__(self, index_path: str = os.path.join("indices"), index_file: str = "index.faiss", dimension: int = 384):
        """
        Inicializa o armazenamento vetorial.
        
        Args:
            index_path (str): Caminho do diretório para salvar/carregar o índice FAISS
            dimension (int): Dimensão dos vetores de embedding
        """
        self.index_path = index_path
        self.index_file = os.path.join(self.index_path, index_file)
        self.dimension = dimension
        self.index = None
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Inicializa ou carrega o índice FAISS existente."""
        os.makedirs(self.index_path, exist_ok=True)
        
        
        if os.path.exists(self.index_file):
            # Carrega o índice existente
            self.index = faiss.read_index(self.index_file)
        else:
            # Cria um novo índice
            self.index = faiss.IndexFlatL2(self.dimension)
            # Salva o índice vazio
            faiss.write_index(self.index, self.index_file)
    
    def add_embeddings(self, embeddings: List[Embedding]) -> None:
        """
        Adiciona embeddings ao índice.
        
        Args:
            embeddings (List[Embedding]): Lista de embeddings
        """
        # Obtém o número de embeddings já existentes no índice
        start_idx = self.index.ntotal
        # Extrai os vetores dos objetos Embedding
        embedding_values = np.vstack([embedding.embedding for embedding in embeddings])
        
        for i, embedding in enumerate(embeddings, 1):
            embedding.chunk_faiss_index = start_idx + i
        # Adiciona os embeddings ao índice FAISS
        self.index.add(embedding_values)
        
        # Salva o estado
        self._save_state()
    
    def _save_state(self) -> None:
        """Salva o estado do índice."""
        faiss.write_index(self.index, self.index_file)