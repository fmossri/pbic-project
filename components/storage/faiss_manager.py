# components/storage/vector_store.py

import faiss
import numpy as np
from typing import List
import os
from components.models import Embedding
class FaissManager:
    """Componente para armazenamento e indexação de vetores usando FAISS."""

    DEFAULT_INDEX_PATH: str = os.path.join("indices", "public", "index.faiss")
    
    def __init__(self, index_path: str = DEFAULT_INDEX_PATH, dimension: int = 384):
        """
        Inicializa o armazenamento vetorial.
        
        Args:
            index_path (str): Caminho do diretório para salvar/carregar o índice FAISS
            dimension (int): Dimensão dos vetores de embedding
        """
        print(f"Inicializando o index FAISS em {index_path}")
        self.index_path = index_path if index_path is not None else self.DEFAULT_INDEX_PATH
        self.dimension = dimension
        self.index = None
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Inicializa ou carrega o índice FAISS existente."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        
        if os.path.exists(self.index_path):
            # Carrega o índice existente
            self.index = faiss.read_index(self.index_path)
        else:
            # Cria um novo índice
            self.index = faiss.IndexFlatL2(self.dimension)
            # Salva o índice vazio
            faiss.write_index(self.index, self.index_path)
    
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

        # Adiciona os embeddings ao índice FAISS
        self.index.add(embedding_values)

        for i, embedding in enumerate(embeddings):
            embedding.chunk_faiss_index = start_idx + i
            embedding.faiss_index_path = self.index_path
        
        # Salva o estado
        self._save_state()

    def search_faiss_index(self, query_embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Realiza uma busca no índice FAISS para encontrar os k embeddings mais similares à query_embedding.
        
        Args:
            query_embedding (np.ndarray): Vetor de embedding da query
        """
        # Verifica se o vetor de embedding é um array unidimensional
        if len(query_embedding.shape) == 1:
            # Se for unidimensional, converte para bidimensional
            query_embedding = query_embedding.reshape(1, -1)
        
        # Realiza a busca no índice FAISS
        distances, indices = self.index.search(query_embedding, k)
        
        return distances, indices
    
    def _save_state(self) -> None:
        """Salva o estado do índice."""
        faiss.write_index(self.index, self.index_path)