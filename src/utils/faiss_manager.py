# src/utils/faiss_manager.py

import faiss
import numpy as np
from typing import List
import os
from src.models import Embedding
from src.utils.logger import get_logger
class FaissManager:
    """Componente para armazenamento e indexação de vetores usando FAISS."""

    DEFAULT_INDEX_PATH: str = os.path.join("storage", "domains", "test_domain", "vector_store", "test.faiss")
    
    def __init__(self, index_path: str = DEFAULT_INDEX_PATH, dimension: int = 384, log_domain: str = "utils"):
        """
        Inicializa o armazenamento vetorial.
        
        Args:
            index_path (str): Caminho do diretório para salvar/carregar o índice FAISS
            dimension (int): Dimensão dos vetores de embedding
        """
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info(f"Inicializando o FaissManager")

        self.index_path = index_path if index_path is not None else self.DEFAULT_INDEX_PATH
        self.dimension = dimension
        self.index = None
        self._initialize_index()

        self.logger.debug(f"Index FAISS inicializado. dimensão: {self.dimension}; index_path: {self.index_path}")
    
    def _initialize_index(self) -> None:
        """Inicializa ou carrega o índice FAISS existente."""
        self.logger.info(f"Inicializando o índice FAISS em {self.index_path}")
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

            if os.path.exists(self.index_path):
                # Carrega o índice existente
                self.index = faiss.read_index(self.index_path)
                self.logger.debug(f"Índice FAISS carregado com sucesso: {self.index_path}")
            else:
                # Cria um novo índice
                self.logger.warning(f"Índice FAISS não encontrado. Criando um novo índice")
                self.index = faiss.IndexFlatL2(self.dimension)
                # Salva o índice vazio
                faiss.write_index(self.index, self.index_path)
                self.logger.warning(f"Índice FAISS salvo com sucesso: {self.index_path}")
        except Exception as e:
            self.logger.error(f"Erro ao inicializar o índice FAISS: {e}")
            raise e
    
    def add_embeddings(self, embeddings: List[Embedding]) -> None:
        """
        Adiciona embeddings ao índice.
        
        Args:
            embeddings (List[Embedding]): Lista de embeddings
        """
        self.logger.debug(f"Adicionando {len(embeddings)} embeddings ao índice FAISS")
        # Obtém o número de embeddings já existentes no índice

        try:
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

            self.logger.debug(f"Embeddings adicionados ao índice FAISS com sucesso")

        except Exception as e:
            self.logger.error(f"Erro ao adicionar embeddings ao índice FAISS: {e}")
            raise e

    def search_faiss_index(self, query_embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Realiza uma busca no índice FAISS para encontrar os k embeddings mais similares à query_embedding.
        
        Args:
            query_embedding (np.ndarray): Vetor de embedding da query
        """
        self.logger.info(f"Realizando busca por similaridade no índice FAISS", top_k=k)
        # Verifica se o vetor de embedding é um array unidimensional
        try:
            if len(query_embedding.shape) == 1:
                # Se for unidimensional, converte para bidimensional
                query_embedding = query_embedding.reshape(1, -1)

            # Realiza a busca no índice FAISS
            distances, indices = self.index.search(query_embedding, k)
            self.logger.debug(f"Busca no índice FAISS realizada com sucesso")
            return distances, indices
        except Exception as e:
            self.logger.error(f"Erro ao realizar busca no índice FAISS: {e}")
            raise e
        
    def _save_state(self) -> None:
        """Salva o estado do índice."""
        self.logger.info(f"Salvando o estado do índice FAISS em: {self.index_path}")
        try:
            faiss.write_index(self.index, self.index_path)
            self.logger.info(f"Estado do índice FAISS salvo com sucesso: {self.index_path}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar o estado do índice FAISS: {e}")
            raise e
