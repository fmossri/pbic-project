# src/utils/faiss_manager.py

import faiss
import numpy as np
from typing import List
import os
from src.utils.logger import get_logger
class FaissManager:
    """Componente para armazenamento e indexação de vetores usando FAISS."""

    DEFAULT_INDEX_PATH: str = os.path.join("storage", "domains", "test_domain", "vector_store", "test.faiss")
    
    def __init__(self, log_domain: str = "utils"):
        """
        Inicializa o armazenamento vetorial.
        
        Args:
            index_path (str): Caminho do diretório para salvar/carregar o índice FAISS
            dimension (int): Dimensão dos vetores de embedding
        """
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info(f"Inicializando o FaissManager")

        self.index_path = None
        self.dimension = None
        self.index = None

        self.logger.debug(f"Index FAISS inicializado. dimensao: {self.dimension}; index_path: {self.index_path}")
    
    def _initialize_index(self) -> None:
        """Inicializa ou carrega o índice FAISS existente."""
        self.logger.info(f"Inicializando o indice FAISS em {self.index_path}")
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

            if os.path.exists(self.index_path):
                # Carrega o índice existente
                self.index = faiss.read_index(self.index_path)
                self.logger.debug(f"Indice FAISS carregado com sucesso: {self.index_path}")
            else:
                # Cria um novo índice
                self.logger.warning(f"Indice FAISS nao encontrado. Criando um novo indice")
                self.index = faiss.IndexFlatL2(self.dimension)
                # Salva o índice vazio
                faiss.write_index(self.index, self.index_path)
                self.logger.warning(f"Indice FAISS salvo com sucesso: {self.index_path}")
        except Exception as e:
            self.logger.error(f"Erro ao inicializar o índice FAISS: {e}")
            raise e
    
    def add_embeddings(self, embeddings: np.ndarray, vector_store_path: str, embedding_dimension: int) -> List[int]:
        """
        Adiciona embeddings ao índice.
        
        Args:
            embeddings (List[Embedding]): Lista de embeddings
        """
        self.logger.debug(f"Adicionando {len(embeddings)} embeddings ao índice FAISS")
        self.index_path = vector_store_path
        self.dimension = embedding_dimension
        try:
            self._initialize_index()
            # Obtém o número de embeddings já existentes no índice
            start_idx = self.index.ntotal
            # Adiciona os embeddings ao índice FAISS
            self.index.add(embeddings)

            faiss_indices = []
            for i in range(embeddings.shape[0]):  #embeddings.shape[0] dá o número de embeddings no ndarray.
                # Adiciona o índice do embedding à lista
                faiss_indices.append(start_idx + i)
                
            # Salva o estado
            self._save_state()
            self.logger.debug(f"Embeddings adicionados ao índice FAISS com sucesso")
            return faiss_indices

        except Exception as e:
            self.logger.error(f"Erro ao adicionar embeddings ao índice FAISS: {e}")
            raise e

    def search_faiss_index(self, query_embedding: np.ndarray, vector_store_path: str, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Realiza uma busca no índice FAISS para encontrar os k embeddings mais similares à query_embedding.
        
        Args:
            query_embedding (np.ndarray): Vetor de embedding da query
            k (int): Número de resultados a serem retornados
            vector_store_path (str): Caminho do índice FAISS

        Returns:
            tuple[np.ndarray, np.ndarray]: Tupla contendo as distâncias e os índices dos embeddings mais similares
        """
        if vector_store_path:
            self.index_path = vector_store_path
        else:
            self.logger.error(f"Caminho do indice FAISS nao fornecido ou invalido")
            raise ValueError(f"Caminho do índice FAISS não fornecido ou inválido")
        
        self.logger.info(f"Realizando busca por similaridade no indice FAISS", top_k=k)

        try:
            self._initialize_index()
            # Verifica se o vetor de embedding é um array unidimensional
            if len(query_embedding.shape) == 1:
                # Se for unidimensional, converte para bidimensional
                query_embedding = query_embedding.reshape(1, -1)

            # Realiza a busca no índice FAISS
            distances, indices = self.index.search(query_embedding, k)

            flat_indices = indices.flatten().tolist()
            self.logger.debug(
                f"Busca no indice FAISS realizada com sucesso", 
                distances_shape=distances.shape, 
                indices_shape=indices.shape,
                indices_flat=flat_indices 
            )
            # ================================================
            return distances, indices
        except Exception as e:
            self.logger.error(f"Erro ao realizar busca no indice FAISS: {e}")
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
