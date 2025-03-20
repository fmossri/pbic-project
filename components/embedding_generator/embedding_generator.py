from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
import logging

class EmbeddingGenerator:
    """Gerador de embeddings para chunks de texto.
    
    Esta classe é responsável por gerar embeddings vetoriais para chunks de texto
    utilizando o modelo sentence-transformers. Os embeddings gerados podem ser
    utilizados para busca semântica e outras operações de similaridade.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Inicializa o gerador de embeddings.
        
        Args:
            model_name (str): Nome do modelo sentence-transformers a ser utilizado.
                            Default: "all-MiniLM-L6-v2"
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logging.info(f"Inicializado EmbeddingGenerator com modelo {model_name}")
        logging.info(f"Dimensão dos embeddings: {self.embedding_dimension}")

    def calculate_embeddings(self, chunks: List[str], batch_size: Optional[int] = 32) -> np.ndarray:
        """Calcula os embeddings para uma lista de chunks.
        
        Args:
            chunks (List[str]): Lista de chunks a serem processados
            batch_size (Optional[int]): Tamanho do batch de chunks para processamento.
                                      Default: 32
        
        Returns:
            np.ndarray: Array numpy contendo os embeddings (shape: [n_chunks, embedding_dimension])
        """
        if not chunks:
            logging.warning("Lista de chunks vazia")
            return np.array([])
            
        logging.info(f"Calculando embeddings para {len(chunks)} chunks")
        embeddings = self.model.encode(chunks, batch_size=batch_size)
        logging.info(f"Embeddings calculados com sucesso. Shape: {embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Retorna a dimensão dos embeddings gerados.
        
        Returns:
            int: Dimensão dos vetores de embedding
        """
        return self.embedding_dimension

    def get_model_info(self) -> Dict:
        """Retorna informações sobre o modelo em uso.
        
        Returns:
            Dict: Dicionário contendo informações do modelo
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self.model.max_seq_length
        }
