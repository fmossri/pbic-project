from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
from src.utils.logger import get_logger
class EmbeddingGenerator:
    """Gerador de embeddings para chunks de texto.
    
    Esta classe é responsável por gerar embeddings vetoriais para chunks de texto
    utilizando o modelo sentence-transformers. Os embeddings gerados podem ser
    utilizados para busca semântica e outras operações de similaridade.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", log_domain: str = "utils"):
        """Inicializa o gerador de embeddings.
        
        Args:
            model_name (str): Nome do modelo sentence-transformers a ser utilizado.
                            Default: "all-MiniLM-L6-v2"
        """
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info(f"Inicializando o EmbeddingGenerator. Modelo: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.logger.debug(f"Gerador de embeddings inicializado", 
                          model = self.model_name, 
                          dimension = self.embedding_dimension, 
                          model_card_data = str(self.model.model_card_data)
                        )

    
    def generate_embeddings(self, chunks: List[str], batch_size: Optional[int] = 32) -> np.ndarray:
        """Calcula os embeddings para uma lista de chunks.
        
        Args:
            chunks (List[str]): Lista de textos a serem processados
            batch_size (Optional[int]): Tamanho do batch de chunks para processamento.
                                      Default: 32
        
        Returns:
            np.ndarray: Array numpy de embeddings com shape (n_chunks, embedding_dimension)
                        onde n_chunks é o número de chunks de entrada
                        e embedding_dimension é a dimensão do embedding (384 para all-MiniLM-L6-v2)
        """
        self.logger.debug(f"Gerando embeddings para {len(chunks)} chunks; batch_size: {batch_size}")
        if not chunks:
            return np.array([])
        try:
            embeddings = self.model.encode(chunks, batch_size=batch_size, normalize_embeddings=True)
            self.logger.debug(f"Embeddings gerados com sucesso: {len(embeddings)} vetores, dimensão: {embeddings.shape}")
            return embeddings
        except Exception as e:
            self.logger.error(f"Erro ao gerar embeddings: {e}")
            raise e
