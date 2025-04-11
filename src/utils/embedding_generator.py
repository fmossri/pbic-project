from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np

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
        
        print(f"Inicializando o gerador de embeddings: modelo {model_name}; dimensão {self.embedding_dimension}")

    
    def generate_embeddings(self, chunks: List[str] | str, batch_size: Optional[int] = 32) -> np.ndarray:
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
        if not chunks:
            return np.array([])
        
        embeddings = self.model.encode(chunks, batch_size=batch_size)

        return embeddings
