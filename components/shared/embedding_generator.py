from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
from components.models import Embedding
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

    
    def generate_embeddings(self, chunks: List[Tuple[str, int]], batch_size: Optional[int] = 32) -> List[Embedding]:
        """Calcula os embeddings para uma lista de chunks.
        
        Args:
            chunks (List[str]): Lista de chunks a serem processados
            batch_size (Optional[int]): Tamanho do batch de chunks para processamento.
                                      Default: 32
        
        Returns:
            List[Embedding]: Lista de objetos Embedding
        """
        if not chunks:
            return []
        
        chunks_text, chunk_ids = zip(*chunks)
        result = self.model.encode(chunks_text, batch_size=batch_size)
        embeddings = []

        for embedding, chunk_id in zip(result, chunk_ids):
            embedding = Embedding(
                id = None,
                chunk_id = chunk_id,
                faiss_index_path = None,
                chunk_faiss_index = None,
                dimension = self.embedding_dimension, 
                embedding = embedding
            )

            embeddings.append(embedding)

        return embeddings
