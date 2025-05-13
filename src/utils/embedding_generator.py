from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from src.utils.logger import get_logger
from src.config.models import EmbeddingConfig

class EmbeddingGenerator:
    """Gerador de embeddings para chunks de texto.
    
    Esta classe é responsável por gerar embeddings vetoriais para chunks de texto
    utilizando o modelo sentence-transformers. Os embeddings gerados podem ser
    utilizados para busca semântica e outras operações de similaridade.
    """
    
    def __init__(self, config: EmbeddingConfig, log_domain: str = "utils"):
        """Inicializa o gerador de embeddings.
        
        Args:
            config (EmbeddingConfig): Objeto de configuração contendo os parâmetros do embedding.
            log_domain (str): Domínio para o logger.
        """
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.config = config.model_copy(deep=True)
        self.logger.info(f"Inicializando o EmbeddingGenerator com configuração: {config}")
        
        self.model = SentenceTransformer(config.model_name, device=config.device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        self.logger.debug(f"Gerador de embeddings inicializado", 
                          model=self.config.model_name, 
                          device=self.config.device,
                          dimension=self.embedding_dimension, 
                          normalize=self.config.normalize_embeddings,
                          model_card_data=str(self.model.model_card_data)
                        )

    def update_config(self, new_config: EmbeddingConfig) -> None:
        """
        Atualiza a configuração do EmbeddingGenerator com base na configuração fornecida.

        Args:
            config (EmbeddingConfig): A nova configuração a ser aplicada.
        """
        if new_config == self.config:
            self.logger.info("Nenhuma alteracao na configuracao detectada")
            return

        if new_config.model_name != self.config.model_name:
            self.model = SentenceTransformer(new_config.model_name, device=new_config.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            self.logger.debug(f"Modelo de embedding atualizado para {new_config.model_name}")

        elif new_config.device != self.config.device:
            self.model.to(new_config.device)
            self.logger.debug(f"Dispositivo de embedding atualizado para {new_config.device}")

        self.config = new_config.model_copy(deep=True)
            
        self.logger.info("Configuracoes do EmbeddingGenerator atualizadas com sucesso")
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Calcula os embeddings para uma lista de chunks.
        
        Utiliza os parâmetros de batch_size e normalize_embeddings definidos na configuração.

        Args:
            chunks (List[str]): Lista de textos a serem processados
        
        Returns:
            np.ndarray: Array numpy de embeddings com shape (n_chunks, embedding_dimension)
                        onde n_chunks é o número de chunks de entrada
                        e embedding_dimension é a dimensão do embedding.
        """
        batch_size = self.config.batch_size
        normalize = self.config.normalize_embeddings
        
        self.logger.debug(f"Gerando embeddings para {len(chunks)} chunks; batch_size: {batch_size}, normalize: {normalize}")
        if not chunks:
            return np.array([])
        try:
            embeddings = self.model.encode(
                chunks, 
                batch_size=batch_size, 
                normalize_embeddings=normalize
            )
            self.logger.debug(f"Embeddings gerados com sucesso: {len(embeddings)} vetores, dimensão: {embeddings.shape}")
            return embeddings
        except Exception as e:
            self.logger.error(f"Erro ao gerar embeddings: {e}", exc_info=True)
            raise e
