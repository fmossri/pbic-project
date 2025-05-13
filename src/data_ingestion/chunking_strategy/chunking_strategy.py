from abc import ABC, abstractmethod
from typing import List

from src.models import Chunk, DocumentFile
from src.config.models import AppConfig
from src.utils.logger import get_logger
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

class ChunkingStrategy(ABC):
    def __init__(self, config: AppConfig, log_domain: str):
        self.config = config
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.embedding_model = SentenceTransformer(self.config.embedding.model_name)
        self.keybert_model = KeyBERT(model=self.embedding_model)
        self.logger.debug(f"Inicializando a estratégia de chunking: {self.__class__.__name__}")
        
    @abstractmethod
    def create_chunks(self, file: DocumentFile) -> List[Chunk]:
        pass

    def update_config(self, new_config: AppConfig) -> None:
        """
        Atualiza a configuração do SemanticClusterStrategy com base na configuração fornecida.

        Args:
            new_config (AppConfig): A nova configuração a ser aplicada.
        """
        if new_config == self.config:
            self.logger.info("Nenhuma alteracao na configuracao detectada")
            return
        
        if new_config.ingestion.chunk_size != self.config.ingestion.chunk_size or new_config.ingestion.chunk_overlap != self.config.ingestion.chunk_overlap:
            self.splitter._chunk_size, self.splitter._chunk_overlap = new_config.ingestion.chunk_size, new_config.ingestion.chunk_overlap
            self.logger.info(f"Parametros de chunking do {self.__class__.__name__} alterados. chunk_size: {self.splitter._chunk_size}, chunk_overlap: {self.splitter._chunk_overlap}")

        if new_config.embedding.model_name != self.config.embedding.model_name:
            self.embedding_model = SentenceTransformer(new_config.embedding.model_name)
            self.keybert_model = KeyBERT(model=self.embedding_model)
            self.logger.info(f"Modelo SentenceTransformer do {self.__class__.__name__} alterado para: {new_config.embedding.model_name}")

        if new_config.embedding.device != self.config.embedding.device:
            self.embedding_model.to(new_config.embedding.device)
            self.logger.info(f"Dispositivo de embedding do {self.__class__.__name__} alterado para: {new_config.embedding.device}")

        self.config = new_config
        self.logger.info(f"Configuracoes do {self.__class__.__name__} atualizadas com sucesso")

    def _generate_keywords(self, big_chunks: List[str]) -> List[List[str]]:
        """
        Gera keywords para cada chunk usando KeyBERT.
        
        Args:
            big_chunks (List[str]): Lista de chunks grandes para gerar keywords
            
        Returns:
            List[List[str]]: Lista de listas de keywords para cada chunk
        """
        self.logger.info("Passo 9: Gerando keywords para cada chunk...")
        keywords = []
        try:
            for chunk in big_chunks:
                # Extrai keywords usando KeyBERT
                chunk_keywords = self.keybert_model.extract_keywords(
                    chunk, 
                    keyphrase_ngram_range=(1, 2), 
                    stop_words='portuguese',  # Usando stop words em português - Desconsidera palavras comuns como "de", "o", "nas", etc.
                    top_n=3,
                    diversity=0.5,
                    use_maxsum=True,
                    nr_candidates=20
                )
                # Extrai apenas as keywords das tuplas (keyword, score)
                keywords.append([kw[0] for kw in chunk_keywords])
            
            self.logger.info(f"Keywords geradas para {len(keywords)} chunks")
            return keywords
        except Exception as e:
            self.logger.error(f"Erro ao gerar keywords: {str(e)}")
            # Retorna lista vazia de keywords em caso de erro
            return [[] for _ in big_chunks]