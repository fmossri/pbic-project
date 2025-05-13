from typing import List, Dict, Optional

from src.models import Chunk, DocumentFile
from src.utils.logger import get_logger
from src.config.models import AppConfig

from .chunking_strategy import ChunkingStrategy
from .recursive_strategy import RecursiveStrategy
from .semantic_cluster_strategy import SemanticClusterStrategy


class ChunkingManager:
    """Gerencia a divisão de conteúdo de texto em chunks.
       Utiliza a estratégia e os parâmetros definidos na configuração.
    """
    
    def __init__(self, config: AppConfig, log_domain: str = "text-chunker"):
        """
        Inicializa o TextChunker com base na configuração fornecida.
        
        Args:
            config (AppConfig): Objeto de configuração contendo os parâmetros de ingestão.
            log_domain (str): Domínio para o logger.
        """
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.config = config.model_copy(deep=True)
        self.logger.info("Inicializando o TextChunker.", config_data=config.model_dump())
        self.chunker = self._create_chunker(self.config)

    def update_config(self, new_config: AppConfig) -> None:
        """
        Atualiza a configuração do TextChunker com base na configuração fornecida.

        Args:
            new_config (AppConfig): A nova configuração a ser aplicada.
        """
        if new_config == self.config:
            self.logger.info("Nenhuma alteracao na configuracao detectada")
            return
        
        if new_config.ingestion.chunking_strategy != self.config.ingestion.chunking_strategy:
            self.chunker = self._create_chunker(new_config)
            self.logger.info(f"Estrategia de chunking alterada para: {new_config.ingestion.chunking_strategy}")

        else:
            self.chunker.update_config(new_config)
            self.logger.info(f"Parametros de configuração de chunking alterados")

        self.config = new_config.model_copy(deep=True)
        self.logger.info("Configuracoes do TextChunker atualizadas com sucesso")


    def _create_chunker(self, new_config: AppConfig) -> ChunkingStrategy:
        """
        Carrega o splitter com base na configuração atual.

        Returns:
            RecursiveCharacterTextSplitter: O splitter carregado. 
            # Por enquanto o único suportado; alterar a assinatura quando adicionar outras estratégias
        """
        chunker = None
        try:
            match new_config.ingestion.chunking_strategy:
                case "recursive":
                    chunker = RecursiveStrategy(new_config, log_domain=self.logger.log_domain)

                case "semantic-cluster":
                    chunker = SemanticClusterStrategy(new_config, log_domain=self.logger.log_domain)

                case _:
                    raise ValueError(f"Estrategia de chunking nao suportada: {new_config.ingestion.chunking_strategy}")
                
            return chunker

        except Exception as e:
            self.logger.error(f"Erro ao carregar o splitter: {e}", exc_info=True)
            raise e
    
    def create_chunks(self, file: DocumentFile) -> List[Chunk]:
        return self.chunker.create_chunks(file)


