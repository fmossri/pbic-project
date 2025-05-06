from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.models import Chunk
from src.config.models import IngestionConfig
from src.utils.logger import get_logger

class ChunkingStrategy(ABC):
    def __init__(self, config: IngestionConfig, log_domain: str):
        self.config = config
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.debug(f"Inicializando a estratégia de chunking: {self.__class__.__name__}")
        
    @abstractmethod
    def create_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        pass

    @abstractmethod
    def update_config(self, new_config: IngestionConfig) -> None:
        pass