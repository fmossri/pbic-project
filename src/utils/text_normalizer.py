import unicodedata
import re
from src.utils.logger import get_logger

class TextNormalizer:
    """Text normalization for RAG system."""
    
    def __init__(self, log_domain: str = "utils"):
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info("Inicializando o normalizador de texto...")

    def normalize(self, texts: list[str]) -> list[str]:
        """Aplica todas as etapas de normalização."""
        self.logger.info("Iniciando a normalização do texto")

        texts = [texts] if isinstance(texts, str) else texts
        try:
            normalized_texts: list[str] = []
            for text in texts:
                normalized_text = self._normalize_unicode(text)
                normalized_text = self._normalize_whitespace(normalized_text)
                normalized_text = self._normalize_case(normalized_text)
                normalized_texts.append(normalized_text)
            self.logger.info("Texto normalizado com sucesso")
            return normalized_texts
        except Exception as e:
            self.logger.error(f"Erro ao normalizar o texto: {e}")
            raise e
        
    def _normalize_unicode(self, text: str) -> str:
        """Normaliza caracteres Unicode usando NFKC."""
        return unicodedata.normalize('NFKC', text)
        
    def _normalize_whitespace(self, text: str) -> str:
        """Normaliza espaços em branco - converte qualquer sequência de caracteres de espaço em um único espaço."""
        # Substitui qualquer sequência de espaços por um único espaço
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _normalize_case(self, text: str) -> str:
        """Normaliza o caso das letras - converte todas as letras para minúsculas."""
        return text.lower()
