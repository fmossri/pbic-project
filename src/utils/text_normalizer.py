import unicodedata
import re
from src.utils.logger import get_logger
from src.config.models import TextNormalizerConfig

class TextNormalizer:
    """Normalização de texto para o sistema RAG, configurável via TextNormalizerConfig."""

    def __init__(self, config: TextNormalizerConfig, log_domain: str = "utils"):
        """Inicializa o TextNormalizer."""
        self.config = config
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info("Inicializando o TextNormalizer", config=config.model_dump())

    def normalize(self, texts: list[str] | str) -> list[str]:
        """Applica etapas de normalização configuradas ao texto(s) de entrada."""
        self.logger.debug("Iniciando a normalizacao do texto")

        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            self.logger.error("O texto de entrada deve ser uma string ou uma lista de strings")
            raise TypeError("O texto de entrada deve ser uma string ou uma lista de strings")

        normalized_texts: list[str] = []
        try:
            for text in texts:
                if not isinstance(text, str):
                     self.logger.error(f"Elemento nao-string na lista de entrada: {type(text)}", text=text)
                     raise TypeError(f"O texto de entrada deve ser uma string ou uma lista de strings")

                normalized_text = text
                if self.config.use_unicode_normalization:
                    normalized_text = self._normalize_unicode(normalized_text)
                if self.config.use_remove_extra_whitespace:
                    normalized_text = self._normalize_whitespace(normalized_text)
                if self.config.use_lowercase:
                    normalized_text = self._normalize_case(normalized_text)
                # Note: We don't have a separate remove_special_chars step here yet.
                # Add it if needed based on config.

                normalized_texts.append(normalized_text)
            self.logger.debug(f"{len(normalized_texts)} segmentos de texto normalizados com sucesso")
            return normalized_texts
        except Exception as e:
            self.logger.error(f"Erro durante a normalizacao do texto: {e}", exc_info=True)
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
