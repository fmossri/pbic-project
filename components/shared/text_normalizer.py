import unicodedata
import re

class TextNormalizer:
    """Text normalization for RAG system."""
    
    def __init__(self):
        print("Inicializando o normalizador de texto...")

    def normalize(self, text: str) -> str:
        """Apply all normalization steps."""
        text = self._normalize_unicode(text)
        text = self._normalize_whitespace(text)
        text = self._normalize_case(text)
        return text
        
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters using NFKC."""
        return unicodedata.normalize('NFKC', text)
        
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace - convert any sequence of whitespace characters to a single space."""
        # Replace any sequence of whitespace with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        return text.strip()
    
    def _normalize_case(self, text: str) -> str:
        """Normalize case."""
        return text.lower()
