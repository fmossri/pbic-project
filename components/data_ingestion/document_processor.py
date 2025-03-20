import hashlib
import os
from typing import List, Tuple
from pypdf import PdfReader
from pypdf.errors import PdfStreamError

class DocumentProcessor:
    """Processa documentos PDF para extração de texto."""
    
    def calculate_hash(self, file_path: str) -> str:
        """
        Calcula o hash MD5 do conteúdo textual de um arquivo PDF.
        
        Args:
            file_path (str): Caminho para o arquivo
            
        Returns:
            str: Hash MD5 do conteúdo textual
            
        Raises:
            FileNotFoundError: Se o arquivo não existir
            PdfStreamError: Se o arquivo não for um PDF válido
        """
        # Extrai o texto do PDF
        pages = self.extract_text(file_path)
        
        # Concatena todo o texto em uma única string
        text_content = "\n".join(text for _, text in pages)
        
        # Calcula o hash do texto
        hash_md5 = hashlib.md5()
        hash_md5.update(text_content.encode('utf-8'))
        return hash_md5.hexdigest()
    
    def extract_text(self, file_path: str) -> List[Tuple[int, str]]:
        """
        Extrai texto de um documento PDF, página por página.
        
        Args:
            file_path (str): Caminho para o arquivo PDF
            
        Returns:
            List[Tuple[int, str]]: Lista de tuplas (número_da_página, texto)
            
        Raises:
            FileNotFoundError: Se o arquivo não existir
            PdfStreamError: Se o arquivo não for um PDF válido
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
        reader = PdfReader(file_path)
        pages = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                pages.append((page_num, text))
        
        return pages 