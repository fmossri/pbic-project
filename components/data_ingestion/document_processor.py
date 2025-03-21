import hashlib
import os
from typing import List, Tuple
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from pypdf.errors import PdfStreamError
from langchain.schema import Document

class DocumentProcessor:
    """Processa documentos PDF para extração de texto."""
    
    def calculate_hash(self, text_content: str) -> str:
        """
        Calcula o hash MD5 do conteúdo textual de um arquivo PDF.
        
        Args:
            text_content (str): Conteúdo do documento
            
        Returns:
            str: Hash MD5 do conteúdo textual
            
        Raises:
            FileNotFoundError: Se o arquivo não existir
            PdfStreamError: Se o arquivo não for um PDF válido
        """

        # Calcula o hash do texto
        hash_md5 = hashlib.md5()
        hash_md5.update(text_content.encode('utf-8'))
        return hash_md5.hexdigest()
    
    def extract_text(self, file_path: str) -> List[Document]:
        """
        Extrai texto de um documento PDF, página por página.
        
        Args:
            file_path (str): Caminho para o arquivo PDF
            
        Returns:
            List[Document]: Lista de documentos LangChain, cada um contendo o texto de uma página
            
        Raises:
            FileNotFoundError: Se o arquivo não existir
            PdfStreamError: Se o arquivo não for um PDF válido
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
        try:
            loader = PyPDFLoader(file_path)
            return loader.load_and_split()
        except Exception as e:
            raise PdfStreamError(f"Erro ao processar PDF: {str(e)}") 