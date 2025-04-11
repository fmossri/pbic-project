import hashlib
import os

from typing import List
from langchain_community.document_loaders import PyPDFLoader
from pypdf.errors import PdfStreamError
from langchain.schema import Document

from src.models import DocumentFile

class DocumentProcessor:
    """Processa documentos PDF para extração de texto."""

    def __init__(self):
        print("Inicializando o processador de documentos...")
    
    def _calculate_hash(self, text_content: str) -> str:
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
        hash_function = hashlib.md5()
        hash_function.update(text_content.encode('utf-8'))
        return hash_function.hexdigest()
    
    def _extract_text(self, file_path: str) -> List[Document]:
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
        

    def process_document(self, file: DocumentFile) -> None:
        """
        Processa um documento PDF, extraindo texto página por página, calculando o hash do documento e atualizando o objeto DocumentFile.
        
        Args:
            file (DocumentFile): Objeto DocumentFile representando o documento PDF"""
        

        # Extrai o texto do PDF
        try:
            pages = self._extract_text(file.path)
            file.pages = pages
            file.total_pages = len(pages)

            text_content = "\n".join(page.page_content for page in pages)
            # Calcula o hash do documento
            file.hash = self._calculate_hash(text_content)

        except Exception as e:
            raise e


            
        