import hashlib
import os

from typing import List
from langchain_community.document_loaders import PyPDFLoader
from pypdf.errors import PdfStreamError
from langchain.schema import Document

from src.models import DocumentFile
from src.utils.logger import get_logger
class DocumentProcessor:
    """Processa documentos PDF para extração de texto."""

    def __init__(self, log_domain: str = "Ingestão de dados"):
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info("Inicializando o DocumentProcessor")
    
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
        self.logger.info("Calculando o hash do documento", hash_function="md5")
        try:
            # Calcula o hash do texto
            hash_function = hashlib.md5()
            hash_function.update(text_content.encode('utf-8'))
            return hash_function.hexdigest()
        except Exception as e:
            self.logger.error("Erro ao calcular o hash do documento", error=str(e))
            raise e
    
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
        self.logger.info(f"Extraindo o texto", file_path=file_path)
        if not os.path.exists(file_path):
            self.logger.error("Arquivo nao encontrado", file_path=file_path)
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
        try:
            self.logger.debug("Carregando o documento com PyPDFLoader")
            loader = PyPDFLoader(file_path)

            pages = loader.load_and_split()
            self.logger.debug(f"Texto extraido com sucesso. {len(pages)} paginas processadas")
            return pages
        
        except Exception as e:
            self.logger.error("Erro ao processar PDF", error=str(e))
            raise PdfStreamError(f"Erro ao processar PDF: {str(e)}") 
        

    def process_document(self, file: DocumentFile) -> None:
        """
        Processa um documento PDF, extraindo texto página por página, calculando o hash do documento e atualizando o objeto DocumentFile.
        
        Args:
            file (DocumentFile): Objeto DocumentFile representando o documento PDF"""
        
        self.logger.info(f"Iniciando o processamento do documento {file.name}", file_path=file.path)

        # Extrai o texto do PDF
        try:
            pages = self._extract_text(file.path)
            file.pages = pages
            file.total_pages = len(pages)

            text_content = "\n".join(page.page_content for page in pages)
            # Calcula o hash do documento
            file.hash = self._calculate_hash(text_content)
            self.logger.debug(f"Hash do documento calculado com sucesso", file_hash=file.hash)
            self.logger.info(f"Documento processado com sucesso")

        except Exception as e:
            self.logger.error(f"Erro ao processar o documento {file.name}", error=str(e))
            raise e


            
        