import os
from typing import Dict, List, Optional
from langchain.schema import Document

from .document_processor import DocumentProcessor
from .text_chunker import TextChunker
from .text_normalizer import TextNormalizer

class DataIngestionOrchestrator:
    """Componente principal para gerenciar o processamento de arquivos PDF."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Inicializa o orquestrador com os componentes necessários.
        
        Args:
            chunk_size (int): Tamanho de cada chunk de texto
            overlap (int): Sobreposição entre chunks
        """
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker(chunk_size, overlap)
        self.text_normalizer = TextNormalizer()
        self.document_hashes: Dict[str, str] = {}

    def _find_original_document(self, duplicate_hash: str) -> Optional[str]:
        """Encontra o documento original do hash duplicado.
        
        Args:
            duplicate_hash (str): Hash do documento duplicado
            
        Returns:
            Optional[str]: Nome do arquivo original ou None se não encontrado
        """
        for filename, hash_value in self.document_hashes.items():
            if hash_value == duplicate_hash:
                return filename
        return None

    def _is_duplicate(self, document_hash: str) -> bool:
        """
        Verifica se um documento é duplicado.
        
        Args:
            document_hash (str): Hash do documento a ser verificado
            
        Returns:
            bool: True se o documento for duplicado, False caso contrário
        """
        # Verifica se já existe um documento com o mesmo hash
        for existing_hash in self.document_hashes.values():
            if existing_hash == document_hash:
                
                """print(f"\nDocumento duplicado encontrado:")
                print(f"- Arquivo: {filename}")
                print(f"- Hash: {document_hash}")
                print(f"- Original: {existing_filename}")
                print("-" * 50)"""
                return True
        return False
    
    def list_pdf_files(self, directory_path: str) -> List[str]:
        """
        Lista todos os arquivos PDF em um diretório.

        Args:
            directory_path (str): Caminho para o diretório

        Returns:
            List[str]: Lista de nomes de arquivos PDF

        Raises:
            FileNotFoundError: Se o diretório não existir
            NotADirectoryError: Se o caminho não for um diretório
            ValueError: Se nenhum arquivo PDF for encontrado
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")

        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Caminho não é um diretório: {directory_path}")

        pdf_files = []
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith('.pdf'):
                    pdf_files.append(entry.name)

        if not pdf_files:
            raise ValueError(f"Nenhum arquivo PDF encontrado em: {directory_path}")

        return pdf_files
    
    def process_directory(self, directory_path: str) -> Dict[str, List[Document]]:
        """
        Processa todos os arquivos PDF em um diretório.
        
        Args:
            directory_path (str): Caminho para o diretório contendo PDFs
            
        Returns:
            Dict[str, List[Document]]: Dicionário mapeando nomes de arquivos para seus chunks de texto
            
        Raises:
            FileNotFoundError: Se o diretório não existir
            NotADirectoryError: Se o caminho não for um diretório
            ValueError: Se não houver arquivos PDF no diretório
        """
        pdf_files = self.list_pdf_files(directory_path)
        results = {}

        for filename in pdf_files:
            document_path = os.path.join(directory_path, filename)
            
            # Extrai o texto do PDF
            pages = self.document_processor.extract_text(document_path)
            text_content = "\n".join(page.page_content for page in pages)
            
            # Calcula o hash do documento
            document_hash = self.document_processor.calculate_hash(text_content)
            
            # Check for duplicates
            if self._is_duplicate(document_hash):
            #TODO:
            #adiciona a duplicata em um arquivo de texto log. O log deve conter: caminho/nome do arquivo original - seu hash:\n Lista de arquivos duplicados 
            # soma 1 a um contador de duplicatas
                continue
            
            self.document_hashes[filename] = document_hash
            
            #TODO:
            # Decidir se os chunks não normalizados devem ser preservados
            if pages:
                all_chunks = []
                
                # Processa cada página e coleta seus chunks
                for page in pages:
                    # Normaliza o conteúdo da página
                    normalized_page = self.text_normalizer.normalize(page.page_content)

                    # Divide a página em chunks
                    chunks = self.text_chunker.chunk_text(
                        text=normalized_page,
                        metadata={
                            "page": page.metadata["page"],
                            "filename": filename
                        }
                    )
                    if chunks:
                        all_chunks.extend(chunks)
                
                # Adiciona chunks aos resultados se houver algum
                if all_chunks:
                    results[filename] = all_chunks
        
        return results 