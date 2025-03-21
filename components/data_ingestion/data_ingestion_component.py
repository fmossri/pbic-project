import os
from typing import Dict, List, Optional, Tuple
import hashlib

from .document_processor import DocumentProcessor
from .text_chunker import TextChunker


class DataIngestionComponent:
    """Componente principal para gerenciar a ingestão de dados PDF."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Inicializa o Componente de Ingestão de Dados.
        
        Args:
            chunk_size (int): Tamanho de cada chunk de texto
            overlap (int): Sobreposição entre chunks
        """
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker(chunk_size, overlap)
        self.document_hashes: Dict[str, str] = {}
        self.document_sizes: Dict[str, int] = {}

    def _find_original_document(self, duplicate_hash: str) -> Optional[str]:
        """Encontra o documento original para um hash duplicado.
        
        Args:
            duplicate_hash (str): Hash do documento duplicado
            
        Returns:
            Optional[str]: Nome do documento original ou None se não encontrado
        """
        for filename, hash_value in self.document_hashes.items():
            if hash_value == duplicate_hash:
                return filename
        return None

    def _get_file_size(self, file_path: str) -> int:
        """Obtém o tamanho do arquivo em bytes.
        
        Args:
            file_path (str): Caminho para o arquivo
            
        Returns:
            int: Tamanho do arquivo em bytes
        """
        return os.path.getsize(file_path)
    
    def _is_duplicate(self, document_hash: str) -> bool:
        """
        Verifica se um documento é duplicado e imprime informações se for.
        
        Args:
            filename (str): Nome do arquivo a ser verificado
            document_path (str): Caminho completo do arquivo
            
        Returns:
            bool: True se o documento for duplicado, False caso contrário
        """
        # Verifica se já existe um documento com o mesmo hash
        for existing_filename, existing_hash in self.document_hashes.items():
            if existing_hash == document_hash:
                #original_size = self.document_sizes[existing_filename]
                #size_diff = abs(self._get_file_size(document_path) - original_size)
                
                """print(f"\nDocumento duplicado encontrado:")
                print(f"- Arquivo: {filename}")
                print(f"- Hash: {document_hash}")
                print(f"- Tamanho: {self._get_file_size(document_path):,} bytes")
                print(f"- Original: {existing_filename}")
                print(f"- Tamanho original: {original_size:,} bytes")
                print(f"- Diferença de tamanho: {size_diff:,} bytes")
                print("-" * 50)"""
                return True
        
        return False
    
    def list_pdf_files(self, directory_path: str) -> List[str]:
        """Lista todos os arquivos PDF em um diretório.
        
        Args:
            directory_path (str): Caminho para o diretório
            
        Returns:
            List[str]: Lista de nomes dos arquivos PDF encontrados
            
        Raises:
            FileNotFoundError: Se o diretório não existir
            NotADirectoryError: Se o caminho não for um diretório
            ValueError: Se não houver arquivos PDF no diretório
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")
            
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Caminho não é um diretório: {directory_path}")
        
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            raise ValueError(f"Nenhum arquivo PDF encontrado em: {directory_path}")
            
        return pdf_files
    
    def process_directory(self, directory_path: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Processa todos os arquivos PDF em um diretório.
        
        Args:
            directory_path (str): Caminho para o diretório contendo PDFs
            
        Returns:
            Dict[str, List[Dict[str, str]]]: Dicionário mapeando nomes de arquivos para seus chunks de texto
            
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
            text_content = "\n".join(text for _, text in pages)
            
            # Calcula o hash do documento
            document_hash = self.document_processor.calculate_hash(text_content)
            
            # Check for duplicates
            if self._is_duplicate(document_hash):
            #TODO:
            #adiciona a duplicata em um arquivo de texto log. O log deve conter: caminho/nome do arquivo original - seu hash - seu tamanho:\n Lista de arquivos duplicados 
            # soma 1 a um contador de duplicatas
                continue
            
            self.document_hashes[filename] = document_hash
            self.document_sizes[filename] = self._get_file_size(document_path)
            
            # Process chunks...
            if pages:
                all_chunks = []
                
                # Processa cada página e coleta seus chunks
                for page_num, page_text in pages:
                    chunks = self.text_chunker.chunk_text(
                        text=page_text,
                        metadata={
                            "page": page_num,
                            "filename": filename,
                            "has_context": False
                        }
                    )
                    if chunks:
                        all_chunks.extend(chunks)
                
                # Adiciona chunks aos resultados se houver algum
                if all_chunks:
                    results[filename] = all_chunks
        
        return results 