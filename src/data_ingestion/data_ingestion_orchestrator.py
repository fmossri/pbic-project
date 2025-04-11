import os
import sqlite3
from typing import Dict, List, Optional

from src.models import DocumentFile, Chunk, Embedding
from src.utils import TextNormalizer, EmbeddingGenerator, FaissManager, SQLiteManager
from .document_processor import DocumentProcessor
from .text_chunker import TextChunker

class DataIngestionOrchestrator:
    """Componente principal para gerenciar o processamento de arquivos PDF."""
    
    def __init__(self, db_path: str = None, index_path: str = None, chunk_size: int = 800, overlap: int = 160):
        """
        Inicializa o orquestrador com os componentes necessários.
        
        Args:
            db_path (str, optional): Caminho para o arquivo de banco de dados SQLite.
            index_path (str, optional): Caminho para o diretório de índices FAISS.
            chunk_size (int): Tamanho de cada chunk de texto.
            overlap (int): Sobreposição entre chunks.
        """
        print("Inicializando o orquestrador...")
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker(chunk_size, overlap)
        self.text_normalizer = TextNormalizer()
        self.embedding_generator = EmbeddingGenerator()
        self.sqlite_manager = SQLiteManager(db_path=db_path)
        self.faiss_manager = FaissManager(index_path=index_path)
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

    def _is_duplicate(self, document_hash: str, conn: sqlite3.Connection) -> bool:
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
            
        cursor = conn.execute("SELECT * FROM document_files WHERE hash = ?", (document_hash,))
        result = cursor.fetchone()
        if result:
            return True
        
        return False
        
        

    
    def list_pdf_files(self, directory_path: str) -> List[DocumentFile]:
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
                    file = DocumentFile(id = None, hash = None, name=entry.name, path=entry.path, total_pages=0)
                    pdf_files.append(file)

        if not pdf_files:
            raise ValueError(f"Nenhum arquivo PDF encontrado em: {directory_path}")

        return pdf_files
    
    def process_directory(self, directory_path: str) -> None:
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
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Path is not a directory: {directory_path}")

        pdf_files = self.list_pdf_files(directory_path)

        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {directory_path}")

        with self.sqlite_manager.get_connection() as conn:
            self.sqlite_manager.begin(conn)



            for file in pdf_files:
                
                try:
                    # processa o documento, adicionando as páginas, hash e total de páginas ao objeto DocumentFile
                    self.document_processor.process_document(file)
                    # Verifica se é duplicado
                    if self._is_duplicate(file.hash, conn):
                    #TODO:
                    #adiciona a duplicata em um arquivo de texto log. O log deve conter: caminho/nome do arquivo original - seu hash:\n Lista de arquivos duplicados 
                    # soma 1 a um contador de duplicatas
                        conn.rollback()
                        continue

                    self.document_hashes[file.name] = file.hash
                    file.id = self.sqlite_manager.insert_document_file(file, conn)
  
                    if not file.pages:
                        conn.rollback()
                        continue

                    document_chunks : List[Chunk] = []
                    # Processa cada página e coleta seus chunks
                    for page in file.pages:
                        # Divide a página em chunks
                            page_chunks = self.text_chunker.create_chunks(
                                text=page.page_content,
                                metadata={
                                    "page_number": page.metadata["page"],
                                    "document_id": file.id,
                                }
                            )

                            if page_chunks:
                                document_chunks.extend(page_chunks)
                        

                    if not document_chunks:
                        conn.rollback()
                        continue
                    # Inicializa a lista de chunks normalizados
                    normalized_chunks : List[str] = []
                    chunk_ids : List[int] = self.sqlite_manager.insert_chunks(document_chunks, file.id, conn)
                    for chunk in document_chunks:
                        #Normaliza o chunk e o adiciona à lista de chunks normalizados
                        normalized_chunk = self.text_normalizer.normalize(chunk.content)
                        normalized_chunks.append(normalized_chunk)

                    #Gera os embeddings
                    embedding_vectors = self.embedding_generator.generate_embeddings(normalized_chunks)                    
                    if embedding_vectors.size == 0:
                        conn.rollback()
                        continue

                    embeddings : List[Embedding] = []
                    for embedding_vector, chunk_id in zip(embedding_vectors, chunk_ids):
                        embedding = Embedding(
                            id = None,
                            chunk_id = chunk_id,
                            faiss_index_path = None,
                            chunk_faiss_index = None,
                            dimension = self.embedding_generator.embedding_dimension, 
                            embedding = embedding_vector
                        )

                        embeddings.append(embedding)
                    #Adiciona os embeddings ao índice FAISS, atualizando os objetos Embedding com o seu índice FAISS
                    self.faiss_manager.add_embeddings(embeddings)
                    #Adiciona os embeddings ao banco de dados
                    self.sqlite_manager.insert_embeddings(embeddings, conn)
                    #Salva as alterações no banco de dados
                    conn.commit()
                
                except Exception as e:
                    conn.rollback()
                    print(f"Erro ao processar o arquivo {file.name}: {e}")
                    continue

 