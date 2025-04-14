import os
import sqlite3
from typing import Dict, List, Optional

from src.models import DocumentFile, Chunk, Embedding
from src.utils import TextNormalizer, EmbeddingGenerator, FaissManager, SQLiteManager
from src.utils.logger import get_logger
from .document_processor import DocumentProcessor
from .text_chunker import TextChunker

class DataIngestionOrchestrator:
    """Componente principal para gerenciar o processamento de arquivos PDF."""

    DEFAULT_LOG_DOMAIN = "Ingestão de dados"
    
    def __init__(self, db_path: str = None, index_path: str = None, chunk_size: int = 800, overlap: int = 160):
        """
        Inicializa o orquestrador com os componentes necessários.
        
        Args:
            db_path (str, optional): Caminho para o arquivo de banco de dados SQLite.
            index_path (str, optional): Caminho para o diretório de índices FAISS.
            chunk_size (int): Tamanho de cada chunk de texto.
            overlap (int): Sobreposição entre chunks.
        """
        self.logger = get_logger(__name__, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.logger.info("Inicializando o DataIngestionOrchestrator", 
                        db_path=db_path, 
                        index_path=index_path,
                        chunk_size=chunk_size,
                        overlap=overlap)
        
        self.document_processor = DocumentProcessor(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.text_chunker = TextChunker(chunk_size, overlap, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.text_normalizer = TextNormalizer(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.embedding_generator = EmbeddingGenerator(log_domain=self.DEFAULT_LOG_DOMAIN)
        self.sqlite_manager = SQLiteManager(db_path=db_path, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.faiss_manager = FaissManager(index_path=index_path, log_domain=self.DEFAULT_LOG_DOMAIN)
        self.document_hashes: Dict[str, str] = {}
        
    def _find_original_document(self, duplicate_hash: str, conn: sqlite3.Connection) -> DocumentFile:
        """
        Encontra o documento original do hash duplicado.
        
        Args:
            duplicate_hash (str): Hash do documento duplicado
            conn (sqlite3.Connection): Conexão com o banco de dados
            
        Returns:
            DocumentFile: Documento original encontrado
            
        Raises:
            ValueError: Se o documento original não for encontrado
        """
        self.logger.info(f"Procurando o documento original com o hash: {duplicate_hash}")
        # Primeiro procura no dicionário de hashes
        try:
            for filename, hash_value in self.document_hashes.items():
                if hash_value == duplicate_hash:
                    # Cria um DocumentFile com as informações básicas
                    return DocumentFile(
                        id=None,
                        hash=hash_value,
                        name=filename,
                        path=os.path.join(os.path.dirname(filename), filename),
                        total_pages=0
                    )
            
        # Se não encontrou no dicionário, procura no banco de dados
            cursor = conn.execute("SELECT * FROM document_files WHERE hash = ?", (duplicate_hash,)) 
            result = cursor.fetchone()
            if result:
                return DocumentFile(
                    id=result[0],
                    hash=result[1],
                    name=result[2],
                    path=result[3],
                    total_pages=result[4]
                )
        except Exception as e:
            self.logger.error(f"Erro ao encontrar o documento original: {e}")
            raise ValueError(f"Documento original não encontrado para o hash: {duplicate_hash}")

    def _is_duplicate(self, document_hash: str, conn: sqlite3.Connection) -> bool:
        """
        Verifica se um documento é duplicado.
        
        Args:
            document_hash (str): Hash do documento a ser verificado
            
        Returns:
            bool: True se o documento for duplicado, False caso contrário
        """
        self.logger.debug(f"Verificando se o documento é duplicado: {document_hash}")    
        try:
            # Verifica se já existe um documento com o mesmo hash
            for existing_hash in self.document_hashes.values():
                if existing_hash == document_hash:
                    self.logger.warning(f"Documento duplicado encontrado")
                    return True
            
            cursor = conn.execute("SELECT * FROM document_files WHERE hash = ?", (document_hash,))
            result = cursor.fetchone()
            if result:
                self.logger.warning(f"Documento duplicado encontrado")
                return True
            return False
        
        except Exception as e:
            self.logger.error(f"Erro ao verificar se o documento e duplicado: {e}")
            raise e
        
        

    
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
        self.logger.info("Listando arquivos PDF no diretorio", directory_path=directory_path)

        if not os.path.exists(directory_path):
            self.logger.error("Diretorio nao encontrado", directory_path=directory_path)
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")

        if not os.path.isdir(directory_path):
            self.logger.error("Caminho nao e um diretorio", directory_path=directory_path)
            raise NotADirectoryError(f"Caminho não é um diretório: {directory_path}")

        pdf_files = []
        try:
            with os.scandir(directory_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith('.pdf'):
                        file = DocumentFile(id = None, hash = None, name=entry.name, path=entry.path, total_pages=0)
                        pdf_files.append(file)
        except Exception as e:
            self.logger.error(f"Erro ao listar arquivos PDF: {e}")
            raise e

        if not pdf_files:
            self.logger.error("Nenhum arquivo PDF encontrado no diretorio", directory_path=directory_path)
            raise ValueError(f"Nenhum arquivo PDF encontrado em: {directory_path}")
        
        self.logger.info(f"{len(pdf_files)} arquivos PDF encontrados no diretorio", directory_path=directory_path)
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
            self.logger.error("Diretorio nao encontrado", directory_path=directory_path)
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")
        if not os.path.isdir(directory_path):
            self.logger.error("Caminho nao e um diretorio", directory_path=directory_path)
            raise NotADirectoryError(f"Caminho não é um diretório: {directory_path}")
        
        self.logger.info("Iniciando o processamento do diretorio", directory_path=directory_path)

        pdf_files = self.list_pdf_files(directory_path)

        with self.sqlite_manager.get_connection() as conn:
            self.sqlite_manager.begin(conn)

            self.logger.info(f"Iniciando o processamento dos arquivos PDF")

            file_counter = 0
            duplicate_counter = 0
            for file in pdf_files:
                file_counter += 1
                self.logger.info(f"Processando arquivos: {file_counter}/{len(pdf_files)}", file_path=file.path)
                
                try:
                    # processa o documento, adicionando as páginas, hash e total de páginas ao objeto DocumentFile
                    self.document_processor.process_document(file)
                    # Verifica se é duplicado
                    if self._is_duplicate(file.hash, conn):
                        duplicate_counter += 1
                        original_file = self._find_original_document(file.hash, conn)
                        if original_file:
                            self.logger.warning(f"Arquivo duplicado: {file.name} hash: {file.hash}", file_path=file.path)
                            self.logger.warning(f"Arquivo original: {original_file.name} hash: {original_file.hash}", file_path=original_file.path)

                        self.logger.info(f"Descartando alteracoes da transacao", file_path=file.path)
                        conn.rollback()
                        continue

                    # Adiciona o hash do documento ao dicionário de hashes
                    self.document_hashes[file.name] = file.hash
                    # Insere o documento no banco de dados
                    file.id = self.sqlite_manager.insert_document_file(file, conn)
  
                    if not file.pages:
                        self.logger.error(f"{file.name} - Nenhuma pagina encontrada", file_path=file.path)
                        self.logger.error(f"Descartando alteracoes da transacao", file_path=file.path)
                        conn.rollback()
                        continue

                    # Processa cada página e coleta seus chunks
                    self.logger.info(f"{file.name} - Total de paginas: {len(file.pages)}", file_path=file.path)
                    self.logger.info(f"{file.name} - Iniciando a criacao de chunks", file_path=file.path)
                    document_chunks : List[Chunk] = []
                    page_counter = 0
                    for page in file.pages:
                        # Divide a página em chunks
                        page_counter += 1
                        self.logger.debug(f"{file.name} - Processando pagina: {page_counter}/{len(file.pages)}", file_path=file.path)

                        page_chunks = self.text_chunker.create_chunks(
                            text=page.page_content,
                            metadata={
                                    "page_number": page.metadata["page"],
                                    "document_id": file.id,
                                }
                            )

                        if not page_chunks:
                            self.logger.error(f"{file.name} - Pagina vazia encontrada: {page.metadata['page']}", file_path=file.path)
                            continue
                        
                        document_chunks.extend(page_chunks)
                    
                    if not document_chunks:
                        self.logger.error(f"{file.name} - Nenhum chunk gerado", file_path=file.path)
                        self.logger.error(f"Descartando alteracoes da transacao", file_path=file.path)
                        conn.rollback()
                        continue

                    self.logger.info(f"{file.name} - Total de chunks: {len(document_chunks)}", file_path=file.path)

                    # Insere os chunks no banco de dados e retorna os ids dos chunks
                    chunk_ids : List[int] = self.sqlite_manager.insert_chunks(document_chunks, file.id, conn)

                    # Normaliza os chunks
                    chunks_content = [chunk.content for chunk in document_chunks]
                    normalized_chunks = self.text_normalizer.normalize(chunks_content)

                    #Gera os embeddings
                    embedding_vectors = self.embedding_generator.generate_embeddings(normalized_chunks)                    
                    if embedding_vectors.size == 0:
                        self.logger.error(f"{file.name} - Nenhum embedding gerado", file_path=file.path)
                        self.logger.error(f"Descartando alteracoes da transacao", file_path=file.path)
                        conn.rollback()
                        continue

                    #Cria os objetos Embedding
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
                    self.logger.info(f"{file.name} - Salvando alteracoes no banco de dados", file_path=file.path)
                    conn.commit()
                    self.logger.info(f"{file.name} - Processamento concluido com sucesso", file_path=file.path)
                except Exception as e:
                    self.logger.error(f"{file.name} - Erro ao processar o arquivo: {e}", file_path=file.path)
                    self.logger.error(f"Descartando alteracoes da transacao", file_path=file.path)
                    conn.rollback()
                    continue

 