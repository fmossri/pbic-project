import sqlite3
import os
from typing import List, Optional
from src.models import DocumentFile, Chunk, Embedding
from src.utils.logger import get_logger

class SQLiteManager:
    """Gerenciador de banco de dados SQLite."""

    DEFAULT_DB_PATH: str = os.path.join("storage", "domains", "test_domain", "test.db")
    DEFAULT_SCHEMA_PATH: str = os.path.join("storage", "schemas", "schema.sql")

    def __init__(self, 
                 db_path: Optional[str] = DEFAULT_DB_PATH,
                 schema_path: Optional[str] = DEFAULT_SCHEMA_PATH,
                 log_domain: str = "utils"
    ):
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info("Inicializando o SQLiteManager", db_path=db_path, schema_path=schema_path)
        
        self.db_path = db_path if db_path is not None else self.DEFAULT_DB_PATH
        self.schema_path = schema_path if schema_path is not None else self.DEFAULT_SCHEMA_PATH
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def initialize_database(self) -> None:
        self.logger.warning(f"Criando novo banco de dados em {self.db_path}")
        try:
            with open(self.schema_path, "r") as f:
                schema = f.read()

            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema)
                conn.commit()

            self.logger.info(f"Banco de dados inicializado com sucesso")

        except FileNotFoundError as e:
            self.logger.error(f"Erro: Schema nao encontrado em {self.schema_path}: {e}")
            raise FileNotFoundError(f"Arquivo do schema nao encontrado em {self.schema_path}: {e}")
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao inicializar o banco de dados: {e}")
            raise e
                
    def get_connection(self) -> sqlite3.Connection:
        """
        Estabelece uma conexão com o banco de dados. Se não existir, inicializa o banco de dados.
        """

        if not os.path.exists(self.db_path):
            self.logger.info(f"Banco de dados nao encontrado em {self.db_path}. Inicializando o banco de dados...")
            self.initialize_database()
        
        self.logger.info(f"Conectando ao banco de dados em: {self.db_path}")
        return sqlite3.connect(self.db_path)
    
    def begin(self, conn: sqlite3.Connection) -> None:
        """
        Inicia uma transação no banco de dados.
        """
        self.logger.info(f"Iniciando uma transacao com o banco de dados")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN TRANSACTION")
    

    def insert_document_file(self, file: DocumentFile, conn: sqlite3.Connection) -> None:
        """
        Insere um arquivo de documento no banco de dados.
        Args:
            file: objeto DocumentFile a ser inserido.
            conn: Conexão com o banco de dados SQLite.
        """
        self.logger.debug(f"Inserindo objeto DocumentFile: {file.name} no banco de dados: {self.db_path}")
        try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO document_files (name, hash, path, total_pages) VALUES (?, ?, ?, ?)", 
                    (file.name, file.hash, file.path, file.total_pages)
                )
                file.id = cursor.lastrowid

                self.logger.debug(f"Arquivo de documento inserido com sucesso: {file.name}")
                return cursor.lastrowid
        
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao inserir o arquivo de documento: {e}")
            raise e
        
    def insert_chunks(self, chunks: List[Chunk], file_id: int, conn: sqlite3.Connection) -> List[int]:
        """
        Insere um chunk no banco de dados.
        Args:
            chunk: objeto Chunk a ser inserido.
            file_id: ID do documento associado ao chunk.
            conn: Conexão com o banco de dados SQLite.
        """
        self.logger.debug(f"Inserindo objetos Chunk no banco de dados: {self.db_path}")
        chunk_ids : List[int] = []
        for chunk in chunks:

            try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO chunks (document_id, page_number, content, chunk_page_index, chunk_start_char_position) VALUES (?, ?, ?, ?, ?)", 
                        (file_id, chunk.page_number, chunk.content, chunk.chunk_page_index, chunk.chunk_start_char_position)
                    )
                    chunk.id = cursor.lastrowid

                    chunk_ids.append(chunk.id)
            
            except sqlite3.Error as e:
                self.logger.error(f"Erro ao inserir chunks: {e}")
                raise e
        
        self.logger.info(f"Chunks inseridos com sucesso: {chunk_ids}")
        return chunk_ids
        
    def insert_embeddings(self, embeddings: List[Embedding], conn: sqlite3.Connection) -> None:
        """
        Insere um embedding no banco de dados.
        Args:
            embedding: objetoEmbedding a ser inserido.
            conn: Conexão com o banco de dados SQLite.
        """
        self.logger.debug(f"Inserindo objetos Embedding no banco de dados: {self.db_path}")
        for embedding in embeddings:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO embeddings (chunk_id, faiss_index_path, chunk_faiss_index, dimension) VALUES (?, ?, ?, ?)", 
                    (embedding.chunk_id, embedding.faiss_index_path, embedding.chunk_faiss_index, embedding.dimension)
                )
                embedding.id = cursor.lastrowid
            
            except sqlite3.Error as e:
                self.logger.error(f"Erro ao inserir embeddings: {e}")
                raise e
    
    def get_embeddings_chunks(self, conn: sqlite3.Connection, faiss_indices: List[int]) -> List[str]:
        """
        Retorna o conteúdo dos chunks associados aos índices faiss fornecidos.

        Args:
            conn: Conexão com o banco de dados SQLite.
            faiss_indices: Lista de índices faiss correspondentes aos chunks.

        Returns:
            chunks_content: List[str], onde cada string contém o conteúdo de um chunk, na ordem dos índices faiss.
        """
        self.logger.debug(f"Recuperando chunks do banco de dados", faiss_indices=faiss_indices)
        try:
            cursor = conn.cursor()

            order_cases = " ".join([f"WHEN {index} THEN {i}" for i, index in enumerate(faiss_indices)])
            placeholders = ", ".join(['?'] * len(faiss_indices))
            query = f"""
                SELECT chunks.content 
                FROM chunks 
                JOIN embeddings ON chunk_id = chunks.id
                WHERE embeddings.chunk_faiss_index IN ({placeholders})
                ORDER BY CASE embeddings.chunk_faiss_index
                    {order_cases}
                END
            """
            
            cursor.execute(query, faiss_indices)
            chunks_content : List[str] = [row[0] for row in cursor.fetchall()]

            self.logger.debug(f"Chunks recuperados com sucesso: {chunks_content}")
            return chunks_content
        
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao recuperar os chunks: {e}")
            raise e
            
    
    
    