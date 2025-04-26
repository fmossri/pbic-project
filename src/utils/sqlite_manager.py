import sqlite3
import os
from typing import List, Optional
from src.models import DocumentFile, Chunk, Domain
from src.utils.logger import get_logger

class SQLiteManager:
    """Gerenciador de banco de dados SQLite."""

    DEFAULT_DB_PATH: str = os.path.join("storage", "domains", "test_domain", "test.db")
    CONTROL_SCHEMA_PATH: str = os.path.join("storage", "schemas", "control_schema.sql")
    CONTROL_DB_PATH: str = os.path.join("storage", "domains", "control.db")
    DEFAULT_SCHEMA_PATH: str = os.path.join("storage", "schemas", "schema.sql")

    def __init__(self, log_domain: str = "utils"):
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info("Inicializando o SQLiteManager")
        self.control_db_path = self.CONTROL_DB_PATH
        self.db_path = self.DEFAULT_DB_PATH
        self.schema_path = self.DEFAULT_SCHEMA_PATH
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def initialize_database(self, control: bool = False, db_path: str = None) -> None:
        self.logger.warning(f"Criando novo banco de dados em {db_path}")
        if control:
            self.db_path = self.control_db_path
            self.schema_path = self.CONTROL_SCHEMA_PATH
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
                
    def get_connection(self, control: bool = False, db_path: str = None) -> sqlite3.Connection:
        """
        Estabelece uma conexão com o banco de dados. Se não existir, inicializa o banco de dados.
        """
        if control:
            if not os.path.exists(self.control_db_path):
                self.logger.info(f"Banco de dados de controle nao encontrado em {self.control_db_path}. Inicializando o banco de dados de controle...")
                self.initialize_database(self.control_db_path)
            self.logger.info(f"Conectando ao banco de dados de controle em: {self.control_db_path}")
            return sqlite3.connect(self.control_db_path)

        self.db_path = db_path if db_path is not None else self.db_path
        if not os.path.exists(self.db_path):
            self.logger.info(f"Banco de dados nao encontrado em {self.db_path}. Inicializando o banco de dados...")
            self.initialize_database(self.db_path)
        
        self.logger.info(f"Conectando ao banco de dados em: {self.db_path}")
        return sqlite3.connect(self.db_path)
    
    def begin(self, conn: sqlite3.Connection) -> None:
        """
        Inicia uma transação no banco de dados.
        """
        self.logger.info(f"Iniciando uma transacao com o banco de dados")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN TRANSACTION")
    
    def insert_domain(self, domain: Domain, conn: sqlite3.Connection) -> None:
        """
        Insere um domínio de conhecimento no banco de dados.
        """
        self.logger.info(f"Inserindo domínio de conhecimento no banco de dados: {domain.domain_name}")
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO domains (name, description, keywords, total_documents, db_path, vector_store_path, faiss_index, embeddings_dimension) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (domain.name, domain.description, domain.keywords, domain.total_documents, domain.db_path, domain.vector_store_path, domain.faiss_index, domain.embeddings_dimension)
            )

            self.logger.debug("Domínio do conhecimento inserido com sucesso", 
                              domain_name=domain.name, 
                              domain_description=domain.description, 
                              domain_keywords=domain.keywords, 
                              domain_db_path=domain.db_path, 
                              domain_vector_store_path=domain.vector_store_path,
                              domain_faiss_index=domain.faiss_index)
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao inserir o domínio de conhecimento: {e}")
            raise e
            
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
        
    def insert_chunks(self, chunks: List[Chunk], file_id: int, conn: sqlite3.Connection) -> None:
        """
        Insere um chunk no banco de dados.
        Args:
            chunk: objeto Chunk a ser inserido.
            file_id: ID do documento associado ao chunk.
            conn: Conexão com o banco de dados SQLite.
        """
        self.logger.debug(f"Inserindo objetos Chunk no banco de dados: {self.db_path}")
        for chunk in chunks:

            try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO chunks (document_id, page_number, content, chunk_page_index, faiss_index, chunk_start_char_position) VALUES (?, ?, ?, ?, ?, ?)", 
                        (file_id, chunk.page_number, chunk.content, chunk.chunk_page_index, chunk.faiss_index, chunk.chunk_start_char_position)
                    )
                    chunk.id = cursor.lastrowid
            
            except sqlite3.Error as e:
                self.logger.error(f"Erro ao inserir chunks: {e}")
                raise e
        
        self.logger.info(f"{len(chunks)} Chunks inseridos com sucesso")
         
    def get_chunks_content(self, conn: sqlite3.Connection, faiss_indices: List[int]) -> List[str]:
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
                WHERE chunks.faiss_index IN ({placeholders})
                ORDER BY CASE chunks.faiss_index
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
            
    def get_domain(self, conn: sqlite3.Connection, domain_name: str) -> Optional[Domain]:
        """
        Retorna um domínio de conhecimento do banco de dados.
        """
        self.logger.debug(f"Recuperando domínio de conhecimento do banco de dados: {domain_name}")
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM domains WHERE name = ?", (domain_name,))
            domain_data = cursor.fetchone()
            if domain_data:
                return Domain(
                    id=domain_data[0],
                    name=domain_data[1],
                    description=domain_data[2],
                    keywords=domain_data[3],
                    total_documents=domain_data[4],
                    db_path=domain_data[5],
                    vector_store_path=domain_data[6],
                    faiss_index=domain_data[7],
                    embeddings_dimension=domain_data[8]
                )
            
            return None
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao recuperar o domínio de conhecimento: {e}")
            raise e
    
    
    