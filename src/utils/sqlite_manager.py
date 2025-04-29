import sqlite3
import os
from typing import List, Optional, Dict, Any
from src.models import DocumentFile, Chunk, Domain
from src.utils.logger import get_logger
import datetime

class SQLiteManager:
    """Gerenciador de banco de dados SQLite."""

    TEST_DB_PATH: str = os.path.join("storage", "domains", "test_domain", "test.db")
    CONTROL_SCHEMA_PATH: str = os.path.join("storage", "schemas", "control_schema.sql")
    CONTROL_DB_PATH: str = os.path.join("storage", "domains", "control.db")
    DOMAIN_SCHEMA_PATH: str = os.path.join("storage", "schemas", "schema.sql")

    def __init__(self, log_domain: str = "utils"):
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info("Inicializando o SQLiteManager")
        self.control_db_path = self.CONTROL_DB_PATH
        self.db_path = self.TEST_DB_PATH
        self.schema_path = self.DOMAIN_SCHEMA_PATH
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def create_database(self, control: bool = False, db_path: str = None, schema_path: str = None) -> None:
        # Determina o banco de dados e o schema a ser utilizado
        if control:
            # Seleciona o banco de dados de controle
            path_to_connect = self.control_db_path
            final_schema_path = schema_path if schema_path is not None else self.CONTROL_SCHEMA_PATH
        else:
            # Seleciona o banco de dados de dominio
            if db_path is None:
                # Fallback to instance db_path if argument is not provided
                self.logger.error("db_path nao pode ser None quando control for False")
                raise ValueError("db_path nao pode ser None quando control for False")
            else:
                path_to_connect = db_path
            final_schema_path = schema_path if schema_path is not None else self.DOMAIN_SCHEMA_PATH
        
        self.logger.warning(f"Tentando criar o banco de dados em {path_to_connect} usando o schema {final_schema_path}")

        try:
            # Read the determined schema path
            with open(final_schema_path, "r") as f:
                schema = f.read()

            self.logger.debug(f"Arquivo schema lido com sucesso", schema_path=final_schema_path)

            # Verifica se o diretorio existe antes de conectar
            if path_to_connect is None:
                raise ValueError("db_path não pode ser None quando control for False")
            db_dir = os.path.dirname(path_to_connect)
            if db_dir and not os.path.exists(db_dir): 
                self.logger.info(f"Diretorio do banco de dados nao existe. Criando: {db_dir}")
                os.makedirs(db_dir, exist_ok=True)

            # Conecta ao banco de dados e executa o schema
            with sqlite3.connect(path_to_connect) as conn:
                conn.executescript(schema)
                conn.commit()

            self.logger.info(f"Banco de dados criado com sucesso em {path_to_connect}")

        except FileNotFoundError as e:
            self.logger.error(f"Erro: Schema nao encontrado em {final_schema_path}: {e}")
            # Re-raise with the path that failed
            raise FileNotFoundError(f"Arquivo do schema nao encontrado em {final_schema_path}: {e}") 
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao criar o banco de dados: {e}")
            raise e
                
    def get_connection(self, control: bool = False, db_path: str = None) -> sqlite3.Connection:
        """
        Estabelece uma conexão com o banco de dados. Se não existir, inicializa o banco de dados.
        """
        if control:
            if not os.path.exists(self.control_db_path):
                self.logger.info(f"Banco de dados de controle nao encontrado em {self.control_db_path}. Inicializando o banco de dados de controle...")
                self.create_database(self.control_db_path)
            self.logger.info(f"Conectando ao banco de dados de controle em: {self.control_db_path}")
            return sqlite3.connect(self.control_db_path)
        
        self.db_path = db_path if db_path is not None else self.db_path

        if not os.path.exists(self.db_path):
            self.logger.info(f"Banco de dados nao encontrado em {self.db_path}. Inicializando o banco de dados...")
            self.create_database(db_path=self.db_path)
        
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

    def get_document_file(self, conn: sqlite3.Connection, file_id: Optional[int] = None) -> Optional[List[DocumentFile]]:
        """
        Recupera um arquivo de documento do banco de dados.
        """
        self.logger.debug(f"Recuperando arquivo de documento do banco de dados: {file_id}")
        try:
            cursor = conn.cursor()
            if file_id:
                cursor.execute("SELECT * FROM document_files WHERE id = ?", (file_id,))
            else:
                cursor.execute("SELECT * FROM document_files")
            
            all_files : List[DocumentFile] = []
            file_data = cursor.fetchall()
            if file_data:
                for row in file_data:
                    file = DocumentFile(
                        id=row[0],
                        name=row[1],
                        hash=row[2],
                        path=row[3],
                        total_pages=row[4]
                    )
                    all_files.append(file)

                return all_files
            else:
                return None

        except sqlite3.Error as e:
            self.logger.error(f"Erro ao recuperar o arquivo de documento: {e}")
            raise e
        
    def update_document_file(self, file: DocumentFile, conn: sqlite3.Connection) -> None:
        """
        Atualiza um arquivo de documento no banco de dados.
        """
        self.logger.debug(f"Atualizando arquivo de documento no banco de dados: {file.name}")
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE document_files SET name = ?, hash = ?, path = ?, total_pages = ? WHERE id = ?", 
                           (file.name, file.hash, file.path, file.total_pages, file.id))
            conn.commit()

        except sqlite3.Error as e:
            self.logger.error(f"Erro ao atualizar o arquivo de documento: {e}")
            raise e
        
    def delete_document_file(self, file: DocumentFile, conn: sqlite3.Connection) -> None:
        """
        Deleta um arquivo de documento do banco de dados.
        """
        self.logger.debug(f"Deletando arquivo de documento do banco de dados: {file.name}")
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_files WHERE id = ?", (file.id,))
            conn.commit()

        except sqlite3.Error as e:
            self.logger.error(f"Erro ao deletar o arquivo de documento: {e}")
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

    def insert_domain(self, domain: Domain, conn: sqlite3.Connection) -> None:
        """
        Insere um domínio de conhecimento no banco de dados de controle.
        """
        self.logger.info(f"Inserindo domínio de conhecimento no banco de dados: {domain.name}")
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO knowledge_domains (name, description, keywords, total_documents, db_path, vector_store_path, embeddings_dimension) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (domain.name, domain.description, domain.keywords, domain.total_documents, domain.db_path, domain.vector_store_path, domain.embeddings_dimension)
            )

            self.logger.debug("Domínio do conhecimento inserido com sucesso", 
                              domain_name=domain.name, 
                              domain_description=domain.description, 
                              domain_keywords=domain.keywords, 
                              domain_db_path=domain.db_path, 
                              domain_vector_store_path=domain.vector_store_path,
                              domain_embeddings_dimension=domain.embeddings_dimension)
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao inserir o domínio de conhecimento: {e}")
            raise e

    def get_domain(self, conn: sqlite3.Connection, domain_name: Optional[str] = None) -> Optional[List[Domain]]:
        """
        Retorna um ou todos os domínios de conhecimento do banco de dados de controle.
        """
        self.logger.debug(f"Recuperando domínio(s) de conhecimento do banco de dados: {domain_name or 'Todos'}")
        try:
            cursor = conn.cursor()
            if domain_name:
                cursor.execute("SELECT * FROM knowledge_domains WHERE name = ?", (domain_name,))
            else:
                cursor.execute("SELECT * FROM knowledge_domains")
            
            all_domains : List[Domain] = []
            domain_data = cursor.fetchall()
            if domain_data:
                for row in domain_data:
                    domain = Domain(
                        id=row[0],
                        name=row[1],
                        description=row[2],
                        keywords=row[3],
                        total_documents=row[4],
                        vector_store_path=row[5],
                        db_path=row[6],
                        embeddings_dimension=row[7],
                        created_at=row[8],
                        updated_at=row[9]
                    )
                    all_domains.append(domain)
                return all_domains
            else:
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao recuperar o(s) domínio(s) de conhecimento: {e}")
            raise e
        
    def update_domain(self, domain: Domain, conn: sqlite3.Connection, update: Dict[str, Any]) -> None:
        """
        Atualiza um domínio de conhecimento no banco de dados de controle.
        """      
        self.logger.debug(f"Atualizando domínio de conhecimento no banco de dados: {domain.name}")

        update["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        set_parts, params = [], []
        for column, value in update.items():
            set_parts.append(f"{column} = ?")
            params.append(value)
        
        if not set_parts:
            self.logger.error(f"Nenhum campo para atualizar")
            raise ValueError("Nenhum campo para atualizar")
        
        set_clause = ", ".join(set_parts)
        query = f"UPDATE knowledge_domains SET {set_clause} WHERE id = ?"
        params.append(domain.id)

        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            self.logger.debug(f"Atualização executada para o domínio. Aguardando commit",)

        except sqlite3.Error as e:
            self.logger.error(f"Erro ao atualizar o domínio de conhecimento: {e}")
            raise e
        
    def delete_domain(self, domain: Domain, conn: sqlite3.Connection) -> None:
        """
        Deleta um domínio de conhecimento do banco de dados de controle.
        """
        self.logger.debug(f"Deletando domínio de conhecimento do banco de dados: {domain.name}")
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_domains WHERE id = ?", (domain.id,))
            self.logger.debug(f"Domínio removido do DB. Aguardando commit", domain_name=domain.name)

        except sqlite3.Error as e:
            self.logger.error(f"Erro ao deletar o domínio de conhecimento: {e}")
            raise e
    