import sqlite3
import os
import json

from typing import List, Optional, Dict, Any

from src.models import DocumentFile, Chunk, Domain, DomainConfig
from src.utils.logger import get_logger
from src.config.models import SystemConfig


class SQLiteManager:
    """Gerenciador de banco de dados SQLite."""

    CONTROL_SCHEMA_PATH: str = os.path.join("storage", "schemas", "control_schema.sql")
    DOMAIN_SCHEMA_PATH: str = os.path.join("storage", "schemas", "schema.sql")

    def __init__(self, config: SystemConfig, log_domain: str = "utils"):
        self.config = config
        self.logger = get_logger(__name__, log_domain=log_domain)
        self.logger.info("Inicializando o SQLiteManager")
        self.control_db_path = os.path.join(config.storage_base_path, config.control_db_filename)
        self.db_path = None
        self.schema_path = self.DOMAIN_SCHEMA_PATH

    def update_config(self, new_config: SystemConfig) -> None:
        """
        Atualiza a configuração do SQLiteManager com base na configuração fornecida.

        Args:
            new_config (SystemConfig): A nova configuração a ser aplicada.
        """

        if new_config.control_db_filename != self.config.control_db_filename:
            self.control_db_path = os.path.join(self.config.storage_base_path, new_config.control_db_filename)

        self.config = new_config
        self.logger.info("Configuracoes do SQLiteManager atualizadas com sucesso")
    

    def _create_database(self, control: bool = False, db_path: str = None, schema_path: str = None) -> None:
        # Determina o banco de dados e o schema a ser utilizado
        if control:
            # Seleciona o banco de dados de controle
            path_to_connect = self.control_db_path
            final_schema_path = schema_path if schema_path is not None else self.CONTROL_SCHEMA_PATH
        else:
            # Seleciona o banco de dados de dominio
            if db_path is None:
                self.logger.error("db_path nao pode ser None quando control for False")
                raise ValueError("db_path nao pode ser None quando control for False")

            path_to_connect = db_path
            final_schema_path = schema_path if schema_path is not None else self.DOMAIN_SCHEMA_PATH
        
        self.logger.warning(f"Tentando criar o banco de dados em {path_to_connect} usando o schema {final_schema_path}")

        try:
            with open(final_schema_path, "r") as f:
                schema = f.read()

            self.logger.debug(f"Arquivo schema lido com sucesso", schema_path=final_schema_path)

            if path_to_connect is None:
                raise ValueError("db_path não pode ser None quando control for False")
            
            db_dir = os.path.dirname(path_to_connect)

            if db_dir and not os.path.exists(db_dir): 
                self.logger.info(f"Diretorio do banco de dados nao existe. Criando: {db_dir}")
                os.makedirs(db_dir, exist_ok=True)

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
                self._create_database(control=True)

            self.logger.info(f"Conectando ao banco de dados de controle em: {self.control_db_path}")
            return sqlite3.connect(self.control_db_path)
        
        if not db_path:
            self.logger.error("db_path nao pode ser None quando control for False")
            raise ValueError("db_path nao pode ser None quando control for False")
        
        self.db_path = db_path

        if not os.path.exists(self.db_path):
            self.logger.info(f"Banco de dados nao encontrado em {self.db_path}. Inicializando o banco de dados...")
            self._create_database(db_path=self.db_path)
        
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
                    file = DocumentFile( # propriedade 'pages' não é armazenada no banco de dados
                        id=row[0],
                        hash=row[1],
                        name=row[2],
                        path=row[3],
                        total_pages=row[4],
                        created_at=row[5],
                        updated_at=row[6]
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
        
    def insert_chunks(self, chunks: List[Chunk], file_id: int, conn: sqlite3.Connection) -> List[int]:
        """
        Insere um chunk no banco de dados.
        Args:
            chunk: objeto Chunk a ser inserido.
            file_id: ID do documento associado ao chunk.
            conn: Conexão com o banco de dados SQLite.
        """
        self.logger.debug(f"Inserindo objetos Chunk no banco de dados: {self.db_path}")
        inserted_ids: List[int] = []
        for chunk in chunks:

            try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO chunks (document_id, content, metadata) VALUES (?, ?, ?)", 
                        (file_id, chunk.content, json.dumps(chunk.metadata))
                    )
                    chunk.id = cursor.lastrowid
                    inserted_ids.append(chunk.id)

            except sqlite3.Error as e:
                self.logger.error(f"Erro ao inserir chunks: {e}")
                raise e
            
        self.logger.info(f"{len(inserted_ids)} Chunks inseridos com sucesso") 
        return inserted_ids
        
    def get_chunks(self, conn: sqlite3.Connection, chunk_ids: Optional[List[int]] = None, file_id: Optional[int] = None) -> List[Chunk]:
        """
        Retorna o conteúdo dos chunks associados aos índices faiss fornecidos.

        Args:
            conn: Conexão com o banco de dados SQLite.
            faiss_indices: Lista de índices faiss correspondentes aos chunks.

        Returns:
            chunks_content: List[str], onde cada string contém o conteúdo de um chunk, na ordem dos índices faiss.
        """
        self.logger.info(f"Recuperando chunks do banco de dados")
        try:
            cursor = conn.cursor()

            # Se um file_id for fornecido, recupera todos os chunks associados ao documento
            if file_id:
                self.logger.info(f"Recuperando chunks do documento: {file_id} no banco de dados: {self.db_path}")
                cursor.execute("SELECT * FROM chunks WHERE document_id = ?", (file_id,))
            elif chunk_ids: 
                # Se ids forem fornecidos, recupera os chunks associados a eles
                self.logger.info(f"Recuperando chunks do banco de dados: {self.db_path}")
                order_cases = " ".join([f"WHEN {index} THEN {i}" for i, index in enumerate(chunk_ids)])
                placeholders = ", ".join(['?'] * len(chunk_ids))
                query = f"""
                    SELECT * 
                    FROM chunks 
                    WHERE id IN ({placeholders})
                    ORDER BY CASE id
                        {order_cases}
                    END
                """
                cursor.execute(query, chunk_ids)

            # Cria objetos Chunk
            chunks : List[Chunk] = []
            chunk_data = cursor.fetchall()
            if chunk_data:
                for row in chunk_data:
                    chunk = Chunk(
                        id=row[0],
                        document_id=row[1],
                        content=row[2],
                        metadata=json.loads(row[3]), 
                        created_at=row[4]
                    )
                    chunks.append(chunk)

            self.logger.info(f"Chunks recuperados com sucesso")
            return chunks
        
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao recuperar os chunks: {e}")
            raise e

    def insert_domain(self, domain: Domain, conn: sqlite3.Connection) -> None:
        """
        Insere um domínio de conhecimento no banco de dados de controle.
        """
        self.logger.info(f"Inserindo dominio de conhecimento no banco de dados: {domain.name}")


        params = []
        field_names_list = []
        field_cont = 0
        for field, value in domain.model_dump().items():
            if value is not None:
                field_cont += 1
                field_names_list.append(f"{field}")
                params.append(value)
        placeholders = ", ".join("?" * field_cont)

        field_names = ", ".join(field_names_list)
        query = f"INSERT INTO knowledge_domains ({field_names}) VALUES ({placeholders})"


        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()

            self.logger.debug("Domínio do conhecimento inserido com sucesso", 
                              domain_name=domain.name, 
                              domain_description=domain.description, 
                              domain_keywords=domain.keywords, 
                              domain_vector_store_path=domain.vector_store_path,
                              domain_db_path=domain.db_path,
                              domain_embeddings_dimension=domain.embeddings_dimension)
            
            cursor.execute("SELECT last_insert_rowid()")
            domain_id = cursor.fetchone()[0]
            
            return domain_id

        except sqlite3.Error as e:
            self.logger.error(f"Erro ao inserir o domínio de conhecimento: {e}")
            raise e

    def get_domain(self, conn: sqlite3.Connection, domain_name: Optional[str] = None) -> Optional[List[Domain]]:
        """
        Retorna um ou todos os domínios de conhecimento do banco de dados de controle.
        """
        self.logger.debug(f"Recuperando dominio(s) de conhecimento do banco de dados: {domain_name or 'Todos'}")
        try:
            cursor = conn.cursor()
            if domain_name:
                cursor.execute("SELECT * FROM knowledge_domains WHERE name = ?", (domain_name,))
            else:
                cursor.execute("SELECT * FROM knowledge_domains")
            
            all_domains : List[Domain] = []
            domain_data = cursor.fetchall()
            if domain_data:
                # Recupera os nomes das colunas
                columns = [description[0] for description in cursor.description]
                
                for row in domain_data:
                    # Cria um dicionário com os nomes das colunas e os valores da linha
                    domain_dict = dict(zip(columns, row))
                    domain = Domain(**domain_dict)
                    all_domains.append(domain)
                return all_domains
            else:
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao recuperar o(s) dominio(s) de conhecimento: {e}")
            raise e
        
    def update_domain(self, domain: Domain, conn: sqlite3.Connection, update: Dict[str, Any]) -> None:
        """
        Atualiza um domínio de conhecimento no banco de dados de controle.
        """      
        self.logger.debug(f"Atualizando dominio de conhecimento no banco de dados: {domain.name}")

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
            self.logger.debug(f"Atualização executada para o dominio. Aguardando commit",)

        except sqlite3.Error as e:
            self.logger.error(f"Erro ao atualizar o dominio de conhecimento: {e}")
            raise e
        
    def delete_domain(self, domain: Domain, conn: sqlite3.Connection) -> None:
        """
        Deleta um domínio de conhecimento do banco de dados de controle.
        """
        self.logger.debug(f"Deletando dominio de conhecimento do banco de dados: {domain.name}")
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_domains WHERE id = ?", (domain.id,))
            self.logger.debug(f"Dominio removido do DB. Aguardando commit", domain_name=domain.name)

        except sqlite3.Error as e:
            self.logger.error(f"Erro ao deletar o dominio de conhecimento: {e}")
            raise e
    
    def insert_domain_config(self, domain_config: DomainConfig, conn: sqlite3.Connection) -> None:
        """
        Insere uma configuração de domínio no banco de dados de controle.
        """
        self.logger.debug(f"Inserindo configuração de domínio no banco de dados: {domain_config.domain_id}")

        params = []
        field_names_list = []
        field_cont = 0
        for field, value in domain_config.model_dump().items():
            if value is not None:
                field_cont += 1
                field_names_list.append(f"{field}")
                params.append(value)
        placeholders = ", ".join("?" * field_cont)

        field_names = ", ".join(field_names_list)
        query = f"INSERT INTO knowledge_domain_configs ({field_names}) VALUES ({placeholders})"

        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()

        except sqlite3.Error as e:
            self.logger.error(f"Erro ao inserir a configuração de domínio: {e}")
            raise e

    def get_domain_config(self, conn: sqlite3.Connection, domain_id: int) -> Optional[DomainConfig]:
        """
        Recupera a configuração de domínio do banco de dados de controle.
        """
        self.logger.debug(f"Recuperando configuração de domínio do banco de dados: {domain_id}")
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_domain_configs WHERE domain_id = ?", (domain_id,))
            domain_config_data = cursor.fetchone()
            if domain_config_data:
                # Recupera os nomes das colunas
                columns = [description[0] for description in cursor.description]
                # Cria um dicionário com os nomes das colunas e os valores da linha
                domain_config_dict = dict(zip(columns, domain_config_data))
                return DomainConfig(**domain_config_dict)
            else:
                return None
        except sqlite3.Error as e:
            self.logger.error(f"Erro ao recuperar a configuração de domínio: {e}")
            raise e