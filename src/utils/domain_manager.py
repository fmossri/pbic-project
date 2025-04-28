import os
import shutil
from typing import Dict, Any, List, Tuple
from src.utils.sqlite_manager import SQLiteManager
from src.utils.logger import get_logger
from src.models import Domain

class DomainManager:
    def __init__(self, log_domain: str = "utils"):
        self.logger = get_logger(__name__, log_domain)
        self.logger.info("Inicializando DomainManager")
        
        self.control_db_path = os.path.join("storage", "domains", "control.db")
        self.sqlite_manager = SQLiteManager()

    def create_domain(self, domain_name: str, domain_description: str, domain_keywords: str) -> None:
        """
        Adiciona um novo domínio de conhecimento ao banco de dados.

        Args:
            domain_name (str): Nome do domínio de conhecimento.
            domain_description (str): Descrição do domínio de conhecimento.
            domain_keywords (str): Palavras-chave do domínio de conhecimento, separadas por virgulas.
        """
        #Cria os caminhos da db e vectorstore
        domain_db_path = os.path.join("storage", "domains", f"{domain_name}", f"{domain_name}.db")
        vector_store_path = os.path.join("storage", "domains", f"{domain_name}", "vector_store",f"{domain_name}.faiss")

        domain_data = {
            "name": domain_name,
            "description": domain_description,
            "keywords": domain_keywords,
            "db_path": domain_db_path,
            "vector_store_path": vector_store_path,
        }
        
        self.logger.info("Adicionando novo domínio de conhecimento", domain_name=domain_name, domain_description=domain_description, domain_keywords=domain_keywords, domain_db_path=domain_db_path, vector_store_path=vector_store_path)

        try:
            with self.sqlite_manager.get_connection(control=True) as conn:  
                self.sqlite_manager.begin(conn)

                #Verifica se o domain já existe
                domain = self.sqlite_manager.get_domain(conn, domain_name)
                if domain:
                    self.logger.error("Domínio já existe", domain=domain.name)
                    conn.rollback()
                    raise ValueError(f"Domínio já existe: {domain.name}")

                domain = Domain(**domain_data)

                self.sqlite_manager.insert_domain(domain, conn)
                conn.commit()
                self.logger.info("Domínio de conhecimento adicionado com sucesso", domain_name=domain_name)

        except Exception as e:
            self.logger.error(f"Erro ao adicionar novo domínio de conhecimento: {e}")
            raise e
        
    def remove_domain_registry_and_files(self, domain_name: str) -> None:
        """
        Remove um domínio de conhecimento do banco de dados.
        """
        self.logger.info("Removendo domínio de conhecimento", domain_name=domain_name)
        try:
            with self.sqlite_manager.get_connection(control=True) as conn:
                self.sqlite_manager.begin(conn)

                domain = self.sqlite_manager.get_domain(conn, domain_name)
                if not domain:
                    self.logger.error("Domínio não encontrado", domain_name=domain_name)
                    conn.rollback()
                    raise ValueError(f"Domínio não encontrado: {domain_name}")

                #Deleta o diretório do domínio
                domain_dir = os.path.join("storage", "domains", f"{domain.name}")
                if os.path.isdir(domain_dir):
                    shutil.rmtree(domain_dir)
                    self.logger.info("Diretório e arquivos do domínio removidos com sucesso", domain_name=domain.name)
                else:
                    self.logger.warning("Diretório do domínio não encontrado, removendo o registro do domínio", domain_name=domain.name)

                self.sqlite_manager.delete_domain(domain, conn)
                conn.commit()
                self.logger.info("Domínio de conhecimento removido com sucesso", domain_name=domain.name)
        except Exception as e:
            self.logger.error(f"Erro ao deletar domínio de conhecimento: {e}")
            raise e
        
    def update_domain_details(self, domain_name: str, updates: Dict[str, Any]) -> None:
        """
        Atualiza um domínio de conhecimento existente no banco de dados.
        """
        self.logger.info("Atualizando domínio de conhecimento", domain_name=domain_name)
        try:
            with self.sqlite_manager.get_connection(control=True) as conn:
                #Verifica se o domínio existe
                [domain] = self.sqlite_manager.get_domain(conn, domain_name)
                if not domain:
                    self.logger.error("Domínio não encontrado", domain_name=domain_name)
                    raise ValueError(f"Domínio não encontrado: {domain_name}")

                update_fields = {}
                for column, value in updates.items():
                    if column in Domain.updatable_fields() and value != getattr(domain, column):
                        update_fields[column] = value

                    else:
                        self.logger.warning("Campo não pode ser atualizado manualmente", column=column)

                if "name" in update_fields.keys():
                    if self.sqlite_manager.get_domain(conn, update_fields["name"]):
                        self.logger.error("Domínio já existe", domain_name=update_fields["name"])
                        raise ValueError(f"Domínio já existe: {update_fields['name']}")
                    
                    old_name = domain.name
                    new_name = update_fields["name"]
                    
                    new_db_path, new_faiss_path = self.rename_domain_paths(old_name, new_name)

                    update_fields["db_path"] = new_db_path
                    update_fields["vector_store_path"] = new_faiss_path

                #Atualiza o domínio
                self.sqlite_manager.begin(conn)
                self.sqlite_manager.update_domain(domain, conn, update_fields)
                conn.commit()
                self.logger.info("Domínio de conhecimento atualizado com sucesso", domain_name=domain_name)
        except Exception as e:
            self.logger.error(f"Erro ao atualizar domínio de conhecimento: {e}")
            raise e

    def rename_domain_paths(self, old_name: str, new_name: str) -> Tuple[str, str]:
        """
        Renomeia os arquivos do domínio.
        """
        self.logger.info("Renomeando arquivos do domínio", old_name=old_name, new_name=new_name)
        try:
            # Tenta renomear o diretório
            old_dir = os.path.join("storage", "domains", f"{old_name}")
            new_dir = os.path.join("storage", "domains", f"{new_name}")
            if os.path.exists(old_dir):
                os.rename(old_dir, new_dir)
            else:
                self.logger.warning(f"Diretório {old_dir} não encontrado, pulando renomeação do diretório.")

            # Tenta renomear o banco de dados
            old_db_path = os.path.join(f"{new_dir}", f"{old_name}.db")
            new_db_path = os.path.join(f"{new_dir}", f"{new_name}.db")
            if os.path.exists(old_db_path):
                os.rename(old_db_path, new_db_path)
            else:
                self.logger.warning(f"Arquivo .db {old_db_path} não encontrado, pulando renomeação do banco de dados.")

            # Tenta renomear o vector_store
            old_faiss_path = os.path.join(f"{new_dir}", "vector_store", f"{old_name}.faiss")
            new_faiss_path = os.path.join(f"{new_dir}", "vector_store", f"{new_name}.faiss")
            if os.path.exists(old_faiss_path):
                os.rename(old_faiss_path, new_faiss_path)
            else:
                self.logger.warning(f"Arquivo .faiss {old_faiss_path} não encontrado, pulando renomeação do vectorstore.")

            return new_db_path, new_faiss_path

        except OSError as e:
            self.logger.error(f"Erro ao renomear caminhos do domínio: {e}")
            if os.path.isdir(new_dir):
                try:
                    os.rename(new_dir, old_dir)
                    self.logger.info("Tentativa de reverter renomeação do diretório.")
                except OSError as rb_err:
                    self.logger.error(f"Falha ao reverter renomeação do diretório: {rb_err}. Sistema de arquivos pode estar inconsistente.")
            raise OSError(f"Falha na renomeação do sistema de arquivos: {e}") from e

    def list_domains(self) -> List[Domain]:
        """
        Lista todos os domínios de conhecimento.
        """
        self.logger.info("Listando domínios de conhecimento")
        try:
            with self.sqlite_manager.get_connection(control=True) as conn:
                domains = self.sqlite_manager.get_domain(conn)
                return domains
        except Exception as e:
            self.logger.error(f"Erro ao listar domínios de conhecimento: {e}")

    def list_domain_documents(self, domain_name: str) -> List[str]:
        """
        Lista todos os documentos de um domínio de conhecimento.
        """
        self.logger.info("Listando documentos do domínio de conhecimento", domain_name=domain_name)
        try:
            with self.sqlite_manager.get_connection(control=True) as conn:
                [domain] = self.sqlite_manager.get_domain(conn, domain_name)

                if not domain:
                    self.logger.error("Domínio não encontrado", domain_name=domain_name)
                    raise ValueError(f"Domínio não encontrado: {domain_name}")

            with self.sqlite_manager.get_connection(control=False, db_path=domain.db_path) as conn:

                documents = self.sqlite_manager.get_document_file(conn)
                return documents

        except Exception as e:
            self.logger.error(f"Erro ao listar documentos do domínio de conhecimento: {e}")
            raise e