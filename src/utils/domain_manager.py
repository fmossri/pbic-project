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
        treated_domain_name = domain_name.lower().replace(" ", "_")
        #Cria os caminhos da db e vectorstore
        domain_db_path = os.path.join("storage", "domains", f"{treated_domain_name}", f"{treated_domain_name}.db")
        vector_store_path = os.path.join("storage", "domains", f"{treated_domain_name}", "vector_store",f"{treated_domain_name}.faiss")

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

                # Verifica se o domain já existe
                existing_domain = self.sqlite_manager.get_domain(conn, domain_name)    
                if existing_domain:
                    self.logger.error("Dominio ja existe", domain_name=domain_name)
                    conn.rollback()
                    raise ValueError(f"Domínio já existe: {domain_name}")

                # Cria o domínio e insere no banco de dados
                domain = Domain(**domain_data)
                self.sqlite_manager.insert_domain(domain, conn)
                conn.commit()
                self.logger.info("Dominio de conhecimento adicionado com sucesso", domain_name=domain_name)

        except Exception as e:
            self.logger.error(f"Erro ao adicionar novo domínio de conhecimento: {e}")
            raise e
        
    def remove_domain_registry_and_files(self, domain_name: str) -> None:
        """
        Remove um domínio de conhecimento do banco de dados.
        """
        self.logger.info("Removendo dominio de conhecimento", domain_name=domain_name)
        try:
            with self.sqlite_manager.get_connection(control=True) as conn:
                self.sqlite_manager.begin(conn)

                domain = self.sqlite_manager.get_domain(conn, domain_name)
                if not domain:
                    self.logger.error("Dominio nao encontrado", domain_name=domain_name)
                    conn.rollback()
                    raise ValueError(f"Domínio não encontrado: {domain_name}")

                # Desempacota o domínio
                [domain] = domain

                #Deleta o diretório do domínio
                domain_dir = os.path.join("storage", "domains", f"{domain.name}")
                if os.path.isdir(domain_dir):
                    shutil.rmtree(domain_dir)
                    self.logger.info("Diretório e arquivos do dominio removidos com sucesso", domain_name=domain.name)
                else:
                    self.logger.warning("Diretório do dominio não encontrado, removendo o registro do dominio", domain_name=domain.name)

                self.sqlite_manager.delete_domain(domain, conn)
                conn.commit()
                self.logger.info("Dominio de conhecimento removido com sucesso", domain_name=domain.name)
        except Exception as e:
            self.logger.error(f"Erro ao deletar dominio de conhecimento: {e}")
            raise e
        
    def update_domain_details(self, domain_name: str, updates: Dict[str, Any]) -> None:
        """
        Atualiza um domínio de conhecimento existente no banco de dados.
        """
        self.logger.info("Atualizando dominio de conhecimento", domain_name=domain_name)
        try:
            with self.sqlite_manager.get_connection(control=True) as conn:
                #Verifica se o domínio existe
                domain = self.sqlite_manager.get_domain(conn, domain_name)
                if not domain:
                    self.logger.error("Dominio não encontrado", domain_name=domain_name)
                    raise ValueError(f"Domínio não encontrado: {domain_name}")

                [domain] = domain
                
                # Seleciona os campos do domínio que podem ser atualizados
                update_fields = {}
                for column, value in updates.items():
                    if column in Domain.updatable_fields() and value != getattr(domain, column):
                        update_fields[column] = value
                    # Log warning if field is not updatable OR if value is the same
                    elif column not in Domain.updatable_fields():
                        self.logger.warning("Campo nao pode ser atualizado manualmente", column=column)
                    else: # Optional: log if value is the same
                        self.logger.debug(f"Valor para '{column}' é o mesmo, atualizacao ignorada.")
                
                # If no valid fields to update, return early
                if not update_fields:
                    self.logger.info("Nenhum campo valido ou alterado para atualizar.")
                    return # Exit before starting transaction

                if "name" in update_fields.keys():
                    # Verifica se o novo nome já existe
                    new_name = update_fields["name"]
                    if self.sqlite_manager.get_domain(conn, new_name):
                        self.logger.error("Dominio ja existe", domain_name=new_name)
                        raise ValueError(f"Domínio já existe: {new_name}")
                    
                    # Renomeia os arquivos do domínio
                    old_name = domain.name
                    new_db_path, new_faiss_path = self.rename_domain_paths(old_name, new_name)
                    update_fields["db_path"] = new_db_path
                    update_fields["vector_store_path"] = new_faiss_path

                # Atualiza o domínio
                self.sqlite_manager.begin(conn)
                self.sqlite_manager.update_domain(domain, conn, update_fields)
                conn.commit()
                self.logger.info("Dominio de conhecimento atualizado com sucesso", domain_name=domain.name)
        except Exception as e:
            self.logger.error(f"Erro ao atualizar dominio de conhecimento: {e}")
            raise e

    def rename_domain_paths(self, old_name: str, new_name: str) -> Tuple[str, str]:
        """
        Renomeia os arquivos do domínio.
        """
        self.logger.info("Renomeando arquivos do dominio", old_name=old_name, new_name=new_name)
        missing_files = []
        try:
            # Tenta renomear o diretório
            old_dir = os.path.join("storage", "domains", f"{old_name}")
            new_dir = os.path.join("storage", "domains", f"{new_name}")
            if os.path.exists(old_dir):
                os.rename(old_dir, new_dir)
            else:
                self.logger.warning(f"Diretório {old_dir} nao encontrado, pulando renomeacao do diretório.")

            # Tenta renomear o banco de dados
            old_db_path = os.path.join(f"{new_dir}", f"{old_name}.db")
            new_db_path = os.path.join(f"{new_dir}", f"{new_name}.db")
            if os.path.exists(old_db_path):
                os.rename(old_db_path, new_db_path)
            else:
                self.logger.warning(f"Arquivo .db {old_db_path} nao encontrado, pulando renomeacao do banco de dados.")
                missing_files.append(old_db_path)
            # Tenta renomear o vector_store
            old_faiss_path = os.path.join(f"{new_dir}", "vector_store", f"{old_name}.faiss")
            new_faiss_path = os.path.join(f"{new_dir}", "vector_store", f"{new_name}.faiss")
            if os.path.exists(old_faiss_path):
                os.rename(old_faiss_path, new_faiss_path)
            else:
                self.logger.warning(f"Arquivo .faiss {old_faiss_path} nao encontrado, pulando renomeacao do vectorstore.")
                missing_files.append(old_faiss_path)

            if len(missing_files) == 1:
                self.logger.critical(f"Alerta! Inconsistência encontrada no sistema de arquivos. {missing_files[0]} nao existe. Remova o dominio e seus arquivos.", domain_name=old_name, missing_files=missing_files)
                if os.path.isdir(new_dir):
                    try:
                        os.rename(new_dir, old_dir)
                        self.logger.info("Tentativa de reverter renomeacao do diretório.")
                    except OSError as rb_err:
                        self.logger.error(f"Falha ao reverter renomeacao do diretório: {rb_err}. Sistema de arquivos pode estar inconsistente.")
            
            return new_db_path, new_faiss_path

        except OSError as e:
            self.logger.error(f"Erro ao renomear caminhos do dominio: {e}")
            if os.path.isdir(new_dir):
                try:
                    os.rename(new_dir, old_dir)
                    self.logger.info("Tentativa de reverter renomeacao do diretório.")
                except OSError as rb_err:
                    self.logger.error(f"Falha ao reverter renomeacao do diretório: {rb_err}. Sistema de arquivos pode estar inconsistente.")
            raise OSError(f"Falha na renomeação do sistema de arquivos: {e}") from e

    def list_domains(self) -> List[Domain]:
        """
        Lista todos os domínios de conhecimento.
        """
        self.logger.info("Listando dominios de conhecimento")
        try:
            with self.sqlite_manager.get_connection(control=True) as conn:
                domains = self.sqlite_manager.get_domain(conn)
                self.logger.info("Dominios listados com sucesso", domains=domains)
                return domains
        except Exception as e:
            self.logger.error(f"Erro ao listar dominios de conhecimento: {e}")

    def list_domain_documents(self, domain_name: str) -> List[str]:
        """
        Lista todos os documentos de um domínio de conhecimento.
        """
        self.logger.info("Listando documentos do dominio de conhecimento", domain_name=domain_name)
        try:
            self.logger.debug("Conectando ao banco de dados de controle")
            with self.sqlite_manager.get_connection(control=True) as conn:
                domain = self.sqlite_manager.get_domain(conn, domain_name)

                if not domain:
                    self.logger.error("Dominio nao encontrado", domain_name=domain_name)
                    raise ValueError(f"Domínio não encontrado: {domain_name}")

            [domain] = domain 

            self.logger.debug("Conectando ao banco de dados do dominio", domain_name=domain_name)
            with self.sqlite_manager.get_connection(control=False, db_path=domain.db_path) as conn:

                documents = self.sqlite_manager.get_document_file(conn)
                self.logger.info("Documentos listados com sucesso", documents=documents)
                return documents

        except Exception as e:
            self.logger.error(f"Erro ao listar documentos do dominio de conhecimento: {e}")
            raise e