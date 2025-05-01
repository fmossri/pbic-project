import os
import shutil
from typing import Dict, Any, List, Tuple
from src.utils.sqlite_manager import SQLiteManager
from src.utils.logger import get_logger
from src.models import Domain, DocumentFile
from src.config.models import SystemConfig

class DomainManager:

    def __init__(self, config: SystemConfig, sqlite_manager: SQLiteManager, log_domain: str = "utils"):
        self.config = config
        self.sqlite_manager = sqlite_manager
        self.storage_base_path = config.storage_base_path

        self.logger = get_logger(__name__, log_domain)
        self.logger.info("Inicializando DomainManager")

    def create_domain(self, domain_name: str, domain_description: str, domain_keywords: str) -> None:
        """
        Adiciona um novo domínio de conhecimento ao banco de dados.

        Args:
            domain_name (str): Nome do domínio de conhecimento.
            domain_description (str): Descrição do domínio de conhecimento.
            domain_keywords (str): Palavras-chave do domínio de conhecimento, separadas por virgulas.
        """
        if not isinstance(domain_name, str) or not isinstance(domain_description, str) or not isinstance(domain_keywords, str):
            self.logger.error("Nome, descrição e palavras-chave devem ser strings")
            raise ValueError("Nome, descrição e palavras-chave devem ser strings")

        treated_domain_name = domain_name.lower().replace(" ", "_")

        # Cria os diretórios e os arquivos do domínio
        domain_root_path = os.path.join(self.storage_base_path, "domains", treated_domain_name)
        domain_db_path = os.path.join(domain_root_path, f"{treated_domain_name}.db")
        vector_store_path = os.path.join(domain_root_path, "vector_store", f"{treated_domain_name}.faiss")

        domain_data = {
            "name": domain_name,
            "description": domain_description,
            "keywords": domain_keywords,
            "db_path": domain_db_path,
            "vector_store_path": vector_store_path,
        }

        self.logger.info("Adicionando novo domínio de conhecimento", **domain_data)

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
            self.logger.error(f"Erro ao adicionar novo domínio de conhecimento: {e}", exc_info=True)
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

                # Remove o diretório e os arquivos do domínio
                treated_domain_name = domain.name.lower().replace(" ", "_")
                domain_dir = os.path.join(self.storage_base_path, "domains", treated_domain_name)
                if os.path.isdir(domain_dir):
                    shutil.rmtree(domain_dir)
                    self.logger.info("Diretorio e arquivos do dominio removidos com sucesso", domain_directory=domain_dir)
                else:
                    self.logger.warning("Diretorio do dominio nao encontrado, removendo o registro do dominio", domain_name=domain.name)

                self.sqlite_manager.delete_domain(domain, conn)
                conn.commit()
                self.logger.info("Dominio de conhecimento removido com sucesso", domain_name=domain.name)
        except Exception as e:
            self.logger.error(f"Erro ao remover dominio de conhecimento: {e}", exc_info=True)
            raise e

    def update_domain_details(self, domain_name: str, updates: Dict[str, Any]) -> None:
        """
        Atualiza um domínio de conhecimento existente no banco de dados.

        Args:
            domain_name (str): Nome do domínio de conhecimento a ser atualizado.
            updates (Dict[str, Any]): Dicionário contendo os campos a serem atualizados e seus novos valores.
        """
        self.logger.info("Atualizando dominio de conhecimento", domain_name=domain_name, updates=updates)
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

                    if column == "name":
                        new_name = value
                        if self.sqlite_manager.get_domain(conn, new_name):
                            self.logger.error("Dominio ja existe", domain_name=new_name)
                            raise ValueError(f"Domínio já existe: {new_name}")
                        
                        # Renomeia os campos do domínio e seus arquivos, se existirem
                        new_db_path, new_faiss_path = self._rename_domain_paths(domain.name, new_name)
                        update_fields["name"] = new_name
                        update_fields["db_path"] = new_db_path
                        update_fields["vector_store_path"] = new_faiss_path

                    # Verifica se o campo pode ser atualizado manualmente
                    elif column in Domain.updatable_fields() and value != getattr(domain, column):
                        update_fields[column] = value

                    elif column not in Domain.updatable_fields():
                        self.logger.warning("Campo nao pode ser atualizado manualmente", column=column)

                    else:
                        self.logger.debug(f"Novo valor para '{column}' é o mesmo que o atual.")
                
                if not update_fields:
                    self.logger.info("Nenhum campo valido ou alterado para atualizar.")
                    return
                    
                # Atualiza o domínio no banco de dados
                self.sqlite_manager.begin(conn)
                self.sqlite_manager.update_domain(domain, conn, update_fields)
                conn.commit()
                self.logger.info("Dominio de conhecimento atualizado com sucesso", domain_name=domain.name, updated_fields=list(update_fields.keys()))
        except Exception as e:
            self.logger.error(f"Erro ao atualizar dominio de conhecimento: {e}", exc_info=True)
            raise e

    def _rename_domain_paths(self, old_name: str, new_name: str) -> Tuple[str, str]:
        """
        Renomeia o diretório e os arquivos contidos (.db, .faiss) do domínio.

        Args:
            old_name (str): Nome antigo do domínio.
            new_name (str): Novo nome do domínio.

        Returns:
            Tuple[str, str]: Novos caminhos para o banco de dados e o vector store.
        """
        self.logger.info("Renomeando arquivos do dominio", old_name=old_name, new_name=new_name)

        old_name_fs = old_name.lower().replace(" ", "_")
        new_name_fs = new_name.lower().replace(" ", "_")
        old_dir = os.path.join(self.storage_base_path, "domains", f"{old_name_fs}")
        new_dir = os.path.join(self.storage_base_path, "domains", f"{new_name_fs}")
        old_db_path = os.path.join(f"{old_dir}", f"{old_name_fs}.db")
        new_db_path = os.path.join(f"{new_dir}", f"{new_name_fs}.db")
        old_faiss_path = os.path.join(f"{old_dir}", "vector_store", f"{old_name_fs}.faiss")
        new_faiss_path = os.path.join(f"{new_dir}", "vector_store", f"{new_name_fs}.faiss")

        try:
            # Tenta renomear o diretório
            if os.path.exists(new_dir):
                self.logger.error("O diretorio ja existe", new_dir=new_dir)
                raise FileExistsError(f"Diretorio já existe: {new_dir}")

            # Se o diretório não existe, provavelmente o domínio ainda não ingeriu documentos. Apenas retorna os novos caminhos
            if not os.path.exists(old_dir):
                return (new_db_path, new_faiss_path)
            
            if not os.path.exists(old_db_path) and os.path.exists(old_faiss_path):
                self.logger.error(f"Arquivo nao encontrado no diretório {old_dir}.")
                raise FileNotFoundError(f"Arquivo nao encontrado no diretório {old_dir}.")
            
            # Renomeia o diretório
            os.rename(old_dir, new_dir)

            # Caminhos dos arquivos antigos no novo diretório
            old_db_path = os.path.join(f"{new_dir}", f"{old_name_fs}.db")
            old_faiss_path = os.path.join(f"{new_dir}", "vector_store", f"{old_name_fs}.faiss")

            os.rename(old_db_path, new_db_path)
            os.rename(old_faiss_path, new_faiss_path)

            return new_db_path, new_faiss_path

        except OSError as e:
            self.logger.error(f"Erro ao renomear caminhos do dominio: {e}", exc_info=True)
            if os.path.exists(new_dir):
                os.rename(new_dir, old_dir)
            raise e

    def list_domains(self) -> List[Domain]:
        """
        Lista todos os domínios de conhecimento.
        """
        self.logger.info("Listando dominios de conhecimento")
        try:
            with self.sqlite_manager.get_connection(control=True) as conn:
                domains = self.sqlite_manager.get_domain(conn)
                
                if domains is None:
                    self.logger.info("Nenhum dominio encontrado no banco de dados.")
                    return None
                else:
                    domain_names = [domain.name for domain in domains]
                    self.logger.info("Dominios listados com sucesso", domains=domain_names)
                    return domains

        except Exception as e:
            self.logger.error(f"Erro ao listar dominios de conhecimento: {e}", exc_info=True)
            raise e

    def list_domain_documents(self, domain_name: str) -> List[DocumentFile]:
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

            if not os.path.exists(domain.db_path):
                self.logger.error("Banco de dados do dominio nao encontrado", domain_db_path=domain.db_path)
                raise FileNotFoundError(f"Banco de dados do domínio não encontrado: {domain.db_path}")
            
            self.logger.info("Conectando ao banco de dados do dominio", domain_db_path=domain.db_path)
            with self.sqlite_manager.get_connection(db_path=domain.db_path) as conn:

                documents = self.sqlite_manager.get_document_file(conn)

                if documents:
                    self.logger.info("Documentos listados com sucesso")
                    return documents
                else:
                    self.logger.info("Nenhum documento encontrado para o dominio", domain_name=domain_name)
                    return []

        except Exception as e:
            self.logger.error(f"Erro ao listar documentos do dominio de conhecimento: {e}", exc_info=True)
            raise e