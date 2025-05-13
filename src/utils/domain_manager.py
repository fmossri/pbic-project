import os
import shutil
from typing import Dict, Any, List, Tuple
from src.utils.sqlite_manager import SQLiteManager
from src.utils.logger import get_logger
from src.models import Domain, DocumentFile, DomainConfig
from src.config.models import AppConfig
class DomainManager:

    def __init__(self, config: AppConfig, sqlite_manager: SQLiteManager, log_domain: str = "utils"):
        self.config = config
        self.sqlite_manager = sqlite_manager
        self.storage_base_path = config.system.storage_base_path

        self.logger = get_logger(__name__, log_domain)
        self.logger.info("Inicializando DomainManager")

    def update_config(self, new_config: AppConfig) -> None:
        """
        Atualiza a configuração do DomainManager com base na configuração fornecida.

        Args:
            config (AppConfig): A nova configuração a ser aplicada.
        """
        self.config = new_config
        self.storage_base_path = new_config.system.storage_base_path
        self.sqlite_manager.update_config(new_config.system)
        self.logger.info("Configuracoes do DomainManager atualizadas com sucesso")

    def create_domain(self, new_domain_data: Dict[str, Any]) -> None:
        """
        Adiciona um novo domínio de conhecimento ao banco de dados.

        Args:
            new_domain_data (Dict[str, Any]): Dicionário contendo os dados do novo domínio de conhecimento. Deve conter os seguintes campos:
                - name: Nome do domínio de conhecimento.
                - description: Descrição do domínio de conhecimento.
                - keywords: Palavras-chave do domínio de conhecimento, separadas por virgulas.
                - embedding_model: Modelo de embedding a ser utilizado.
                - faiss_index_type: Tipo de índice Faiss a ser utilizado.
        """
        if not isinstance(new_domain_data["name"], str) or not isinstance(new_domain_data["description"], str) or not isinstance(new_domain_data["keywords"], str):
            self.logger.error("Nome, descrição e palavras-chave devem ser strings")
            raise ValueError("Nome, descrição e palavras-chave devem ser strings")
        
        if not new_domain_data["embeddings_model"] in self.config.embedding.embedding_options:
            self.logger.error("Modelo de embedding invalido", model=new_domain_data["embeddings_model"])
            raise ValueError("Modelo de embedding invalido")
        
        if not new_domain_data["faiss_index_type"] in self.config.vector_store.vector_store_options:
            self.logger.error("Tipo de indice Faiss invalido", index_type=new_domain_data["faiss_index_type"])
            raise ValueError("Tipo de indice Faiss invalido")

        treated_domain_name = new_domain_data["name"].lower().replace(" ", "_")

        # Cria os diretórios e os arquivos do domínio
        domain_root_path = os.path.join(self.storage_base_path, treated_domain_name)
        domain_db_path = os.path.join(domain_root_path, f"{treated_domain_name}.db")
        vector_store_path = os.path.join(domain_root_path, "vector_store", f"{treated_domain_name}.faiss")

        domain_data = {
            "name": new_domain_data["name"],
            "description": new_domain_data["description"],
            "keywords": new_domain_data["keywords"],
            "db_path": domain_db_path,
            "vector_store_path": vector_store_path,
        }

        self.logger.info("Adicionando novo domínio de conhecimento", **domain_data)

        try:
            with self.sqlite_manager.get_connection(control=True) as conn:  
                self.sqlite_manager.begin(conn)

                # Verifica se o domain já existe
                existing_domain = self.sqlite_manager.get_domain(conn, domain_data["name"])    
                if existing_domain:
                    self.logger.error("Dominio ja existe", domain_name=domain_data["name"])
                    conn.rollback()
                    raise ValueError(f"Domínio já existe: {domain_data['name']}")

                # Cria o domínio e insere no banco de dados
                domain = Domain(**domain_data)
                domain_id = self.sqlite_manager.insert_domain(domain, conn)

                domain_config_data = {
                    "domain_id": domain_id,
                    "embeddings_model": new_domain_data["embeddings_model"],
                    "faiss_index_type": new_domain_data["faiss_index_type"],
                    "chunking_strategy": new_domain_data["chunking_strategy"],
                    "chunk_size": new_domain_data["chunk_size"],
                    "chunk_overlap": new_domain_data["chunk_overlap"],
                    "cluster_distance_threshold": new_domain_data["cluster_distance_threshold"],
                    "chunk_max_words": new_domain_data["chunk_max_words"],
                    "normalize_embeddings": new_domain_data["normalize_embeddings"],
                    "combine_embeddings": new_domain_data["combine_embeddings"],
                    "embedding_weight": new_domain_data["embedding_weight"],
                }

                domain_config = DomainConfig(**domain_config_data)
                self.sqlite_manager.insert_domain_config(domain_config, conn)

                conn.commit()
                self.logger.info("Dominio de conhecimento adicionado com sucesso", domain_name=domain_data["name"])

        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Erro ao adicionar novo domínio de conhecimento: {e}", exc_info=True)
            raise e

    def load_domain_config(self, domain_id: int) -> DomainConfig:
        """
        Carrega a configuração de um domínio de conhecimento.
        """
        self.logger.info("Carregando configuracao de dominio", domain_id=domain_id)
        try:
            with self.sqlite_manager.get_connection(control=True) as conn:
                domain_config = self.sqlite_manager.get_domain_config(conn=conn, domain_id=domain_id)
                if not domain_config:
                    self.logger.error("Configuracao de dominio nao encontrada", domain_id=domain_id)
                    raise ValueError(f"Configuracao de dominio nao encontrada: domain_id={domain_id}")
        
            return domain_config

        except Exception as e:
            self.logger.error(f"Erro ao carregar configuracao de dominio: {e}", exc_info=True)
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
                domain_dir = os.path.join(self.storage_base_path, treated_domain_name)

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
        old_dir = os.path.join(self.storage_base_path, f"{old_name_fs}")
        new_dir = os.path.join(self.storage_base_path, f"{new_name_fs}")
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