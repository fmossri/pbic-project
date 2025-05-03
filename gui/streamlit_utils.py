import streamlit as st
import logging
import traceback
import os
import torch

from typing import Optional, List
from src.models import DocumentFile
from src.utils.logger import get_logger, setup_logging

from src.config.config_manager import ConfigurationError, ConfigManager
from src.config.models import AppConfig
from src.utils import DomainManager, SQLiteManager 
from src.data_ingestion import DataIngestionOrchestrator
from src.query_processing import QueryOrchestrator

logger = get_logger(__name__, log_domain="streamlit_utils")

@st.cache_resource
def get_config_manager() -> ConfigManager:
    """Cria e cacheia uma instância ConfigManager para o caminho de configuração padrão."""
    logger.info("Criando e cacheando instância ConfigManager para o caminho padrão.")
    return ConfigManager()

@st.cache_data # Cacheia os dados carregados
def load_configuration() -> Optional[AppConfig]:
    """Carrega a configuração da aplicação usando o ConfigManager cacheado e cacheia os dados."""
    logger.info("Tentando carregar a configuração da aplicação (dados cacheados)")
    try:
        manager = get_config_manager()
        config = manager.load_config()
        logger.info("Configuração carregada com sucesso.")
        
        # Check CUDA availability and store in session state
        cuda_status = torch.cuda.is_available()
        st.session_state['cuda_available'] = cuda_status
        logger.info(f"CUDA Availability Check: {cuda_status}")
        
        return config
    except ConfigurationError as e:
        logger.error(f"Falha ao carregar configuração: {e}", exc_info=True)
        st.error(f"Erro Crítico ao Carregar Configuração: {e}\n\nVerifique o arquivo 'config.toml' e reinicie a aplicação.")
        st.code(traceback.format_exc())
        st.stop()
        return None
    except Exception as e:
        logger.error(f"Um erro inesperado ocorreu durante o carregamento da configuração: {e}", exc_info=True)
        st.error(f"Erro Inesperado ao Carregar Configuração: {e}")
        st.code(traceback.format_exc())
        st.stop()
        return None

@st.cache_resource 
def get_domain_manager(_config: AppConfig) -> Optional[DomainManager]:
    """Cria uma instância DomainManager usando a configuração carregada."""
    logger.info("Criando instância DomainManager (cacheada)")
    sqlite_manager = SQLiteManager(_config.system)
    if not _config:
        logger.error("Não é possível criar DomainManager: Objeto de configuração é None.")
        return None 
    try:
        return DomainManager(config=_config, sqlite_manager=sqlite_manager)
    except Exception as e:
        logger.error(f"Erro ao criar instancia do DomainManager: {e}", exc_info=True)
        st.error(f"Erro ao inicializar o DomainManager: {e}")
        st.code(traceback.format_exc())
        st.stop()
        return None

@st.cache_resource
def get_sqlite_manager(_config: AppConfig) -> Optional[SQLiteManager]:
    """Cria uma instância SQLiteManager usando a configuração carregada e cacheia a instância."""
    logger.info("Criando instância SQLiteManager (cacheada)")
    if not _config:
        logger.error("Não é possível criar SQLiteManager: Objeto de configuração é None.")
        return None
    try:
        return SQLiteManager(_config.system)
    except Exception as e:
        logger.error(f"Erro ao criar instancia do SQLiteManager: {e}", exc_info=True)
        st.error(f"Erro ao inicializar o SQLiteManager: {e}")
        st.code(traceback.format_exc())
        st.stop()
        return None

@st.cache_resource
def get_data_ingestion_orchestrator(_config: AppConfig) -> Optional[DataIngestionOrchestrator]:
    """Cria uma instância DataIngestionOrchestrator usando a configuração carregada."""
    logger.info("Criando instância DataIngestionOrchestrator (cacheada)")
    if not _config:
        logger.error("Não é possível criar DataIngestionOrchestrator: Objeto de configuração é None.")
        return None
    try:
        return DataIngestionOrchestrator(config=_config)
    except Exception as e:
        logger.error(f"Falha ao criar instância do DataIngestionOrchestrator: {e}", exc_info=True)
        st.error(f"Erro ao inicializar o DataIngestionOrchestrator: {e}")
        st.code(traceback.format_exc())
        st.stop()
        return None


@st.cache_resource
def get_query_orchestrator(_config: AppConfig) -> Optional[QueryOrchestrator]:
    """Cria uma instância QueryOrchestrator usando a configuração carregada."""
    logger.info("Criando instância QueryOrchestrator (cacheada)")
    if not _config:
        logger.error("Não é possível criar QueryOrchestrator: Objeto de configuração é None.")
        return None
    try:
        return QueryOrchestrator(config=_config)
    except Exception as e:
        logger.error(f"Falha ao criar instância do QueryOrchestrator: {e}", exc_info=True)
        st.error(f"Erro ao inicializar o QueryOrchestrator: {e}")
        st.code(traceback.format_exc())
        st.stop()
        return None


@st.cache_resource
def initialize_logging_session():
    """Inicializa o logging para a sessão Streamlit, configurando diretório e nível inicial."""
    initial_debug_state = st.session_state.get('debug_mode', False)
    log_dir = os.path.join("logs", "gui")
    setup_logging(log_dir=log_dir, debug=initial_debug_state)
    logger.info("--- Logger inicializado ---")
    return True

def update_log_levels_callback():
    """Callback para atualizar os níveis de log dinamicamente baseado no toggle de debug."""
    # Usa um logger temporário para a mensagem de callback
    callback_logger = get_logger("DebugCallback", log_domain="gui_callback")
    callback_logger.info("Toggle de debug alterado, atualizando níveis de log...")
    
    try:
        is_debug = st.session_state.get('debug_mode', False)
        root_logger = logging.getLogger()
        
        file_level = logging.DEBUG if is_debug else logging.INFO
        console_level = logging.INFO if is_debug else logging.WARNING
        root_level = min(file_level, console_level) # Root precisa ser o mais verboso
        
        callback_logger.info(f"Definindo níveis de log: Root={logging.getLevelName(root_level)}, File={logging.getLevelName(file_level)}, Console={logging.getLevelName(console_level)}")
        logger.info(f"--- Nível de debug alterado: {is_debug} ---")

        root_logger.setLevel(root_level)
    
        handler_found = False
        for handler in root_logger.handlers:
            handler_found = True
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setLevel(file_level)
                callback_logger.debug(f"Define nível FileHandler para {logging.getLevelName(file_level)}")
            elif isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_level)
                callback_logger.debug(f"Define nível StreamHandler para {logging.getLevelName(console_level)}")
            else:
                callback_logger.warning(f"Tipo de handler desconhecido encontrado: {type(handler)}")
        
        if not handler_found:
             callback_logger.warning("Nenhum handler encontrado no root logger durante o callback.")
             
    except Exception as e:
        callback_logger.error(f"Erro em update_log_levels_callback: {e}", exc_info=True)

# --- Domain/Document Specific Helpers ---

def get_domain_documents(domain_manager, domain_name: str) -> List[DocumentFile]:
    """Retorna a lista de DocumentFile para um domínio específico."""
    logger = get_logger(__name__, log_domain="gui/utils")
    if not domain_manager or not domain_name:
        logger.warning("Tentativa de listar documentos com domain_manager ou domain_name inválidos.")
        return []
    try:
        logger.info(f"Listando documentos para o domínio: {domain_name}")
        documents = domain_manager.list_domain_documents(domain_name)
        logger.info(f"Encontrados {len(documents)} documentos para {domain_name}.")
        return documents if documents else []
    except FileNotFoundError as fnf:
        st.warning(f"Banco de dados do domínio '{domain_name}' não encontrado. Execute a ingestão primeiro. Detalhes: {fnf}")
        logger.warning(f"DB não encontrado ao listar documentos para {domain_name}: {fnf}")
        return []
    except ValueError as ve:
        st.error(f"Erro ao buscar domínio '{domain_name}': {ve}")
        logger.error(f"ValueError ao listar documentos para {domain_name}: {ve}")
        return []
    except Exception as e:
        st.error(f"Erro inesperado ao listar documentos para o domínio '{domain_name}': {e}")
        logger.error(f"Erro inesperado ao listar documentos para {domain_name}: {e}", exc_info=True)
        st.code(traceback.format_exc())
        return []

def delete_document_from_domain(domain_manager, domain_name: str, document_file: DocumentFile) -> bool:
    """Deleta um DocumentFile de um domínio específico e atualiza a contagem."""
    logger = get_logger(__name__, log_domain="gui/utils")
    if not domain_manager or not domain_name or not document_file or not document_file.id:
        logger.error("Tentativa de deletar documento com parâmetros inválidos.", 
                      domain_name=domain_name, doc_id=document_file.id if document_file else None)
        st.error("Erro interno: Informações inválidas para deletar o documento.")
        return False
    
    try:
        # 1. Obter detalhes do domínio (para pegar o db_path e a contagem atual)
        with domain_manager.sqlite_manager.get_connection(control=True) as control_conn:
            domain_list = domain_manager.sqlite_manager.get_domain(control_conn, domain_name)
            if not domain_list:
                raise ValueError(f"Domínio de controle '{domain_name}' não encontrado para exclusão do documento.")
            domain = domain_list[0]
            current_doc_count = domain.total_documents

        # 2. Conectar ao DB específico do domínio e deletar o registro do documento
        logger.info(f"Tentando deletar documento ID {document_file.id} ({document_file.name}) do domínio '{domain_name}' no DB: {domain.db_path}")
        with domain_manager.sqlite_manager.get_connection(db_path=domain.db_path) as domain_conn:
            domain_manager.sqlite_manager.begin(domain_conn)
            domain_manager.sqlite_manager.delete_document_file(document_file, domain_conn)
            domain_conn.commit()
            logger.info(f"Documento ID {document_file.id} deletado com sucesso do DB do domínio.")

        # 3. Atualizar a contagem de documentos no DB de controle
        new_doc_count = max(0, current_doc_count - 1) # Evitar contagem negativa
        logger.info(f"Atualizando contagem de documentos para o domínio '{domain_name}' de {current_doc_count} para {new_doc_count}.")
        domain_manager.update_domain_details(domain_name, {"total_documents": new_doc_count})
        logger.info(f"Contagem de documentos atualizada com sucesso.")
        
        return True

    except FileNotFoundError:
        st.error(f"Erro: Banco de dados do domínio '{domain_name}' não encontrado em {domain.db_path}. Não foi possível deletar o documento.")
        logger.error(f"DB do domínio não encontrado durante exclusão do documento: {domain.db_path}")
        return False
    except ValueError as ve:
        st.error(f"Erro ao deletar documento: {ve}")
        logger.error(f"ValueError durante exclusão do documento {document_file.id} do domínio {domain_name}: {ve}")
        # Rollback não é necessário pois a transação do domain_conn já foi comitada ou falhou antes
        return False
    except Exception as e:
        st.error(f"Erro inesperado ao deletar o documento '{document_file.name}': {e}")
        logger.error(f"Erro inesperado durante exclusão do documento {document_file.id} do domínio {domain_name}: {e}", exc_info=True)
        st.code(traceback.format_exc())
        return False

