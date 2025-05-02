import streamlit as st
import logging
import traceback
import os

from typing import Optional
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
        return DomainManager(config=_config.system, sqlite_manager=sqlite_manager)
    except Exception as e:
        logger.error(f"Erro ao criar instancia do DomainManager: {e}", exc_info=True)
        st.error(f"Erro ao inicializar o DomainManager: {e}")
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

