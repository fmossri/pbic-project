import streamlit as st
import logging
import traceback
import os

from typing import Optional
from src.utils.logger import get_logger, setup_logging

# Import config loader and components
from src.config.config_manager import get_config, ConfigurationError
from src.config.models import AppConfig
from src.utils import DomainManager, SQLiteManager 
from src.data_ingestion import DataIngestionOrchestrator
from src.query_processing import QueryOrchestrator

logger = get_logger(__name__, log_domain="streamlit_utils")

# --- Function to load and cache configuration --- 
@st.cache_resource
def load_configuration() -> Optional[AppConfig]:
    """Loads the application configuration using get_config and caches it."""
    logger.info("Attempting to load application configuration (cached)")
    try:
        config = get_config()
        logger.info("Configuration loaded successfully.")
        return config
    except ConfigurationError as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        st.error(f"Erro Crítico ao Carregar Configuração: {e}\n\nVerifique o arquivo 'config.toml' e reinicie a aplicação.")
        st.code(traceback.format_exc())
        st.stop() # Stop execution if config fails to load
        return None # Should not be reached due to st.stop()
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during configuration loading: {e}", exc_info=True)
        st.error(f"Erro Inesperado ao Carregar Configuração: {e}")
        st.code(traceback.format_exc())
        st.stop()
        return None


# --- Modify component getters to use the loaded config --- 

@st.cache_resource 
def get_domain_manager(_config: AppConfig) -> Optional[DomainManager]:
    """Creates a DomainManager instance using the loaded configuration."""
    logger.info("Creating DomainManager instance (cached)")
    sqlite_manager = SQLiteManager(_config.system)
    if not _config:
        logger.error("Cannot create DomainManager: Configuration object is None.")
        # Config loading error should have already stopped the app via load_configuration
        return None 
    try:
        # Pass the system config part to DomainManager
        return DomainManager(config=_config.system, sqlite_manager=sqlite_manager)
    except Exception as e:
        logger.error(f"Erro ao criar instancia do DomainManager: {e}", exc_info=True)
        st.error(f"Erro ao inicializar o DomainManager: {e}")
        st.code(traceback.format_exc())
        st.stop()
        return None

@st.cache_resource
def get_data_ingestion_orchestrator(_config: AppConfig) -> Optional[DataIngestionOrchestrator]:
    """Creates a DataIngestionOrchestrator instance using the loaded configuration."""
    logger.info("Creating DataIngestionOrchestrator instance (cached)")
    if not _config:
        logger.error("Cannot create DataIngestionOrchestrator: Configuration object is None.")
        return None
    try:
        # Pass the full config
        return DataIngestionOrchestrator(config=_config)
    except Exception as e:
        logger.error(f"Failed to create DataIngestionOrchestrator instance: {e}", exc_info=True)
        st.error(f"Erro ao inicializar o DataIngestionOrchestrator: {e}")
        st.code(traceback.format_exc())
        st.stop()
        return None


@st.cache_resource
def get_query_orchestrator(_config: AppConfig) -> Optional[QueryOrchestrator]:
    """Creates a QueryOrchestrator instance using the loaded configuration."""
    logger.info("Creating QueryOrchestrator instance (cached)")
    if not _config:
        logger.error("Cannot create QueryOrchestrator: Configuration object is None.")
        return None
    try:
        # Pass the full config
        return QueryOrchestrator(config=_config)
    except Exception as e:
        logger.error(f"Failed to create QueryOrchestrator instance: {e}", exc_info=True)
        st.error(f"Erro ao inicializar o QueryOrchestrator: {e}")
        st.code(traceback.format_exc())
        st.stop()
        return None


@st.cache_resource
def initialize_logging_session():
    initial_debug_state = st.session_state.get('debug_mode', False)
    log_dir = os.path.join("logs", "gui")
    setup_logging(log_dir=log_dir, debug=initial_debug_state)
    print(f"--- Logger inicializado ---")
    return True

# --- Função de callback para atualizar os níveis de log --- 
def update_log_levels_callback():
    # Usa um logger temporario para o callback da mensagem
    callback_logger = get_logger("AdminCallback", log_domain="gui_callback")
    callback_logger.info("Debug toggle alterado, atualizando niveis de log...")
    
    try:
        is_debug = st.session_state.get('debug_mode', False)
        root_logger = logging.getLogger()
        
        # Determina os niveis de destino baseados no estado
        file_level = logging.DEBUG if is_debug else logging.INFO
        console_level = logging.INFO if is_debug else logging.WARNING
        root_level = min(file_level, console_level) # Root precisa ser o mais verboso
        
        callback_logger.info(f"Definindo niveis de log: Root={root_level}, File={file_level}, Console={console_level}")
        print(f"--- DEBUG debug_mode: {is_debug} ---")

        # Define o nivel do root logger
        root_logger.setLevel(root_level)
        
        # Define os niveis dos handlers
        handler_found = False
        for handler in root_logger.handlers:
            handler_found = True
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setLevel(file_level)
                callback_logger.debug(f"Define nivel FileHandler para {file_level}")
            elif isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_level)
                callback_logger.debug(f"Define nivel StreamHandler para {console_level}")
            else:
                callback_logger.warning(f"Tipo de handler desconhecido encontrado: {type(handler)}")
        
        if not handler_found:
             callback_logger.warning("Nenhum handler encontrado no root logger durante o callback.")


             
    except Exception as e:
        callback_logger.error(f"Error in update_log_levels_callback: {e}", exc_info=True)

