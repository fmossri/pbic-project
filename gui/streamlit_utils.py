import streamlit as st
import logging
import traceback

from src.utils.logger import get_logger
from src.utils.domain_manager import DomainManager

logger = get_logger(__name__, log_domain="streamlit_utils")

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

# --- Função de cache para criar o DomainManager ---
@st.cache_resource 
def get_domain_manager():
    logger.info("Criando instancia do DomainManager (cacheada)")
    try:
        return DomainManager()
    except Exception as e:
        logger.error(f"Erro ao criar instancia do DomainManager: {e}", exc_info=True)
        st.error(f"Erro ao inicializar o DomainManager: {e}")
        st.code(traceback.format_exc())
        st.stop()