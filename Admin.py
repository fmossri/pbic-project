import streamlit as st
import os
import sys

from src.utils.logger import setup_logging
from gui.streamlit_utils import update_log_levels_callback


st.set_page_config(
    layout="wide",
    page_title="RAG Admin",
    page_icon="📚",
)

# --- Inicializa session state ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False 

# --- Titulo da Página ---
st.title("📚 Página do Admin do Sistema RAG")
st.write("Bem-vindo! Use o sidebar para navegar entre as seções de gerenciamento.")

# --- Configura o logging uma vez via função cacheada ---
@st.cache_resource
def initialize_logging_session():
    print(f"--- DEBUG Admin.py: Executando initialize_logging_session ---", file=sys.stderr)
    initial_debug_state = st.session_state.get('debug_mode', False)
    log_dir = os.path.join("logs", "gui")
    setup_logging(log_dir=log_dir, debug=initial_debug_state)
    return True

initialize_logging_session()

# --- Sidebar Debug Toggle --- 
st.sidebar.divider()

st.sidebar.toggle(
    "Debug Logging", 
    key="debug_mode",
    value=st.session_state.get('debug_mode', False), 
    help="Ativa o logging detalhado de nível DEBUG para o arquivo e INFO para o console. Requer atualização da página/interação após a alteração.",
    on_change=update_log_levels_callback
)
st.sidebar.divider()
# --------------------------

st.sidebar.success("Selecione uma seção acima.")
