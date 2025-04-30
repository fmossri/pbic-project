import streamlit as st
import os
import sys
import traceback

from src.data_ingestion import DataIngestionOrchestrator
from src.utils.domain_manager import DomainManager
from src.utils.logger import get_logger

from gui.streamlit_utils import update_log_levels_callback, get_domain_manager


st.set_page_config(
    page_title="Ingestão de Dados",
    layout="wide"
)

# --- Logger ---
logger = get_logger(__name__, log_domain="gui")

@st.cache_resource
def get_data_ingestion_orchestrator():
    logger.info("Creating DataIngestionOrchestrator instance (cached)")
    try:
        return DataIngestionOrchestrator()
    except Exception as e:
        logger.error(f"Failed to create DataIngestionOrchestrator instance: {e}", exc_info=True)
        raise SystemExit(f"Failed to initialize DataIngestionOrchestrator: {e}. Check logs.")

# --- Inicialização de Recursos
domain_manager = get_domain_manager()
orchestrator = get_data_ingestion_orchestrator()

# --- Função de Callback para envio do formulario de Ingestão ---
def handle_submission():
    domain_name = st.session_state.selected_domain_name_input
    dir_path = st.session_state.directory_path_input
    
    logger.info("Formulario de ingestao de dados enviado (via callback).", selected_domain=domain_name, dir_path=dir_path)

    if not domain_name:
        st.warning("Por favor, selecione um domínio.")
        logger.warning("Erro ao ingerir dados: Nenhum dominio selecionado.", selected_domain=domain_name)
        return
    if not dir_path:
        st.warning("Por favor, insira um caminho de diretório.")
        logger.warning("Erro ao ingerir dados: Nenhum caminho de diretorio fornecido.", selected_domain=domain_name)
        return
    if not os.path.isdir(dir_path):
        st.error(f"Erro: O caminho especificado '{dir_path}' não é um diretório válido ou não existe.")
        logger.error("Erro ao ingerir dados: Caminho de diretorio invalido.", selected_domain=domain_name, dir_path=dir_path)
        return

    # Encontra o objeto do domínio
    selected_domain_object = next(
            (domain for domain in domains if domain.name == domain_name), 
            None 
    )
    if not selected_domain_object:
        st.error(f"Erro interno: Domínio '{domain_name}' selecionado mas não encontrado na lista.")
        logger.error("Inconsistencia: Dominio selecionado mas nao encontrado na lista.", selected_domain=domain_name)
        return 

    st.info(f"Iniciando processo de ingestão para o domínio '{domain_name}' com o diretório '{dir_path}'. Aguarde...") # Give initial feedback
    logger.info("Iniciando ingestao de dados", selected_domain=domain_name, dir_path=dir_path)
    

    try:
        results = orchestrator.process_directory(directory_path=dir_path, domain_name=domain_name) 
        
        if results["processed_files"] == 0:
            st.warning(f"Nenhum arquivo processado. Arquivos invalidos ou duplicados.")
            logger.warning("Nenhum arquivo processado. Arquivos invalidos ou duplicados.", dir_path=dir_path, selected_domain=domain_name)
        else:
            st.success(f"{results['processed_files']} arquivos ingeridos de '{dir_path}' por '{domain_name}'. {results['duplicate_files']} duplicatas; {results['invalid_files']} arquivos invalidos.")
            logger.info("Processo de ingestao de dados concluido com sucesso.", selected_domain=domain_name)
        
        st.session_state.directory_path_input = "" 
        
    except Exception as e:
        st.error(f"Erro ao ingerir dados: {e}")
        logger.error("Erro durante a ingestao de dados.", selected_domain=domain_name, error=str(e), exc_info=True)

# --- Inicializa session state ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
# ----------------------------------------------

# --- Titulo da Página ---
st.title("📥 Ingestão de Dados") 

# --- Sidebar Debug Toggle --- 
st.sidebar.divider()
print(f"--- DEBUG Data Ingestion: Rendering toggle, state is {st.session_state.get('debug_mode', 'Not Set Yet')} ---", file=sys.stderr)
st.sidebar.toggle(
    "Debug Logging", 
    key="debug_mode", 
    value=st.session_state.get('debug_mode', False), 
    help="Enable detailed DEBUG level logging...",
    on_change=update_log_levels_callback 
)
st.sidebar.divider()

# --- Listagem dos Domínios ---
try:
    domains = domain_manager.list_domains()
    domain_names = [domain.name for domain in domains] if domains else []
    logger.info("Lista de dominios carregada com sucesso.", domain_count=len(domain_names))
except Exception as e:
    logger.error("Erro ao carregar lista de dominios.", error=str(e), exc_info=True)
    st.error(f"Erro ao carregar lista de dominios: {e}")
    st.stop()

# --- Formulario de Ingestão ---
st.write("Selecione um domínio e forneça o caminho para o diretório contendo os arquivos a serem ingeridos.")

with st.form("data_ingestion_form"):
    selected_domain_name = st.selectbox(
        "Selecione o Domínio",
        options=domain_names,
        index=0 if domain_names else None,
        help="Escolha um domínio de conhecimento para adicionar os documentos.",
        key="selected_domain_name_input"
    )

    directory_path = st.text_input(
        "Caminho do Diretório",
        placeholder="e.g., /linux/path/documentos or C:\\\\Usuarios-Windows\\\\Voce\\\\Documentos",
        help="Insira o caminho completo para o diretório desejado.",
        key="directory_path_input" 
    )

    submitted = st.form_submit_button("Iniciar", on_click=handle_submission)


if not domain_names:
    st.warning("Nenhum domínio encontrado. Por favor, crie um domínio na seção 'Gerenciamento de Domínios' antes de ingerir dados.")
    logger.warning("Pagina de ingestao de dados carregada, mas nenhum dominio foi encontrado.") 