import streamlit as st
import os
from src.data_ingestion import DataIngestionOrchestrator
from src.utils.domain_manager import DomainManager
from src.utils.logger import get_logger

# --- Logger ---
logger = get_logger(__name__, log_domain="gui")

# --- Page Configuration ---
st.set_page_config(
    page_title="Ingestão de Dados",
    layout="wide"
)
st.title("📥 Ingestão de Dados")

# --- Domain Management ---
try:
    domain_manager = DomainManager(log_domain="gui")
    domains = domain_manager.list_domains()
    domain_names = [domain.name for domain in domains] if domains else []
    logger.info("Lista de dominios carregada com sucesso.", domain_count=len(domain_names))
except Exception as e:
    logger.error("Erro ao carregar lista de dominios.", error=str(e), exc_info=True)
    st.error(f"Erro ao carregar lista de dominios: {e}")
    st.stop() # Stop execution if domains can't be loaded

# --- Ingestion Form ---
st.write("Selecione um domínio e forneça o caminho para o diretório contendo os arquivos a serem ingeridos.")

with st.form("data_ingestion_form"):
    selected_domain_name = st.selectbox(
        "Selecione o Domínio",
        options=domain_names,
        index=0 if domain_names else None, # Default to first domain if available
        help="Escolha um domínio de conhecimento para adicionar os documentos."
    )

    directory_path = st.text_input(
        "Caminho do Diretório",
        placeholder="e.g., /linux/path/documentos or C:\\\\Usuarios-Windows\\\\Voce\\\\Documentos",
        help="Insira o caminho completo para o diretório desejado.",
        key="directory_path_input" # Added key for potential state management later
    )

    submitted = st.form_submit_button("Iniciar")

# --- Form Submission Logic ---
if submitted:
    logger.info("Formulario de ingestao de dados enviado.", selected_domain=selected_domain_name, dir_path=directory_path)
    if not selected_domain_name:
        st.warning("Por favor, selecione um domínio.")
        logger.warning("Erro ao ingerir dados: Nenhum dominio selecionado.", selected_domain=selected_domain_name)
    elif not directory_path:
        st.warning("Por favor, insira um caminho de diretório.")
        logger.warning("Erro ao ingerir dados: Nenhum caminho de diretorio fornecido.", selected_domain=selected_domain_name)
    elif not os.path.isdir(directory_path):
        st.error(f"Erro: O caminho especificado '{directory_path}' não é um diretório válido ou não existe.")
        logger.error("Erro ao ingerir dados: Caminho de diretorio invalido.", selected_domain=selected_domain_name, dir_path=directory_path)
    
    st.success(f"Processo de ingestao de dados iniciado para o domínio '{selected_domain_name}' com o diretório '{directory_path}'.")
    logger.info("Iniciando ingestao de dados", selected_domain=selected_domain_name, dir_path=directory_path)
    selected_domain_object = next(
            (domain for domain in domains if domain.name == selected_domain_name), 
            None  # Return None if no match is found
    )
    orchestrator = DataIngestionOrchestrator()
    try:
        orchestrator.process_directory(directory_path=directory_path, domain_name=selected_domain_name)
        st.success(f"Dados ingeridos com sucesso de '{directory_path}' para o domínio '{selected_domain_name}'.")
        logger.info("Processo de ingestao de dados concluido com sucesso.", selected_domain=selected_domain_name)
    except Exception as e:
        st.error(f"Erro ao ingerir dados: {e}")
        logger.error("Erro ao ingerir dados.", selected_domain=selected_domain_name, error=str(e), exc_info=True)

# --- Display Note if No Domains ---
if not domain_names:
    st.warning("Nenhum domínio encontrado. Por favor, crie um domínio na seção 'Gerenciamento de Domínios' antes de ingerir dados.")
    logger.warning("Pagina de ingestao de dados carregada, mas nenhum dominio foi encontrado.") 