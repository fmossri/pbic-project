import streamlit as st
import os

from src.utils.logger import get_logger
from gui.streamlit_utils import update_log_levels_callback, get_domain_manager, initialize_logging_session, get_data_ingestion_orchestrator, load_configuration


st.set_page_config(
    page_title="Ingestão de Dados",
    layout="wide"
)

initialize_logging_session()
logger = get_logger(__name__, log_domain="gui")
config = load_configuration()
if config:
    domain_manager = get_domain_manager(config)
    orchestrator = get_data_ingestion_orchestrator(config)

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
    
    # --- Atualiza as configurações de processamento de embeddings ---
    config.embedding.device = "cuda" if embedding_device == "gpu" else embedding_device
    config.embedding.batch_size = embedding_batch_size

    # --- Atualiza as configurações do domínio se necessário ---
    try:
        if selected_domain_object.embeddings_model != config.embedding.model_name or selected_domain_object.faiss_index_type != config.vector_store.index_type:
            logger.info("Configuracoes de embeddings ou vector store nao correspondem aos valores do dominio. Atualizando configuracoes.", selected_domain=domain_name)
            config.embedding.model = selected_domain_object.embeddings_model
            config.vector_store.faiss.index_type = selected_domain_object.faiss_index_type
            try:
                orchestrator.update_config(config)
            except Exception as e:
                logger.error("Erro ao atualizar configuracoes.", selected_domain=domain_name, error=str(e), exc_info=True)
                st.error(f"Erro ao atualizar configuracoes: {e}")
                st.stop()


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
logger.info(f"--- DEBUG Data Ingestion: Rendering toggle, state is {st.session_state.get('debug_mode', 'Not Set Yet')} ---")
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

    # Get CUDA availability from session state
    cuda_is_present = st.session_state.get('cuda_available', False) 
    
    # Define options and default index based on CUDA availability
    embedding_options = ["cpu", "cuda"] if cuda_is_present else ["cpu"]
    default_device = config.embedding.device if config else "cpu"
    
    # Fallback if default is cuda but cuda is not available
    if default_device == "cuda" and not cuda_is_present:
        default_device = "cpu"
    
    # Find index in the *actual* options list
    try:
        default_index = embedding_options.index(default_device)
    except ValueError:
        default_index = 0 # Fallback if somehow default_device is not in options

    col1, col2 = st.columns(2)
    with col1:
        embedding_device = st.selectbox(
            "Dispositivo para Embedding", 
            options=embedding_options, 
            index=default_index,
            key="embedding_device_select",
            help="Dispositivo para calcular embeddings. 'cpu' ou 'gpu' (se disponível)."
        )
    with col2:
        embedding_batch_size = st.number_input(
            "Batch Size", 
            min_value=1, 
            step=1, 
            value=config.embedding.batch_size if config else 1,
            key="embedding_batch_size",
            help="Tamanho do lote para calcular embeddings."
        )

    submitted = st.form_submit_button("Iniciar", on_click=handle_submission)


if not domain_names:
    st.warning("Nenhum domínio encontrado. Por favor, crie um domínio na seção 'Gerenciamento de Domínios' antes de ingerir dados.")
    logger.warning("Pagina de ingestao de dados carregada, mas nenhum dominio foi encontrado.") 