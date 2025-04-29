# Placeholder for Query Interface Page 

import streamlit as st
import os
from src.utils.domain_manager import DomainManager
from src.utils.logger import get_logger
from src.query_processing import QueryOrchestrator 
import sys
from Admin import update_log_levels_callback 
import traceback

# --- Page Configuration (at the top) ---
st.set_page_config(
    page_title="Query Interface",
    layout="wide"
)

# --- Logger --- 
logger = get_logger(__name__, log_domain="gui")

# --- Cached Resource Initialization (MUST HAPPEN BEFORE st.set_page_config) ---
@st.cache_resource
def get_domain_manager_instance():
    logger.info("Creating DomainManager instance (cached)")
    try:
        return DomainManager(log_domain="gui")
    except Exception as e:
        logger.error(f"Failed to create DomainManager instance: {e}", exc_info=True)
        # Logged the error, now exit script
        raise SystemExit(f"Failed to initialize DomainManager: {e}. Check logs.")

@st.cache_resource
def get_query_orchestrator_instance():
    logger.info("Creating QueryOrchestrator instance (cached)")
    try:
        return QueryOrchestrator()
    except Exception as e:
        logger.error(f"Failed to create QueryOrchestrator instance: {e}", exc_info=True)
        # Logged the error, now exit script
        raise SystemExit(f"Failed to initialize QueryOrchestrator: {e}. Check logs.")

domain_manager = get_domain_manager_instance()
orchestrator = get_query_orchestrator_instance()
# -------------------------------------------------------------------------

st.title("💬 Query Interface")

# --- Initialize Session State (if not exists) ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
# Initialize other session state variables after page config
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if 'selected_query_domain' not in st.session_state:
    st.session_state.selected_query_domain = "Auto" 
# -----------------------------------------------------

# --- Domain Selection Setup ---
try:
    
    all_domains = domain_manager.list_domains()
    
    # Filtra apenas os domínios com um arquivo DB
    valid_domain_names = []
    if all_domains:
        for domain in all_domains:
            # Verifica se o dominio tem um arquivo DB
            if domain.db_path and os.path.exists(domain.db_path):
                # Adiciona o dominio a lista de dominios validos
                valid_domain_names.append(domain.name)

            # Se o dominio tem um db_path mas nao tem um arquivo DB
            elif domain.db_path:
                 logger.info(f"Dominio '{domain.name}' listado mas armazenamento nao inicializado em: {domain.db_path}")

            # Se o dominio nao tem um db_path, declara erro de registro
            else:
                 logger.error(f"Dominio '{domain.name}' nao tem um db_path definido: Erro de registro. Necessario refazer ou remover o registro do dominio.")
        logger.info(f"Encontrados {len(valid_domain_names)} dominios com arquivos DB existentes.")
    else:
        logger.info("Nenhum dominio encontrado no banco de controle.")
        
    # Opcoes para o botao de seleção
    domain_options = ["Auto"] + sorted(valid_domain_names)
    
except Exception as e:
    logger.error("Erro ao carregar ou filtrar dominios para o sidebar.", error=str(e), exc_info=True)
    st.error(f"Erro ao carregar a lista de dominios: {e}")
    # Define opções padrão se o carregamento falhar
    domain_options = ["Auto"] 
    valid_domain_names = []


# --- Sidebar de seleção de dominio ---
with st.sidebar:
    st.header("Opções de busca")
    # UsA session state para manter o registro do dominio selecionado
    if 'selected_query_domain' not in st.session_state:
        st.session_state.selected_query_domain = "Auto" # Default to Auto

    selected_domain = st.radio(
        "Selecione o domínio de busca:",
        options=domain_options,
        key="selected_query_domain", # Gera persistência da seleção através dos reruns
        help="Escolha um domínio específico ou 'Auto' para seleção automática com base na query."
    )
    st.divider()
    
    # --- Add Debug Toggle to Sidebar ---
    # --- Sidebar Debug Toggle --- 
    st.sidebar.divider()
    # Print state just before rendering toggle
    print(f"--- DEBUG Query Interface: Rendering toggle, state is {st.session_state.get('debug_mode', 'Not Set Yet')} ---", file=sys.stderr)
    st.sidebar.toggle(
        "Debug Logging", 
        key="debug_mode", # Must match the session state key
        value=st.session_state.get('debug_mode', False), # Set initial value from state
        help="Enable detailed DEBUG level logging to file and INFO to console.",
        on_change=update_log_levels_callback # Add the callback here
    )
    st.sidebar.divider()
    # --------------------------
    # ---------------------------------
    
    #TODO: Adicionar outras opcoes do sidebar aqui mais tarde (e.g., numero de resultados k)


# --- Gerenciamento do historico de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Inicializa o historico de chat

# --- Exibe o historico de chat ---
st.write("Chat History:")
chat_container = st.container(height=400, border=False) # Container para o historico de chat
with chat_container:
    # Usa st.chat_message para exibir mensagens com 'roles'
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Pergunte aqui..."):
    # 1. Adiciona a mensagem do usuario ao historico de chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Exibe a mensagem do usuario no container de mensagens de chat imediatamente
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # 3. Prepara e exibe a resposta do assistente
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Usar placeholder para efeito de streaming
            
            domain_target_message = f"Pesquisando no domínio: **{selected_domain}**"
            message_placeholder.markdown(f"""{domain_target_message}

Pensando...""")
            logger.info(f"User query received. Target: {selected_domain}", user_query=prompt)
            
            try:
                if selected_domain == "Auto":
                    all_valid_domains = [domain.name for domain in all_domains if domain.name in valid_domain_names]
                    # Pass None to query_llm when Auto is selected, so it performs selection
                    logger.debug("Calling query_llm with Auto mode (domain_names=None)")
                    response_data = orchestrator.query_llm(prompt, domain_names=all_valid_domains) 
                else:
                    # Pass the selected domain name as a single-element list
                    logger.debug(f"Calling query_llm with specific domain: {selected_domain}")
                    response_data = orchestrator.query_llm(prompt, domain_names=[selected_domain])

                # Extract answer from the returned dictionary
                if isinstance(response_data, dict) and 'answer' in response_data:
                    agent_answer = response_data['answer']
                else:
                    # Log the unexpected structure
                    logger.error("Formato de resposta inesperado do QueryOrchestrator", response_data=response_data)
                    agent_answer = "Erro: Formato de resposta inesperado do backend." # Provide user feedback
            except Exception as e:
                # Catch errors during the query_llm call itself
                logger.error(f"Error calling orchestrator.query_llm: {e}", exc_info=True)
                agent_answer = f"Ocorreu um erro ao processar sua pergunta: {e}"

            # Exibe a resposta do assistente (ou mensagem de erro)
            message_placeholder.markdown(agent_answer)
            
    # 4. Adiciona a resposta do assistente ao historico de chat
    st.session_state.messages.append({"role": "assistant", "content": agent_answer})

# --- Exibe uma nota se nenhum dominio for encontrado ---
if not valid_domain_names:
     st.sidebar.warning("Nenhum dominio com arquivos DB existentes encontrado. Por favor, ingerir dados primeiro.")
     if "messages" not in st.session_state or not st.session_state.messages:
         st.info("Nenhum dominio disponível para consulta. Por favor, use a seção 'Ingestão de dados' para processar documentos em um domínio primeiro.") 