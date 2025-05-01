import streamlit as st
import os

from src.utils.logger import get_logger
from gui.streamlit_utils import update_log_levels_callback, get_domain_manager, initialize_logging_session, get_query_orchestrator, load_configuration


st.set_page_config(
    page_title="Query Interface",
    layout="wide"
)

initialize_logging_session()
logger = get_logger(__name__, log_domain="gui")

config = load_configuration()
if config:
    domain_manager = get_domain_manager(config)
    orchestrator = get_query_orchestrator(config)

# --- Inicializa session state ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if 'selected_query_domain' not in st.session_state:
    st.session_state.selected_query_domain = "Auto" 

# --- Titulo da Página ---
st.title("💬 Query Interface")

# --- Configuração de seleção de domínio ---
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
    # Usa session state para manter o registro do dominio selecionado
    if 'selected_query_domain' not in st.session_state:
        st.session_state.selected_query_domain = "Auto" # Default to Auto

    selected_domain = st.radio(
        "Selecione o domínio de busca:",
        options=domain_options,
        key="selected_query_domain", # Gera persistência da seleção através dos reruns
        help="Escolha um domínio específico ou 'Auto' para seleção automática com base na query."
    )
    st.divider()
    
    # --- Sidebar Debug Toggle --- 
    st.sidebar.divider()
    logger.info(f"--- Renderizando toggle, debug_mode = {st.session_state.get('debug_mode', 'Nao definido ainda')} ---")
    st.sidebar.toggle(
        "Debug Logging", 
        key="debug_mode",
        value=st.session_state.get('debug_mode', False),
        help="Enable detailed DEBUG level logging to file and INFO to console.",
        on_change=update_log_levels_callback
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
chat_container = st.container(height=400, border=False)
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Pergunte aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Exibe a mensagem do usuario no container de mensagens de chat imediatamente
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Prepara e exibe a resposta do assistente
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            domain_target_message = f"Pesquisando no domínio: **{selected_domain}**"
            message_placeholder.markdown(f"""{domain_target_message}

Pensando...""")
            logger.info(f"User query received. Target: {selected_domain}", user_query=prompt)
            
            try:
                if selected_domain == "Auto":
                    all_valid_domains = [domain.name for domain in all_domains if domain.name in valid_domain_names]
                    logger.debug("Chamando query_llm em modo automatico (domain_names=None)")
                    response_data = orchestrator.query_llm(prompt, domain_names=all_valid_domains) 
                else:
                    logger.debug(f"Chamando query_llm com dominio especifico: {selected_domain}")
                    response_data = orchestrator.query_llm(prompt, domain_names=[selected_domain])

                if isinstance(response_data, dict) and 'answer' in response_data:
                    agent_answer = response_data['answer']
                else:
                    logger.error("Formato de resposta inesperado do QueryOrchestrator", response_data=response_data)
                    agent_answer = "Erro: Formato de resposta inesperado do backend."
            except Exception as e:
                logger.error(f"Erro ao chamar orchestrator.query_llm: {e}", exc_info=True)
                agent_answer = f"Ocorreu um erro ao processar sua pergunta: {e}"

            message_placeholder.markdown(agent_answer)
            
    # Adiciona a resposta do assistente ao histórico de chat
    st.session_state.messages.append({"role": "assistant", "content": agent_answer})

# --- Exibe uma nota se nenhum dominio for encontrado ---
if not valid_domain_names:
     st.sidebar.warning("Nenhum dominio com arquivos DB existentes encontrado. Por favor, ingerir dados primeiro.")
     if "messages" not in st.session_state or not st.session_state.messages:
         st.info("Nenhum dominio disponível para consulta. Por favor, use a seção 'Ingestão de dados' para processar documentos em um domínio primeiro.") 