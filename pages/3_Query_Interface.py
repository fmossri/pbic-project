import streamlit as st
import os
import copy

from pydantic import ValidationError

from src.utils.logger import get_logger
from gui.streamlit_utils import (
    update_log_levels_callback, 
    get_domain_manager, 
    initialize_logging_session, 
    get_query_orchestrator, 
    load_configuration,
    get_config_manager
)
from src.config.config_manager import ConfigurationError
from src.config.models import LLMConfig, QueryConfig

st.set_page_config(
    page_title="Query Interface",
    layout="wide"
)

initialize_logging_session()
logger = get_logger(__name__, log_domain="gui")

config = load_configuration()
manager = get_config_manager()

if config:
    domain_manager = get_domain_manager(config)
    orchestrator = get_query_orchestrator(config)
else:
    # Trata o caso onde a configuração falha para carregar early
    st.error("Failed to load application configuration. Cannot initialize components.")
    st.stop() # Para a execução do script se a configuração é essencial

# --- Armazena configuração do LLM original para comparação ---
if 'original_config' not in st.session_state or st.session_state.original_config != config:
    # Armazena uma cópia profunda para evitar modificações que afetam a configuração original
    st.session_state.original_config = copy.deepcopy(config)
    logger.debug("Stored/Updated original_config in session state.")

# --- Inicializa session state ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if 'selected_query_domain' not in st.session_state:
    st.session_state.selected_query_domain = "Auto" 
if 'confirming_config_reset' not in st.session_state:
    st.session_state.confirming_config_reset = False
if 'original_llm_config' not in st.session_state:
    st.session_state.original_llm_config = copy.deepcopy(config.llm)
if 'original_query_config' not in st.session_state:
    st.session_state.original_query_config = copy.deepcopy(config.query)
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
                 logger.debug(f"Dominio '{domain.name}' listado mas armazenamento nao inicializado em: {domain.db_path}")

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

        
    # --- Sidebar Debug Toggle --- 
    logger.info(f"--- Renderizando toggle, debug_mode = {st.session_state.get('debug_mode', 'Nao definido ainda')} ---")
    st.sidebar.toggle(
        "Debug Logging", 
        key="debug_mode",
        value=st.session_state.get('debug_mode', False),
        help="Enable detailed DEBUG level logging to file and INFO to console.",
        on_change=update_log_levels_callback
    )
    st.sidebar.divider()
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

    # --- Config Widgets ---
    st.header("Configuração de busca")
    st.caption("Ajustes serão atualizados automaticamente ao enviar uma query.")
    
    st.header("Top-K Documentos")
    query_retrieval_k = st.number_input("Top-K", min_value=1, step=1, value=config.query.retrieval_k, key="sidebar_query_retrieval_k")
    
    st.header("Parâmetros do LLM")
    llm_model_repo_id = st.text_input("Model Repo ID", value=config.llm.model_repo_id, key="sidebar_llm_model_repo_id")
    llm_prompt_template = st.text_area("Prompt Template", value=config.llm.prompt_template, key="sidebar_llm_prompt_template", height=100)
    llm_max_new_tokens = st.number_input("Max New Tokens", min_value=1, step=1, value=config.llm.max_new_tokens, key="sidebar_llm_max_new_tokens")
    llm_temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.01, value=config.llm.temperature, key="sidebar_llm_temperature")
    llm_top_p = st.slider("Top P", min_value=0.0, max_value=1.0, step=0.01, value=config.llm.top_p or 0.9, key="sidebar_llm_top_p") 
    llm_top_k = st.number_input("Top K", min_value=0, step=1, value=config.llm.top_k or 50, key="sidebar_llm_top_k")
    llm_repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, step=0.01, value=config.llm.repetition_penalty or 1.0, key="sidebar_llm_repetition_penalty")
    # --- Reset button --- 
    if not st.session_state.confirming_config_reset:
        if st.button("Reset Default", type="secondary", use_container_width=True, key="trigger_reset_button"):
            st.session_state.confirming_config_reset = True
            st.rerun()
    
    # --- Diálogo de Confirmação --- 
    if st.session_state.confirming_config_reset:
        st.warning("**Confirmar Reset?**\nTem certeza que deseja resetar todas as configurações do LLM para os valores padrão? Qualquer alteração será perdida.")
        col1_confirm, col2_confirm = st.columns(2)
        with col1_confirm:
            if st.button("Reset", use_container_width=True, key="confirm_reset_llm"):
                try:
                    logger.info("Confirmado: Resetando as configurações padrão do LLM...")
                    current_full_config = load_configuration() 
                    if not current_full_config:
                        st.error("Erro: Não foi possível carregar a configuração completa para reset.")
                    else:
                        sessions_to_reset = []
                        if current_full_config.llm != st.session_state.original_config.llm:
                            sessions_to_reset.append("llm")
                        if current_full_config.query != st.session_state.original_config.query:
                            sessions_to_reset.append("query")
                        manager.reset_config(current_full_config, sessions_to_reset) 
                        load_configuration.clear()
                        st.session_state.confirming_config_reset = False
                        st.sidebar.success("Configurações do LLM resetadas para os valores padrão")
                        st.rerun()
                except ValueError as e: 
                    st.error(f"Erro ao resetar as configurações do LLM: {e}")
                    st.session_state.confirming_config_reset = False 
                except ConfigurationError as e:
                    st.error(f"Erro ao resetar as configurações padrão do LLM: {e}") 
                    st.session_state.confirming_config_reset = False 
                except Exception as e:
                    st.error(f"Erro inesperado ao resetar as configurações padrão do LLM: {e}")
                    st.session_state.confirming_config_reset = False 
        with col2_confirm:
            if st.button("Cancel", use_container_width=True, key="cancel_reset_llm"):
                logger.debug("Reset cancelado pelo usuário.")
                st.session_state.confirming_config_reset = False
                st.rerun()
    
    st.divider()


# --- Gerenciamento do historico de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

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

    # --- Salva automaticamente a configuração do LLM se tiver sido alterada --- 
    try:
        current_query_config = QueryConfig(retrieval_k=config.query.retrieval_k)
        current_llm_config = LLMConfig(
            model_repo_id=llm_model_repo_id,
            prompt_template=llm_prompt_template,
            max_new_tokens=llm_max_new_tokens,
            temperature=llm_temperature,
            top_p=llm_top_p,
            top_k=llm_top_k,
            repetition_penalty=llm_repetition_penalty,
            max_retries=config.llm.max_retries, # Não é alterado nessa página
            retry_delay_seconds=config.llm.retry_delay_seconds, # Não é alterado nessa página
        )

        if current_llm_config != st.session_state.original_llm_config or current_query_config != st.session_state.original_query_config:
            logger.info("Parâmetros alterados no sidebar, salvando configuração automaticamente...")
            
            # Usa o objeto de configuração principal carregado no inicio da execução do script
            if not config: 
                 st.error("Não é possível salvar as alterações do LLM: Configuração principal não carregada.")
            else:
                config.llm = current_llm_config
                config.query = current_query_config
                orchestrator.update_config(config)
                updated_app_config = config.model_copy(update={'llm': current_llm_config})
                manager.save_config(updated_app_config)
                load_configuration.clear() 
                # Atualiza o estado da session com a nova configuração salva
                st.session_state.original_llm_config = copy.deepcopy(current_llm_config) 
                st.toast("Configuração do LLM salva automaticamente!")
        else:
            logger.debug("Parâmetros do LLM não alterados, prosseguindo com a query.")

    except ValidationError as e:
        st.error(f"Erro de validação da configuração do LLM durante o salvamento automático:\n{e}")
        st.stop()

    except ConfigurationError as e:
        st.error(f"Erro ao salvar o arquivo de configuração durante o salvamento automático:\n{e}")
        st.stop()

    except Exception as e:
        st.error(f"Erro inesperado durante o salvamento automático da configuração do LLM:\n{e}")
        logger.error("Erro durante o salvamento automático da configuração do LLM", exc_info=True)
        st.stop()
        
    # --- Processa a query --- 
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