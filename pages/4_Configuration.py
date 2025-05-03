# Placeholder for Configuration Page 

import streamlit as st
from typing import Dict, Any

from src.utils.logger import get_logger
from src.config.models import AppConfig, SystemConfig, IngestionConfig, EmbeddingConfig, VectorStoreConfig, QueryConfig, LLMConfig, TextNormalizerConfig
from src.config.config_manager import ConfigManager, ConfigurationError
from gui.streamlit_utils import load_configuration, initialize_logging_session, update_log_levels_callback, get_config_manager
from pydantic import ValidationError



st.set_page_config(
    layout="wide",
    page_title="Configuration",
    page_icon="⚙️",
)

config: AppConfig | None = load_configuration()
manager: ConfigManager = get_config_manager()

initialize_logging_session()
logger = get_logger(__name__, log_domain="gui/Configuration")

st.title("⚙️ Configurações Gerais")

# --- Sidebar Debug Toggle ---
with st.sidebar:
    logger.info(f"--- DEBUG Configuration: Rendering toggle, state is {st.session_state.get('debug_mode', 'Not Set Yet')} ---")
    st.toggle(
        "Debug Logging",
        key="debug_mode",
        value=st.session_state.get("log_level_debug", False),
        on_change=update_log_levels_callback,
        help="Ativa logs detalhados no terminal e no arquivo.",
    )

    st.sidebar.divider()

# --- Exibe Configuração Atual ---
st.subheader("Configuração Atual")

if config:
    
    # --- Formata a configuração para exibição --- 
    config_display_parts = []
    sections_to_display = {
        "System Settings": config.system,
        "Ingestion Settings": config.ingestion,
        "Embedding Settings": config.embedding,
        "Query Settings": config.query,
        "Vector Store Settings": config.vector_store,
        "LLM Settings": config.llm,
        "Text Normalizer Settings": config.text_normalizer
    }
    
    # --- Formata cada seção em uma string separada ---
    for title, section_model in sections_to_display.items():
        try:
            section_data = section_model.model_dump()
            lines = []
            for key, value in section_data.items():
                if isinstance(value, str):
                    # Coloca aspas em torno de todas as strings
                    formatted_value = '\"{}\"'.format(value)
                elif isinstance(value, bool):
                    formatted_value = str(value).lower()
                elif value is None:
                    # Omite valores None, pois não são utilizados
                    continue 
                else:
                    formatted_value = str(value)
                    
                lines.append("  {} = {}".format(key, formatted_value))
                
            section_string = f"\n".join(lines)
            config_display_parts.append("{}:\n{}".format(title, section_string))
        except Exception as e:
            config_display_parts.append("{}:\\n  Error displaying section: {}".format(title, e))
            
    full_config_display_string = f"\n\n".join(config_display_parts)
    
    # Exibe em uma área de texto somente leitura
    st.text_area(
        label="Detalhes da Configuração", 
        value=full_config_display_string, 
        height=400, # Ajusta a altura conforme necessário
        disabled=True, 
        key="readonly_config_display"
    )
    
    st.divider()

    # --- Formulário de Edição da Configuração ---
    st.header("Editar Configurações Gerais")
    
    if not config:
         st.warning("Não é possível exibir o formulário de edição. Configuração não foi carregada.")
    else:
        with st.form("config_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                # --- Sistema --- 
                st.subheader("Sistema")
                system_storage_base_path = st.text_input("Diretório base de armazenamento", value=config.system.storage_base_path, key="system_storage_base_path")
                system_control_db_filename = st.text_input("Arquivo do banco de dados de controle", value=config.system.control_db_filename, key="system_control_db_filename")
            with col2:
                # --- LLM --- 
                st.subheader("LLM")
                llm_max_retries = st.number_input("Max Retries (Erro de LLM)", min_value=0, step=1, value=config.llm.max_retries, key="llm_max_retries")
                llm_retry_delay_seconds = st.number_input("Retry Delay (segundos)", min_value=1, step=1, value=config.llm.retry_delay_seconds, key="llm_retry_delay_seconds")
            with col3:
            # --- Text Normalizer --- 
                st.subheader("Normalização de Texto")
                norm_unicode = st.checkbox("Normalização Unicode", value=config.text_normalizer.use_unicode_normalization, key="norm_unicode")
                norm_lowercase = st.checkbox("Lowercase", value=config.text_normalizer.use_lowercase, key="norm_lowercase")
                norm_remove_whitespace = st.checkbox("Normalização de whitespaces", value=config.text_normalizer.use_remove_extra_whitespace, key="norm_remove_whitespace")
            
            # --- Submit Button --- 
            submitted = st.form_submit_button("Save Configuration")

            if submitted:
                try:
                    final_embedding_model_name = config.embedding.model_name


                    # Reconstruct the AppConfig object from form values
                    new_system_config = SystemConfig(
                        storage_base_path=system_storage_base_path,
                        control_db_filename=system_control_db_filename,
                    )

                    new_llm_config = LLMConfig(
                        model_repo_id=config.llm.model_repo_id, # Não é alterado nessa página
                        prompt_template=config.llm.prompt_template, # Não é alterado nessa página
                        max_new_tokens=config.llm.max_new_tokens, # Não é alterado nessa página
                        temperature=config.llm.temperature, # Não é alterado nessa página
                        top_p=config.llm.top_p, # Não é alterado nessa página
                        top_k=config.llm.top_k, # Não é alterado nessa página
                        repetition_penalty=config.llm.repetition_penalty, # Não é alterado nessa página
                        max_retries=llm_max_retries,
                        retry_delay_seconds=llm_retry_delay_seconds, 
                    )
                    new_normalizer_config = TextNormalizerConfig(
                        use_unicode_normalization=norm_unicode,
                        use_lowercase=norm_lowercase,
                        use_remove_extra_whitespace=norm_remove_whitespace,
                    )
                    
                    # Cria o objeto AppConfig final
                    validated_config = AppConfig(
                        system=new_system_config,
                        ingestion=config.ingestion,
                        embedding=config.embedding,
                        vector_store=config.vector_store,
                        query=config.query, # Não é alterado nessa página
                        llm=new_llm_config,
                        text_normalizer=new_normalizer_config,
                    )

                    manager.save_config(validated_config)
                    load_configuration.clear() # Limpa o cache de dados
                    st.success("Configuração salva.")
                    st.rerun()

                except ValidationError as e:
                    st.error(f"Erro de validação da configuração:\n{e}")
                except ConfigurationError as e:
                    st.error(f"Erro ao salvar arquivo de configuração: {e}")
                except Exception as e:
                    st.error(f"Erro inesperado: {e}")
        
        st.divider()
        # --- Botões de Restore Backup e Reset---
        col1, col2 = st.columns(2)
        with col1:
            restore_clicked = st.button("🔄 Restaurar Backup", key="restore_btn")
        with col2:
            reset_clicked = st.button("Reset Geral", key="reset_btn")

        if restore_clicked:
            st.info("Tentando restaurar a configuração a partir do backup...")
            # Use the manager instance to restore
            success = manager.restore_config_from_backup()
            if success:
                st.success("Configuração restaurada com sucesso a partir do backup!")
                load_configuration.clear() # Limpa o cache de dados
                st.rerun() # Recarrega a página para carregar os valores restaurados no formulário
            else:
                st.error("Falha ao restaurar a configuração. O arquivo de backup pode estar ausente ou corrompido.")
                # Não recarrega em caso de falha, mantém os valores atuais no formulário

        if reset_clicked:
            st.info("Reiniciando a configuração para os valores padrão...")
            manager.reset_config(config)
            st.success("Configuração reiniciada para os valores padrão com sucesso!")
            st.rerun()
else: 
    st.error("Falha ao carregar a configuração da aplicação. Não é possível exibir o formulário de edição. Por favor, verifique os logs e o arquivo config.toml.")