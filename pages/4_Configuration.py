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
        "Vector Store Settings": config.vector_store,
        "Query Settings": config.query,
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
    st.header("Editar Configuração")
    
    if not config:
         st.warning("Não é possível exibir o formulário de edição. Configuração não foi carregada.")
    else:
        with st.form("config_form"): 
            # --- Sistema --- 
            st.subheader("Sistema")
            system_storage_base_path = st.text_input("Diretório base de armazenamento", value=config.system.storage_base_path, key="system_storage_base_path")
            system_control_db_filename = st.text_input("Arquivo do banco de dados de controle", value=config.system.control_db_filename, key="system_control_db_filename")
            
            # --- Ingestão --- 
            st.subheader("Ingestão")
            ingestion_chunk_strategy = st.selectbox("Estratégia de chunk", options=["recursive"], index=0, key="ingestion_chunk_strategy") # Only recursive for now
            ingestion_chunk_size = st.number_input("Tamanho do chunk em chars", min_value=50, step=10, value=config.ingestion.chunk_size, key="ingestion_chunk_size")
            ingestion_chunk_overlap = st.number_input("Overlap", min_value=0, step=10, value=config.ingestion.chunk_overlap, key="ingestion_chunk_overlap")

            # --- Embedding --- 
            st.subheader("Embedding")
            
            # Lista de modelos comuns
            embedding_model_options = [
                "sentence-transformers/all-MiniLM-L6-v2", # Escolha padrão
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # Bom modelo multilingue
                "sentence-transformers/all-mpnet-base-v2", # Maior, mais poderoso
                "intfloat/e5-large-v2", # E5 models
                # Adicionar outros modelos a gosto
            ]
            current_embedding_model = config.embedding.model_name
            # Se o modelo atual não está na lista de modelos, adiciona ele no topo da lista
            if current_embedding_model not in embedding_model_options:
                embedding_model_options.insert(0, current_embedding_model)
                current_embedding_index = 0
                is_custom_model = True
            else:
                current_embedding_index = embedding_model_options.index(current_embedding_model)
                is_custom_model = False
                
            embedding_model_name_select = st.selectbox(
                "Modelo Recomendado (Selecione um)", 
                options=embedding_model_options, 
                index=current_embedding_index, 
                key="embedding_model_name_select",
                help="Selecione um modelo comum, ou use o campo 'Modelo Personalizado' abaixo."
            )
            
            # Campo de modelo personalizado - toma precedência se preenchido
            #custom_model_value = current_embedding_model if is_custom_model else "" # Pre-fill only if current is custom
            #embedding_model_name_custom = st.text_input(
            #    "Custom Model Name (Overrides Selection Above if Filled)", 
            #    value=custom_model_value, 
            #    key="embedding_model_name_custom",
            #    help="Enter the Hugging Face Hub ID of any Sentence Transformer model (e.g., 'sentence-transformers/paraphrase-MiniLM-L6-v1')."
            #)

            embedding_device = st.selectbox("Dispositivo", options=["cpu", "cuda"], index=["cpu", "cuda"].index(config.embedding.device), key="embedding_device")
            embedding_batch_size = st.number_input("Batch Size", min_value=1, step=1, value=config.embedding.batch_size, key="embedding_batch_size")
            embedding_normalize_embeddings = st.checkbox("Normaliza Embeddings", value=config.embedding.normalize_embeddings, key="embedding_normalize_embeddings")
            
            # --- Vector Store --- 
            st.subheader("Vector Store (FAISS)")
            vector_store_options = ["IndexFlatL2"] # Currently only option from Literal
            current_vector_store_type = config.vector_store.index_type
            vector_store_index = 0 if current_vector_store_type == "IndexFlatL2" else 0 
            vector_store_index_type = st.selectbox("Index Type", options=vector_store_options, index=vector_store_index, key="vector_store_index_type")

            # --- LLM --- 
            st.subheader("LLM")
            llm_max_retries = st.number_input("Max Retries (Erro de LLM)", min_value=0, step=1, value=config.llm.max_retries, key="llm_max_retries")
            llm_retry_delay_seconds = st.number_input("Retry Delay (segundos)", min_value=1, step=1, value=config.llm.retry_delay_seconds, key="llm_retry_delay_seconds")

            # --- Text Normalizer --- 
            st.subheader("Normalização de Texto")
            norm_unicode = st.checkbox("Normalização Unicode", value=config.text_normalizer.use_unicode_normalization, key="norm_unicode")
            norm_lowercase = st.checkbox("Lowercase", value=config.text_normalizer.use_lowercase, key="norm_lowercase")
            norm_remove_whitespace = st.checkbox("Normalização de whitespaces", value=config.text_normalizer.use_remove_extra_whitespace, key="norm_remove_whitespace")
            
            # --- Submit Button --- 
            submitted = st.form_submit_button("Save Configuration")

            if submitted:
                try:
                    final_embedding_model_name = embedding_model_name_select
                    #if embedding_model_name_custom.strip(): # If custom field is filled
                    #    final_embedding_model_name = embedding_model_name_custom.strip()
                    #    st.info(f"Using custom embedding model name: {final_embedding_model_name}") # Inform user
                    
                    # Reconstruct the AppConfig object from form values
                    new_system_config = SystemConfig(
                        storage_base_path=system_storage_base_path,
                        control_db_filename=system_control_db_filename,
                    )
                    new_ingestion_config = IngestionConfig(
                        chunk_strategy=ingestion_chunk_strategy,
                        chunk_size=ingestion_chunk_size,
                        chunk_overlap=ingestion_chunk_overlap,
                    )
                    new_embedding_config = EmbeddingConfig(
                        model_name=final_embedding_model_name,
                        device=embedding_device,
                        batch_size=embedding_batch_size,
                        normalize_embeddings=embedding_normalize_embeddings,
                    )
                    new_vector_store_config = VectorStoreConfig(
                        index_type=vector_store_index_type,
                    )
                    new_llm_config = LLMConfig(
                        model_repo_id=config.llm.llm_model_repo_id, # Não é alterado nessa página
                        prompt_template=config.llm.llm_prompt_template, # Não é alterado nessa página
                        max_new_tokens=config.llm.llm_max_new_tokens, # Não é alterado nessa página
                        temperature=config.llm.llm_temperature, # Não é alterado nessa página
                        top_p=config.llm.llm_top_p, # Não é alterado nessa página
                        top_k=config.llm.llm_top_k, # Não é alterado nessa página
                        repetition_penalty=config.llm.llm_repetition_penalty, # Não é alterado nessa página
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
                        ingestion=new_ingestion_config,
                        embedding=new_embedding_config,
                        vector_store=new_vector_store_config,
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