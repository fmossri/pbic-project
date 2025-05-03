import streamlit as st
import traceback
import pandas as pd

from gui.streamlit_utils import update_log_levels_callback, initialize_logging_session, load_configuration, get_domain_manager
from src.utils.logger import get_logger

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


initialize_logging_session()


logger = get_logger(__name__, log_domain="gui/Admin")
config = load_configuration()
if config:
    domain_manager = get_domain_manager(config)

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

def refresh_domains_dataframe():
    """Recupera a lista mais recente de domínios."""
    try:
        domains = domain_manager.list_domains()
        if domains:
            # Converte a lista de objetos Domain para uma lista de dicionários para o DataFrame
            domain_data = [
                {
                    "ID": d.id,
                    "name": d.name,
                    "description": d.description,
                    "keywords": d.keywords,
                    "total_documents": d.total_documents,
                    "db_path": d.db_path,
                    "vector_store_path": d.vector_store_path,
                    "created_at": d.created_at,
                    "updated_at": d.updated_at
                }
                for d in domains
            ]
            return pd.DataFrame(domain_data)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao listar domínios: {e}")
        st.code(traceback.format_exc())
        return pd.DataFrame()
    


# === Exibição da Lista de Domínios ===
st.header("Domínios Existentes")
domains_df = refresh_domains_dataframe()
if not domains_df.empty:
    # Seleciona e reordena as colunas desejadas
    display_df = domains_df[["name", "description", "keywords", "total_documents"]]
    st.dataframe(display_df, use_container_width=True)
else:
    st.info("Nenhum domínio encontrado.")

st.divider()

st.header("Criar Novo Domínio")
# === Formulario de Criação de Domínio ===
with st.form("create_domain_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        domain_name = st.text_input("Nome do Domínio", key="domain_name", placeholder="Ex: Ingestão de Dados em RAG")
        description = st.text_area("Descrição", key="description", placeholder="Descreva o propósito deste domínio.")
        keywords = st.text_input("Palavras-chave (separadas por vírgula)", key="keywords", placeholder="Ex: RAG, Ingestão de Dados, Processamento de Texto")
    
    with col2:

        ingestion_chunk_strategy = st.selectbox("Estratégia de chunking", options=["recursive"], index=0, key="ingestion_chunk_strategy") # Only recursive for now
        ingestion_chunk_size = st.number_input("Tamanho do chunk em chars", min_value=50, step=10, value=config.ingestion.chunk_size, key="ingestion_chunk_size")
        ingestion_chunk_overlap = st.number_input("Overlap", min_value=0, step=10, value=config.ingestion.chunk_overlap, key="ingestion_chunk_overlap")
              
        faiss_index_type = st.selectbox(
            "Índice Faiss", 
            options=config.vector_store.vector_store_options,
            key="faiss_index_type",
            help="Índice Faiss usado para armazenar e buscar representações vetoriais dos documentos e da query. Não pode ser alterado após a criação do domínio."
            )

        embedding_model = st.selectbox(
            "Modelo de Embedding", 
            options=config.embedding.embedding_options,  
            key="embedding_model_name_select",
            help="Modelo de Embedding usado para criar representações vetoriais dos documentos e da query. Não pode ser alterado após a criação do domínio."
            )
        embedding_normalize_embeddings = st.checkbox("Normaliza Embeddings", value=config.embedding.normalize_embeddings, key="embedding_normalize_embeddings")
            


    submitted = st.form_submit_button("Criar Domínio")
    if submitted:
        if not domain_name or not description or not keywords:
            st.warning("Todos os campos são obrigatórios.")
        else:
            try:
                domain_data = {
                    "name": domain_name,
                    "description": description,
                    "keywords": keywords,
                    "embeddings_model": embedding_model,
                    "faiss_index_type": faiss_index_type,
                    #"ingestion_chunk_strategy": ingestion_chunk_strategy,
                    #"ingestion_chunk_size": ingestion_chunk_size,
                    #"ingestion_chunk_overlap": ingestion_chunk_overlap,
                    #"embedding_normalize_embeddings": embedding_normalize_embeddings
                }
                domain_manager.create_domain(domain_data)
                st.success(f"Domínio '{domain_name}' criado com sucesso!")
                st.rerun()
            except ValueError as ve:
                st.error(f"Erro ao criar domínio: {ve}")
            except Exception as e:
                st.error(f"Erro inesperado ao criar domínio: {e}")
                st.code(traceback.format_exc())
