import streamlit as st
import pandas as pd
import traceback 
from src.models import DocumentFile
from src.utils.logger import get_logger
from gui.streamlit_utils import (
    update_log_levels_callback, 
    get_domain_manager, 
    initialize_logging_session, 
    load_configuration,
    get_domain_documents,
    delete_document_from_domain
)


st.set_page_config(
    page_title="Gerenciamento de Domínios",
    layout="wide"
)

initialize_logging_session()

logger = get_logger(__name__, log_domain="gui/DomainManagement")
config = load_configuration()
if config:
    domain_manager = get_domain_manager(config)

# --- Inicializa o estado da sessão (se não existir) ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'confirming_delete_id' not in st.session_state:
    st.session_state.confirming_delete_id = None
if 'selected_domain_id' not in st.session_state:
    st.session_state.selected_domain_id = None

# --- Titulo da Página ---
st.title("🧠 Gerenciamento de Domínios de Conhecimento")

# --- Sidebar Debug Toggle --- 
logger.info(f"--- DEBUG Domain Management: Renderizando toggle, estado é {st.session_state.get('debug_mode', 'Nao definido ainda')} ---")
st.sidebar.toggle(
    "Debug Logging", 
    key="debug_mode", 
    value=st.session_state.get('debug_mode', False), 
    help="Ativa o logging detalhado de nível DEBUG...",
    on_change=update_log_levels_callback 
)
st.sidebar.divider()

# --- Função auxiliar para atualizar os dados ---
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


# --- Exibe os domínios existentes ---
st.header("Selecione um domínio")


# Recupera os dados originais, completos
original_domain_df = refresh_domains_dataframe()

if not original_domain_df.empty:
    # Define as colunas para exibição
    cols = st.columns((1, 2, 3, 2, 1, 1))
    headers = ["ID", "Nome", "Descrição", "Palavras-chave", "Documentos", "Ações"]
    for col, header in zip(cols, headers):
        col.write(f"**{header}**")

    # --- Função para truncar o texto  ---
    max_chars = 30
    def truncate_text(text, limit, position: str = "end"):
        if isinstance(text, str) and len(text) > limit:
            if position == "start":
                return "..." + text[-limit+3:]
            else:
                return text[:limit-3] + "..."
        return text
    
    domain_names = original_domain_df['name'].tolist()
    
    # Inicializa selected_domain_name na sessão se não existir ou se a seleção anterior não for mais válida
    if 'selected_domain_name' not in st.session_state or st.session_state.selected_domain_name not in domain_names:
        st.session_state.selected_domain_name = domain_names[0] if domain_names else None

    # --- Domain Selection Dropdown ---
    selected_domain_name = st.selectbox(
        "Selecione um Domínio para Gerenciar:",
        options=domain_names,
        key='selected_domain_name',
        index=domain_names.index(st.session_state.selected_domain_name) if st.session_state.selected_domain_name in domain_names else 0,
        help="Escolha o domínio cujos detalhes você deseja ver ou editar.",
    )

    domain_id = original_domain_df[original_domain_df["name"] == selected_domain_name]["ID"].values[0]
    
    # --- Botão de Remoção com Confirmação ---
    if 'confirming_delete_name' not in st.session_state:
        st.session_state.confirming_delete_name = None

    delete_placeholder = st.empty() # Placeholder for delete buttons

    if st.session_state.confirming_delete_name == selected_domain_name:
        # Show confirmation buttons in the placeholder
        with delete_placeholder.container():
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✔️ Confirmar Remoção", key=f"confirm_delete_{domain_id}", type="primary"):
                    try:
                        st.toast(f"Removendo domínio '{selected_domain_name}'...", icon="⏳") 
                        domain_manager.remove_domain_registry_and_files(selected_domain_name)
                        st.toast(f"Domínio '{selected_domain_name}' removido com sucesso!", icon="✅")
                        st.session_state.confirming_delete_name = None # Reset confirmation state
                        st.session_state.selected_domain_name = None # Reset selection
                        st.rerun()
                    except ValueError as ve:
                        st.error(f"Erro ao remover {selected_domain_name}: {ve}")
                        st.session_state.confirming_delete_name = None # Reset on error
                    except Exception as e:
                        st.error(f"Erro inesperado ao remover {selected_domain_name}: {e}")
                        st.code(traceback.format_exc())
                        st.session_state.confirming_delete_name = None # Reset on error
            with col_cancel:
                if st.button("✖️ Cancelar", key=f"cancel_delete_{domain_id}"):
                    st.session_state.confirming_delete_name = None # Reset confirmation state
                    st.rerun()
    else:
        # Show initial delete button in the placeholder
        if delete_placeholder.button(f"❌ Remover", key=f"delete_{domain_id}"):
            st.session_state.confirming_delete_name = selected_domain_name # Set confirmation state
            st.rerun()


else:
    st.info("Nenhum domínio de conhecimento encontrado. Crie um na página Admin.")
    if 'selected_domain_name' in st.session_state:
        st.session_state.selected_domain_name = None 
    selected_domain_name = None


# --- Área de exibição de detalhes ---
if st.session_state.selected_domain_name is not None:
    st.divider()
    st.subheader("Editar Detalhes do Domínio Selecionado")
    
    # Encontra os dados do domínio selecionado do DataFrame original
    selected_domain_series = original_domain_df[original_domain_df["name"] == st.session_state.selected_domain_name].iloc[0].copy()
    
    if selected_domain_series is not None:
        original_name = selected_domain_series['name'] 
        domain_id = selected_domain_series['ID']

        # --- Two-Column Layout for Details/Edit ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"**ID:** {domain_id}")
            
            # --- Campos editáveis ---
            updatables = {}
            new_name = st.text_input("Nome", value=selected_domain_series['name'], key=f"edit_name_{domain_id}")
            updatables['name'] = new_name
            new_description = st.text_area("Descrição", value=selected_domain_series['description'], height=150, key=f"edit_desc_{domain_id}")
            updatables['description'] = new_description
            new_keywords = st.text_input("Palavras-chave", value=selected_domain_series['keywords'], key=f"edit_keywords_{domain_id}")
            updatables['keywords'] = new_keywords
            
            # --- Botão Salvar ---
            if st.button("💾 Salvar Alterações", key=f"save_{domain_id}", type="primary"):
                updates = {}
                for key, value in updatables.items():
                    if value != selected_domain_series[key] and value.strip():
                        updates[key] = value
                
                if updates:
                    try:
                        st.toast("Salvando alterações...", icon="⏳")
                        domain_manager.update_domain_details(original_name, updates)
                        st.toast("Alterações salvas com sucesso!", icon="✅")
                        # Clear selection state after save
                        st.session_state.selected_domain_name = None 
                        st.session_state.confirming_delete_name = None # Reset delete confirm just in case
                        st.rerun()
                    except ValueError as ve:
                        st.error(f"Erro ao salvar alterações: {ve}")
                    except Exception as e:
                        st.error(f"Erro inesperado ao salvar alterações: {e}")
                        st.code(traceback.format_exc())
                else:
                    st.info("Nenhuma alteração válida detectada. Campos não podem ser vazios.")

        with col_right:
            # --- Campos de leitura ---
            st.markdown(f"**Documentos:** {selected_domain_series['total_documents']}")
            st.markdown(f"**Criado em:** {selected_domain_series['created_at']}")
            st.markdown(f"**Atualizado em:** {selected_domain_series['updated_at']}")
            st.markdown(f"**Caminho DB:**")
            st.code(selected_domain_series["db_path"], language=None)
            st.markdown(f"**Caminho Vector Store:**")
            st.code(selected_domain_series["vector_store_path"], language=None)


        # --- Seção da lista de documentos ---
        st.divider()
        st.markdown("#### Documentos no Domínio")

        if 'confirming_delete_doc_id' not in st.session_state:
            st.session_state.confirming_delete_doc_id = None

        documents = get_domain_documents(domain_manager, selected_domain_name)

        if not documents:
            st.info("Nenhum documento encontrado para este domínio.")
        else:
            # Exibir documentos em um formato de lista com botões de delete
            list_cols = st.columns((3, 1)) # Colunas para nome e botão
            list_cols[0].write("**Nome do Arquivo**")
            list_cols[1].write("**Ação**")

            for doc in documents:
                doc_id = doc.id # Use doc.id para chaves únicas
                doc_name = doc.name
                cols = st.columns((3, 1))
                  
                with cols[0]:
                    st.write(doc_name)
                    # Opcionalmente, exibir outras informações como hash ou número de páginas
                    # st.caption(f"ID: {doc_id}, Hash: {doc.hash[:8]}...")

                with cols[1]:
                    delete_key_base = f"doc_{doc_id}"
                    # Lógica de confirmação para remover o documento
                    if st.session_state.confirming_delete_doc_id == doc_id:
                        confirm_key = f"confirm_delete_{delete_key_base}"
                        cancel_key = f"cancel_delete_{delete_key_base}"
                          
                        action_cols = st.columns(2)
                        with action_cols[0]:
                            if st.button("✔️", key=confirm_key, help=f"Confirmar remoção de {doc_name}", type="primary"):
                                st.toast(f"Removendo documento '{doc_name}'...", icon="⏳")
                                success = delete_document_from_domain(domain_manager, selected_domain_name, doc)
                                if success:
                                    st.toast(f"Documento '{doc_name}' removido com sucesso!", icon="✅")
                                    st.session_state.confirming_delete_doc_id = None   
                                    st.rerun()
                                else:
                                    st.session_state.confirming_delete_doc_id = None
                        with action_cols[1]:
                            if st.button("✖️", key=cancel_key, help="Cancelar remoção"):
                                st.session_state.confirming_delete_doc_id = None
                                st.rerun()
                    else:
                        # Botão de delete inicial
                        delete_key = f"delete_{delete_key_base}"
                        if st.button("❌", key=delete_key, help=f"Remover documento '{doc_name}'"):
                            st.session_state.confirming_delete_doc_id = doc_id
                            st.rerun()

# Cleanup session state 
if 'selected_domain_id' in st.session_state:
    del st.session_state['selected_domain_id']
if 'confirming_delete_id' in st.session_state:
    del st.session_state['confirming_delete_id']