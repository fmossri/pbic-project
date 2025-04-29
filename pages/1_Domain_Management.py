import streamlit as st
import pandas as pd
from src.utils.domain_manager import DomainManager
import traceback 
from src.utils.logger import get_logger
import sys
from Admin import update_log_levels_callback # Import callback

# --- Logger ---
logger = get_logger(__name__, log_domain="gui")

# --- Initialization (MUST HAPPEN BEFORE st.set_page_config) --- 
@st.cache_resource 
def get_domain_manager():
    logger.info("Creating DomainManager instance (cached)")
    try:
        return DomainManager()
    except Exception as e:
        logger.error(f"Failed to create DomainManager instance: {e}", exc_info=True)
        st.error(f"Erro ao inicializar o DomainManager: {e}")
        st.code(traceback.format_exc())
        st.stop()

domain_manager = get_domain_manager()
# ---------------------------------------------------------------

# --- Page Configuration (NOW SAFE TO CALL) ---
st.set_page_config(
    page_title="Gerenciamento de Domínios",
    layout="wide"
)
st.title("🧠 Gerenciamento de Domínios de Conhecimento")
# ----------------------------------------------

# --- Initialize Session State (if not exists) ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# --- Sidebar Debug Toggle --- 
st.sidebar.divider()
print(f"--- DEBUG Domain Management: Rendering toggle, state is {st.session_state.get('debug_mode', 'Not Set Yet')} ---", file=sys.stderr)
st.sidebar.toggle(
    "Debug Logging", 
    key="debug_mode", 
    value=st.session_state.get('debug_mode', False), 
    help="Enable detailed DEBUG level logging...",
    on_change=update_log_levels_callback # Add the callback here
)
st.sidebar.divider()

# --- Initialize Session State for Confirmation ---
if 'confirming_delete_id' not in st.session_state:
    st.session_state.confirming_delete_id = None
if 'selected_domain_id' not in st.session_state:
    st.session_state.selected_domain_id = None

# --- Helper Function to Refresh Data ---
def refresh_domains():
    """Fetches the latest list of domains."""
    try:
        domains = domain_manager.list_domains()
        if domains:
            # Convert list of Domain objects to list of dicts for DataFrame
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
            return pd.DataFrame() # Return empty DataFrame if no domains
    except Exception as e:
        st.error(f"Erro ao listar domínios: {e}")
        st.code(traceback.format_exc())
        return pd.DataFrame() # Return empty DataFrame on error

# === MOVED BLOCK START ===
with st.form("create_domain_form", clear_on_submit=True): # Moved Form
    domain_name = st.text_input("Nome do Domínio", key="domain_name", placeholder="Ex: Ingestão de Dados em RAG")
    description = st.text_area("Descrição", key="description", placeholder="Descreva o propósito deste domínio.")
    keywords = st.text_input("Palavras-chave (separadas por vírgula)", key="keywords", placeholder="Ex: RAG, Ingestão de Dados, Processamento de Texto")
    
    submitted = st.form_submit_button("Criar Domínio")
    if submitted:
        if not domain_name or not description or not keywords:
            st.warning("Todos os campos são obrigatórios.")
        else:
            try:
                domain_manager.create_domain(domain_name, description, keywords)
                st.success(f"Domínio '{domain_name}' criado com sucesso!")
                st.rerun() # Explicitly trigger a rerun immediately
            except ValueError as ve:
                st.error(f"Erro ao criar domínio: {ve}")
            except Exception as e:
                st.error(f"Erro inesperado ao criar domínio: {e}")
                st.code(traceback.format_exc())
st.divider()

# --- Display Existing Domains ---
st.header("Domínios Existentes")
# Fetch the original, full data
original_domain_df = refresh_domains()

if not original_domain_df.empty:
    # Define columns for display (adjust as needed)
    cols = st.columns((1, 2, 3, 2, 1, 1)) # Adjust ratios as needed
    headers = ["ID", "Nome", "Descrição", "Palavras-chave", "Documentos", "Ações"]
    for col, header in zip(cols, headers):
        col.write(f"**{header}**")

    # --- Function to truncate text (already defined earlier) ---
    max_chars = 30 # Maybe shorter for row display
    def truncate_text(text, limit, position: str = "end"):
        if isinstance(text, str) and len(text) > limit:
            if position == "start":
                return "..." + text[-limit+3:]
            else:
                return text[:limit-3] + "..."
        return text

    # Iterate through the DataFrame rows
    for index, row in original_domain_df.iterrows():
        domain_id = row["ID"]
        domain_name = row["name"]
        
        col1, col2, col3, col4, col5, col6 = st.columns((1, 2, 3, 2, 1, 1)) # Same ratios as header
        
        # Display main row data
        with col1:
            st.write(domain_id)
        with col2:
            st.write(truncate_text(domain_name, max_chars))
        with col3:
            st.write(truncate_text(row["description"], max_chars))
        with col4:
            st.write(truncate_text(row["keywords"], max_chars))
        with col5:
            st.write(row["total_documents"])
        with col6:
            # Action buttons in the last column
            action_cols = st.columns(2) # Create sub-columns for buttons
            with action_cols[0]: # Info button
                info_key = f"info_{domain_id}"
                if st.button("ℹ️", key=info_key, help="Ver detalhes"):
                    st.session_state.selected_domain_id = domain_id
                    st.session_state.confirming_delete_id = None
                    st.rerun()
            
            with action_cols[1]: # Delete / Confirm / Cancel button
                # --- Confirmation Logic ---
                if st.session_state.confirming_delete_id == domain_id:
                    # Show Confirmation Buttons
                    confirm_key = f"confirm_delete_{domain_id}"
                    cancel_key = f"cancel_delete_{domain_id}"
                    
                    if st.button("✔️", key=confirm_key, help="Confirmar remoção", type="primary"):
                        try:
                            st.toast(f"Removendo domínio '{domain_name}'...", icon="⏳") 
                            domain_manager.remove_domain_registry_and_files(domain_name)
                            st.session_state.confirming_delete_id = None # Reset state
                            st.session_state.selected_domain_id = None # Close details if this was selected
                            st.toast(f"Domínio '{domain_name}' removido com sucesso!", icon="✅")
                            st.rerun()
                        except ValueError as ve:
                            st.error(f"Erro ao remover {domain_name}: {ve}")
                            st.session_state.confirming_delete_id = None # Reset state on error too
                        except Exception as e:
                            st.error(f"Erro inesperado ao remover {domain_name}: {e}")
                            st.code(traceback.format_exc())
                            st.session_state.confirming_delete_id = None
                    
                    if st.button("✖️", key=cancel_key, help="Cancelar remoção"):
                        st.session_state.confirming_delete_id = None
                        st.rerun()
                else:
                    # Show Initial Delete Button
                    delete_key = f"delete_{domain_id}"
                    if st.button("❌", key=delete_key, help=f"Remover domínio '{domain_name}'"):
                        st.session_state.confirming_delete_id = domain_id # Set state to ask for confirmation
                        st.session_state.selected_domain_id = None # Close details view if open
                        st.rerun()

else:
    st.info("Nenhum domínio de conhecimento encontrado.")

st.divider() # Divider before Detail Display Area

# --- Detail Display Area ---
if st.session_state.selected_domain_id is not None:
    st.divider()
    st.subheader("Editar Detalhes do Domínio Selecionado")
    
    # Find the selected domain's data from the original DataFrame
    # Use .copy() to avoid modifying the original DataFrame if needed later
    selected_domain_series = original_domain_df[original_domain_df["ID"] == st.session_state.selected_domain_id].iloc[0].copy()
    
    if selected_domain_series is not None:
        # Use lowercase key for original_name lookup now
        original_name = selected_domain_series['name'] 
        domain_id = selected_domain_series['ID']

        st.markdown(f"**ID:** {domain_id}")
        
        # --- Editable Fields ---
        # Use unique keys based on domain ID to handle state correctly when selection changes
        updatables = {}
        new_name = st.text_input("Nome", value=selected_domain_series['name'], key=f"edit_name_{domain_id}")
        updatables['name'] = new_name
        new_description = st.text_area("Descrição", value=selected_domain_series['description'], height=150, key=f"edit_desc_{domain_id}")
        updatables['description'] = new_description
        new_keywords = st.text_input("Palavras-chave", value=selected_domain_series['keywords'], key=f"edit_keywords_{domain_id}")
        updatables['keywords'] = new_keywords
        
        # --- Read-only Fields --- (Use lowercase keys where applicable)
        st.markdown(f"**Caminho DB:**")
        st.code(selected_domain_series["db_path"], language=None) # Use lowercase key
        st.markdown(f"**Caminho Vector Store:**")
        st.code(selected_domain_series["vector_store_path"], language=None) # Use lowercase key
        st.markdown(f"**Documentos:** {selected_domain_series['total_documents']}") # Use lowercase key
        st.markdown(f"**Criado em:** {selected_domain_series['created_at']}") # Use lowercase key
        st.markdown(f"**Atualizado em:** {selected_domain_series['updated_at']}") # Use lowercase key
        
        # --- Action Buttons ---
        col_save, col_close = st.columns(2)
        with col_save:
            if st.button("💾 Salvar Alterações", key=f"save_{domain_id}", type="primary"):
                updates = {}
                # Iterate through the potential updates from the input widgets
                for key, value in updatables.items():
                    # Check if the value is actually different from the original
                    # AND ensure the new value is not empty/whitespace
                    if value != selected_domain_series[key] and value.strip():
                        updates[key] = value
                
                if updates:
                    try:
                        st.toast("Salvando alterações...", icon="⏳")
                        # Use original_name for lookup, pass potentially new name in updates
                        domain_manager.update_domain_details(original_name, updates)
                        st.toast("Alterações salvas com sucesso!", icon="✅")
                        st.session_state.selected_domain_id = None # Close details view
                        st.session_state.confirming_delete_id = None # Just in case
                        st.rerun()
                    except ValueError as ve:
                        st.error(f"Erro ao salvar alterações: {ve}")
                    except Exception as e:
                        st.error(f"Erro inesperado ao salvar alterações: {e}")
                        st.code(traceback.format_exc())
                else:
                    # Clarify message and close the view to reset fields on next open
                    st.info("Nenhuma alteração válida detectada. Campos não podem ser vazios.") 
                    st.session_state.selected_domain_id = None # Close the details view
                    # st.rerun()

        with col_close:
            if st.button("❌ Fechar Detalhes", key=f"close_{domain_id}"):
                st.session_state.selected_domain_id = None
                st.rerun()
    else:
        st.warning("Não foi possível encontrar os detalhes do domínio selecionado.")
        st.session_state.selected_domain_id = None # Reset if data not found