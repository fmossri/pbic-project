import streamlit as st
import sys
import pandas as pd
import traceback 

from src.utils.logger import get_logger
from gui.streamlit_utils import update_log_levels_callback, get_domain_manager


st.set_page_config(
    page_title="Gerenciamento de Domínios",
    layout="wide"
)

st.title("🧠 Gerenciamento de Domínios de Conhecimento")

# --- Inicialização de Recursos ---
logger = get_logger(__name__, log_domain="gui")
domain_manager = get_domain_manager()

# --- Inicializa o estado da sessão (se não existir) ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'confirming_delete_id' not in st.session_state:
    st.session_state.confirming_delete_id = None
if 'selected_domain_id' not in st.session_state:
    st.session_state.selected_domain_id = None

# --- Sidebar Debug Toggle --- 
st.sidebar.divider()
print(f"--- DEBUG Domain Management: Renderizando toggle, estado é {st.session_state.get('debug_mode', 'Nao definido ainda')} ---", file=sys.stderr)
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

# === Formulario de Criação de Domínio ===
with st.form("create_domain_form", clear_on_submit=True):
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
                st.rerun()
            except ValueError as ve:
                st.error(f"Erro ao criar domínio: {ve}")
            except Exception as e:
                st.error(f"Erro inesperado ao criar domínio: {e}")
                st.code(traceback.format_exc())
st.divider()

# --- Exibe os domínios existentes ---
st.header("Domínios Existentes")
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

    # Itera através das linhas do DataFrame
    for index, row in original_domain_df.iterrows():
        domain_id = row["ID"]
        domain_name = row["name"]
        
        col1, col2, col3, col4, col5, col6 = st.columns((1, 2, 3, 2, 1, 1)) # Mesmas proporções que o header
        
        # Exibe os dados principais da linha
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
            # Botões de ação na última coluna
            action_cols = st.columns(2) # Cria sub-colunas para os botões
            with action_cols[0]: # Botão de informação
                info_key = f"info_{domain_id}"
                if st.button("ℹ️", key=info_key, help="Ver detalhes"):
                    st.session_state.selected_domain_id = domain_id
                    st.session_state.confirming_delete_id = None
                    st.rerun()
            
            with action_cols[1]: # Delete / Confirm / Cancel button
                # --- Logica de Cofirmação ---
                if st.session_state.confirming_delete_id == domain_id:
                    # Mostra os botões de confirmação
                    confirm_key = f"confirm_delete_{domain_id}"
                    cancel_key = f"cancel_delete_{domain_id}"
                    
                    if st.button("✔️", key=confirm_key, help="Confirmar remoção", type="primary"):
                        try:
                            st.toast(f"Removendo domínio '{domain_name}'...", icon="⏳") 
                            domain_manager.remove_domain_registry_and_files(domain_name)
                            st.session_state.confirming_delete_id = None # Reseta o estado
                            st.session_state.selected_domain_id = None # Fecha os detalhes se este foi selecionado
                            st.toast(f"Domínio '{domain_name}' removido com sucesso!", icon="✅")
                            st.rerun()
                        except ValueError as ve:
                            st.error(f"Erro ao remover {domain_name}: {ve}")
                            st.session_state.confirming_delete_id = None
                        except Exception as e:
                            st.error(f"Erro inesperado ao remover {domain_name}: {e}")
                            st.code(traceback.format_exc())
                            st.session_state.confirming_delete_id = None
                    
                    if st.button("✖️", key=cancel_key, help="Cancelar remoção"):
                        st.session_state.confirming_delete_id = None
                        st.rerun()
                else:
                    # Exibe o botão de remoção inicial
                    delete_key = f"delete_{domain_id}"
                    if st.button("❌", key=delete_key, help=f"Remover domínio '{domain_name}'"):
                        st.session_state.confirming_delete_id = domain_id # Define o estado para solicitar confirmação
                        st.rerun()

else:
    st.info("Nenhum domínio de conhecimento encontrado.")

st.divider()

# --- Área de exibição de detalhes ---
if st.session_state.selected_domain_id is not None:
    st.divider()
    st.subheader("Editar Detalhes do Domínio Selecionado")
    
    # Encontra os dados do domínio selecionado do DataFrame original
    selected_domain_series = original_domain_df[original_domain_df["ID"] == st.session_state.selected_domain_id].iloc[0].copy()
    
    if selected_domain_series is not None:
        original_name = selected_domain_series['name'] 
        domain_id = selected_domain_series['ID']

        st.markdown(f"**ID:** {domain_id}")
        
        # --- Campos editáveis ---
        updatables = {}
        new_name = st.text_input("Nome", value=selected_domain_series['name'], key=f"edit_name_{domain_id}")
        updatables['name'] = new_name
        new_description = st.text_area("Descrição", value=selected_domain_series['description'], height=150, key=f"edit_desc_{domain_id}")
        updatables['description'] = new_description
        new_keywords = st.text_input("Palavras-chave", value=selected_domain_series['keywords'], key=f"edit_keywords_{domain_id}")
        updatables['keywords'] = new_keywords
        
        # --- Campos de leitura ---
        st.markdown(f"**Caminho DB:**")
        st.code(selected_domain_series["db_path"], language=None)
        st.markdown(f"**Caminho Vector Store:**")
        st.code(selected_domain_series["vector_store_path"], language=None)
        st.markdown(f"**Documentos:** {selected_domain_series['total_documents']}")
        st.markdown(f"**Criado em:** {selected_domain_series['created_at']}")
        st.markdown(f"**Atualizado em:** {selected_domain_series['updated_at']}")
        
        # --- Botões de ação ---
        col_save, col_close = st.columns(2)
        with col_save:
            if st.button("💾 Salvar Alterações", key=f"save_{domain_id}", type="primary"):
                updates = {}

                for key, value in updatables.items():
                    # Verifica se o valor é realmente diferente do original
                    # E garante que o novo valor não esteja vazio/em branco
                    if value != selected_domain_series[key] and value.strip():
                        updates[key] = value
                
                if updates:
                    try:
                        st.toast("Salvando alterações...", icon="⏳")
                        # Usa original_name para lookup, passa o novo nome em updates
                        domain_manager.update_domain_details(original_name, updates)
                        st.toast("Alterações salvas com sucesso!", icon="✅")
                        st.session_state.selected_domain_id = None
                        st.session_state.confirming_delete_id = None 
                        st.rerun()
                    except ValueError as ve:
                        st.error(f"Erro ao salvar alterações: {ve}")
                    except Exception as e:
                        st.error(f"Erro inesperado ao salvar alterações: {e}")
                        st.code(traceback.format_exc())
                else:
                    # Clarifica a mensagem e fecha a exibição para resetar os campos na próxima abertura
                    st.info("Nenhuma alteração válida detectada. Campos não podem ser vazios.") 
                    st.session_state.selected_domain_id = None

        with col_close:
            if st.button("❌ Fechar Detalhes", key=f"close_{domain_id}"):
                st.session_state.selected_domain_id = None
                st.rerun()
    
    # --- Se o domínio selecionado não existe no DataFrame, acusa erro e fecha a exibição ---
    else:
        st.error("Erro: Não foi possível encontrar os detalhes do domínio selecionado.")
        logger.critical("Erro Critico: Nao foi possivel encontrar os detalhes do dominio selecionado. Registro corrompido, acao necessaria.", domain_id=domain_id)
        st.session_state.selected_domain_id = None