import streamlit as st
import sys # Add sys for stderr printing

from src.utils.logger import setup_logging

setup_logging()

st.set_page_config(
    layout="wide",
    page_title="RAG Admin",
    page_icon="📚",
)

st.sidebar.success("Selecione uma seção acima.")

st.title("📚 Página do Admin do Sistema RAG")
st.write("Bem-vindo! Use o sidebar para navegar entre as seções de gerenciamento.")