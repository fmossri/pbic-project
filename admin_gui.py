import streamlit as st
import sys # Add sys for stderr printing

from src.utils.logger import setup_logging

setup_logging()

st.set_page_config(
    layout="wide",
    page_title="RAG Admin",
    page_icon="📚",
)

st.sidebar.success("Select a section above.")

st.title("📚 RAG System Admin Interface")
st.write("Welcome! Use the sidebar to navigate between management sections.")