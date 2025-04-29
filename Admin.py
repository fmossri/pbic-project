import streamlit as st
import sys # Add sys for stderr printing
import logging
import logging.handlers # Import handlers for type checking

from src.utils.logger import setup_logging, get_logger # Import get_logger if needed for callback log

st.set_page_config(
    layout="wide",
    page_title="RAG Admin",
    page_icon="📚",
)
# --- Callback Function to Update Log Levels --- 
def update_log_levels_callback():
    # Use a temporary logger for the callback message itself
    callback_logger = get_logger("AdminCallback", log_domain="gui_callback")
    callback_logger.info("Debug toggle changed, updating log levels...")
    
    try:
        is_debug = st.session_state.get('debug_mode', False)
        root_logger = logging.getLogger()
        
        # Determine target levels based on state
        file_level = logging.DEBUG if is_debug else logging.INFO
        console_level = logging.INFO if is_debug else logging.WARNING
        root_level = min(file_level, console_level) # Root needs to be most verbose
        
        callback_logger.info(f"Setting log levels: Root={root_level}, File={file_level}, Console={console_level}")
        
        # Set root logger level
        root_logger.setLevel(root_level)
        
        # Set handler levels
        handler_found = False
        for handler in root_logger.handlers:
            handler_found = True
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setLevel(file_level)
                callback_logger.debug(f"Set FileHandler level to {file_level}")
            elif isinstance(handler, logging.StreamHandler):
                # Be careful not to affect other StreamHandlers if any exist
                # Assuming the first StreamHandler is our console handler
                handler.setLevel(console_level)
                callback_logger.debug(f"Set StreamHandler level to {console_level}")
            else:
                callback_logger.warning(f"Unknown handler type found: {type(handler)}")
        
        if not handler_found:
             callback_logger.warning("No handlers found on root logger during callback.")
             
    except Exception as e:
        # Log any error during callback
        callback_logger.error(f"Error in update_log_levels_callback: {e}", exc_info=True)
# --------------------------------------------

# --- Initialize Session State for Debug Toggle (if not exists) ---
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False 
# ----------------------------------------------------------------

# --- Setup Logging Once via Cached Function ---
@st.cache_resource
def initialize_logging_session():
    # This function's body runs only once per session
    print(f"--- DEBUG Admin.py: Running initialize_logging_session (should only see once per session) ---", file=sys.stderr)
    # Use the state that exists when this *first* runs.
    # The callback will handle level changes later.
    initial_debug_state = st.session_state.get('debug_mode', False)
    print(f"--- DEBUG Admin.py: Initial setup_logging called by cache with debug={initial_debug_state} --- ", file=sys.stderr)
    setup_logging(log_dir="logs/gui", debug=initial_debug_state)
    print("--- DEBUG Admin.py: Returned from cached setup_logging call ---", file=sys.stderr)
    return True # Cache resource requires a return value

logging_initialized = initialize_logging_session() # Ensure this runs

# --- Sidebar Debug Toggle --- 
st.sidebar.divider()
# Print state just before rendering toggle
print(f"--- DEBUG Admin.py: Rendering toggle, state is {st.session_state.get('debug_mode', 'Not Set Yet')} ---", file=sys.stderr)
st.sidebar.toggle(
    "Debug Logging", 
    key="debug_mode", # Must match the session state key
    value=st.session_state.get('debug_mode', False), # Set initial value from state
    help="Enable detailed DEBUG level logging to file and INFO to console. Requires page refresh/interaction after toggling.",
    on_change=update_log_levels_callback # Add the callback here
)
st.sidebar.divider()
# --------------------------

st.sidebar.success("Selecione uma seção acima.")

st.title("📚 Página do Admin do Sistema RAG")
st.write("Bem-vindo! Use o sidebar para navegar entre as seções de gerenciamento.")