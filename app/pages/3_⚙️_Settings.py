"""
Settings Page
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from app.core.config import settings

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Settings")

st.subheader("Application Configuration")

col1, col2 = st.columns(2)

with col1:
    st.write("**Project Information**")
    st.text_input("Project Name", value=settings.PROJECT_NAME, disabled=True)
    st.text_input("Version", value=settings.PROJECT_VERSION, disabled=True)
    st.text_area("Description", value=settings.PROJECT_DESCRIPTION, disabled=True)

with col2:
    st.write("**Server Configuration**")
    st.text_input("Host", value=settings.HOST, disabled=True)
    st.number_input("Port", value=settings.PORT, disabled=True)
    st.text_input("API Prefix", value=settings.API_PREFIX, disabled=True)
    st.checkbox("Debug Mode", value=settings.DEBUG, disabled=True)

st.divider()

st.subheader("Environment Variables")
st.info("💡 To change these settings, set the corresponding environment variables before starting the application.")

st.code("""
# Example environment variables:
export PROJECT_NAME="My Application"
export PORT=8000
export DEBUG=True
export STREAMLIT_PORT=8501
""", language="bash")

