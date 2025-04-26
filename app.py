### app.py

import streamlit as st
from auth import login
from rag import setup_rag
from chat import main_chat

st.set_page_config(page_title="Streamlit 챗봇 with RAG", layout="wide")

# 세션 초기화
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "index" not in st.session_state:
    st.session_state.index = None
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 앱 흐름
if not st.session_state.logged_in:
    login()
else:
    setup_rag()
    main_chat()

