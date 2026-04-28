### app.py

import streamlit as st
from auth import login
from rag import setup_rag
from chat import main_chat

st.set_page_config(page_title="Assistant Workspace", layout="wide")

st.markdown(
    """
    <style>
    .chat-row {
        max-width: 80%;
        padding: 0.7rem 0.9rem;
        border-radius: 0.8rem;
        margin: 0.35rem 0 0.65rem 0;
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .user-bubble {
        margin-left: auto;
        background: #e8f1ff;
        color: #0f172a;
    }
    .assistant-bubble {
        margin-right: auto;
        background: #f3f4f6;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 세션 초기화
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None
if "uploaded_file_ids" not in st.session_state:
    st.session_state.uploaded_file_ids = []
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = True

# 앱 흐름
if not st.session_state.logged_in:
    login()
else:
    setup_rag()
    main_chat()

