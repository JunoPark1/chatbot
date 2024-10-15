import streamlit as st
import os
from login_screen import login_screen
from main_chat import main_chat

# 세션 상태 초기화
def initialize_session():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

initialize_session()

# Streamlit 앱 실행
if st.session_state.get('authenticated', False):
    main_chat()
else:
    login_screen()