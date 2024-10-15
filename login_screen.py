import streamlit as st
import openai

# 로그인 화면 구성
def login_screen():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "seah" and password == "minji":
            st.session_state['authenticated'] = True
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            st.rerun()
        else:
            st.error("Invalid username or password")