### auth.py

import streamlit as st

def login():
    st.title("🔒 로그인")
    user_id = st.text_input("아이디", key="user_id")
    user_pw = st.text_input("비밀번호", type="password", key="user_pw")

    if st.button("로그인"):
        if user_id == "seah" and user_pw == "minji":
            st.session_state.logged_in = True
            st.success("로그인 성공!")
            st.rerun()
        else:
            st.error("아이디 또는 비밀번호가 잘못되었습니다.")

