import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["openai_api_key"])

def main_chat():
    st.title("무엇이든 물어보세요")
    st.write("---")

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    user_input = st.chat_input("메시지를 입력하세요...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("답변 생성 중..."):
            if st.session_state.rag_mode and st.session_state.index:
                query_engine = st.session_state.index.as_query_engine()
                response = query_engine.query(user_input)
                reply = response.response
            else:
                response = client.chat.completions.create(
                    model=st.session_state.selected_model,
                    messages=[
                        {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
                        *[
                            {"role": c["role"], "content": c["content"]}
                            for c in st.session_state.chat_history
                        ],
                        {"role": "user", "content": user_input},
                    ],
                )
                reply = response.choices[0].message.content

            st.chat_message("assistant").markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
