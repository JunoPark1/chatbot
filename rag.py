import streamlit as st
import openai
import tempfile
import os

from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.core import download_loader

PDFReader = download_loader("PDFReader")

def setup_rag():
    with st.sidebar:
        st.header("⚙️ 설정")

        # OpenAI API Key를 Streamlit Secret에서 가져옴
        if "openai_api_key" not in st.secrets:
            st.error("OpenAI API 키가 설정되어 있지 않습니다. .streamlit/secrets.toml 파일을 확인하세요.")
            st.stop()

        openai.api_key = st.secrets["openai_api_key"]

        model = st.selectbox("모델 선택", [
            "gpt-4o",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo"
        ], key="selected_model")

        # ✨ Settings에 기본 LLM 설정
        Settings.llm = OpenAI(model=model, temperature=0)

        uploaded_files = st.file_uploader(
            "RAG용 파일 업로드 (txt, pdf, md)", type=["txt", "pdf", "md"], accept_multiple_files=True
        )

        if uploaded_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                filepaths = []

                for f in uploaded_files:
                    path = os.path.join(tmpdir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.read())
                    filepaths.append(path)

                documents = []
                for path in filepaths:
                    if path.endswith(".pdf"):
                        reader = PDFReader()
                        documents.extend(reader.load_data(path))
                    else:
                        reader = SimpleDirectoryReader(input_files=[path])
                        documents.extend(reader.load_data())

                # ✨ service_context 없이 바로 인덱스 생성
                index = VectorStoreIndex.from_documents(documents)

                st.session_state.index = index
                st.session_state.rag_mode = True

            st.success("✅ 파일 업로드 및 인덱스 생성 완료")
