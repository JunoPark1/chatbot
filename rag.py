import streamlit as st
from openai import OpenAI
import tempfile
import os

def setup_rag():
    with st.sidebar:
        st.header("⚙️ 설정")

        # OpenAI API Key를 Streamlit Secret에서 가져옴
        if "openai_api_key" not in st.secrets:
            st.error("OpenAI API 키가 설정되어 있지 않습니다. .streamlit/secrets.toml 파일을 확인하세요.")
            st.stop()

        client = OpenAI(api_key=st.secrets["openai_api_key"])

        model = st.selectbox("모델 선택", [
            "gpt-5.1",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5"
        ], key="selected_model", index=0)

        # Web Search 활성화 옵션
        # key를 사용하면 자동으로 st.session_state.web_search_enabled에 저장됨
        st.checkbox(
            "🌐 웹 검색 활성화",
            value=st.session_state.get("web_search_enabled", False),
            key="web_search_enabled",
            help="최신 정보를 웹에서 검색하여 답변에 포함합니다."
        )

        st.divider()

        uploaded_files = st.file_uploader(
            "파일 업로드 (txt, pdf, md)", type=["txt", "pdf", "md"], accept_multiple_files=True
        )

        # 현재 업로드된 파일 상태 표시
        if "uploaded_file_ids" in st.session_state and st.session_state.uploaded_file_ids:
            st.info(f"📎 현재 {len(st.session_state.uploaded_file_ids)}개 파일이 업로드되어 있습니다.")
            if st.button("🗑️ 파일 제거"):
                for file_id in st.session_state.uploaded_file_ids:
                    try:
                        client.files.delete(file_id)
                    except:
                        pass
                del st.session_state.uploaded_file_ids
                st.session_state.rag_mode = False
                st.rerun()

        if uploaded_files:
            # 기존 파일이 있으면 먼저 삭제
            if "uploaded_file_ids" in st.session_state and st.session_state.uploaded_file_ids:
                for file_id in st.session_state.uploaded_file_ids:
                    try:
                        client.files.delete(file_id)
                    except:
                        pass
            
            uploaded_file_ids = []
            
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    # 임시 파일로 저장
                    path = os.path.join(tmpdir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.read())
                    
                    # OpenAI에 파일 업로드
                    with open(path, "rb") as file:
                        file_obj = client.files.create(
                            file=file,
                            purpose="user_data"
                        )
                        uploaded_file_ids.append(file_obj.id)
                        
                        # 파일 업로드 후 상태 확인 (디버깅용)
                        try:
                            file_info = client.files.retrieve(file_obj.id)
                            if hasattr(file_info, 'status'):
                                st.write(f"📄 {f.name}: {file_info.status}")
                        except Exception as e:
                            st.write(f"⚠️ {f.name} 상태 확인 실패: {str(e)}")
            
            st.session_state.uploaded_file_ids = uploaded_file_ids
            st.session_state.rag_mode = True
            st.success(f"✅ {len(uploaded_file_ids)}개 파일 업로드 완료")
            if uploaded_file_ids:
                st.info(f"📋 파일 ID: {', '.join(uploaded_file_ids)}")
        
        # 파일이 업로드되지 않았을 때도 기존 파일이 있으면 rag_mode 유지
        if not uploaded_files and "uploaded_file_ids" in st.session_state and st.session_state.uploaded_file_ids:
            st.session_state.rag_mode = True
