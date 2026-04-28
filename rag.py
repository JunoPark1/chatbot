import streamlit as st
from openai import OpenAI

def setup_rag():
    with st.sidebar:
        # OpenAI API Key를 Streamlit Secret에서 가져옴
        if "openai_api_key" not in st.secrets:
            st.error("OpenAI API 키가 설정되어 있지 않습니다. .streamlit/secrets.toml 파일을 확인하세요.")
            st.stop()

        client = OpenAI(api_key=st.secrets["openai_api_key"])

        model_options = ["gpt-5.4-mini", "gpt-5.5", "gpt-5.4"]
        preset_options = ["빠른응답", "균형", "정밀"]
        preset_map = {
            "빠른응답": {"reasoning": "low", "verbosity": "low"},
            "균형": {"reasoning": "medium", "verbosity": "medium"},
            "정밀": {"reasoning": "high", "verbosity": "high"},
        }

        if "settings_open" not in st.session_state:
            st.session_state.settings_open = False
        if "web_search_enabled" not in st.session_state:
            st.session_state.web_search_enabled = True
        # 과거 기본값(False)으로 저장된 세션 1회 마이그레이션
        if "web_search_default_migrated" not in st.session_state:
            st.session_state.web_search_enabled = True
            st.session_state.web_search_default_migrated = True
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "gpt-5.4-mini"
        # 과거 기본값/세션값(예: gpt-5.5) 1회 마이그레이션
        if "selected_model_default_migrated" not in st.session_state:
            st.session_state.selected_model = "gpt-5.4-mini"
            st.session_state.selected_model_default_migrated = True
        if "reasoning_by_model" not in st.session_state:
            st.session_state.reasoning_by_model = {model_name: "low" for model_name in model_options}
        if "verbosity_by_model" not in st.session_state:
            st.session_state.verbosity_by_model = {model_name: "low" for model_name in model_options}
        if "preset_by_model" not in st.session_state:
            st.session_state.preset_by_model = {model_name: "빠른응답" for model_name in model_options}
        if "last_applied_preset_by_model" not in st.session_state:
            st.session_state.last_applied_preset_by_model = {model_name: None for model_name in model_options}

        toggle_label = "⚙️ 설정 닫기" if st.session_state.settings_open else "⚙️ 설정 열기"
        if st.button(toggle_label, use_container_width=True):
            st.session_state.settings_open = not st.session_state.settings_open
            st.rerun()

        if st.session_state.settings_open:
            model = st.selectbox("모델", model_options, key="selected_model")

            selected_preset = st.selectbox(
                "프리셋",
                preset_options,
                index=preset_options.index(st.session_state.preset_by_model.get(model, "균형")),
                key=f"model_preset_{model}"
            )
            st.session_state.preset_by_model[model] = selected_preset

            if st.session_state.last_applied_preset_by_model.get(model) != selected_preset:
                preset_values = preset_map[selected_preset]
                st.session_state.reasoning_by_model[model] = preset_values["reasoning"]
                st.session_state.verbosity_by_model[model] = preset_values["verbosity"]
                st.session_state.last_applied_preset_by_model[model] = selected_preset

            # effort_options = ["low", "medium", "high"]
            # current_effort = st.session_state.reasoning_by_model.get(model, "medium")
            # selected_effort = st.selectbox(
            #     "Reasoning",
            #     effort_options,
            #     index=effort_options.index(current_effort if current_effort in effort_options else "medium"),
            #     key=f"reasoning_effort_{model}",
            # )
            # st.session_state.reasoning_by_model[model] = selected_effort

            # verbosity_options = ["low", "medium", "high"]
            # current_verbosity = st.session_state.verbosity_by_model.get(model, "medium")
            # selected_verbosity = st.selectbox(
            #     "Verbosity",
            #     verbosity_options,
            #     index=verbosity_options.index(current_verbosity if current_verbosity in verbosity_options else "medium"),
            #     key=f"verbosity_{model}",
            # )
            # st.session_state.verbosity_by_model[model] = selected_verbosity

            # Web Search 활성화 옵션
            # key를 사용하면 자동으로 st.session_state.web_search_enabled에 저장됨
            st.checkbox(
                "🌐 웹 검색 활성화",
                key="web_search_enabled",
                help="최신 정보를 웹에서 검색하여 답변에 포함합니다."
            )

            st.divider()

        if st.session_state.settings_open and "uploaded_file_ids" in st.session_state and st.session_state.uploaded_file_ids:
            st.info(f"📎 현재 {len(st.session_state.uploaded_file_ids)}개 파일이 첨부되어 있습니다.")
            if st.button("🗑️ 첨부 파일 제거"):
                for file_id in st.session_state.uploaded_file_ids:
                    try:
                        client.files.delete(file_id)
                    except:
                        pass
                st.session_state.uploaded_file_ids = []
                st.session_state.rag_mode = False
                st.success("첨부 파일을 제거했습니다.")
