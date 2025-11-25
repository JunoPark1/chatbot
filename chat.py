import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["openai_api_key"])

def format_chat_history(chat_history, user_input, file_ids=None):
    """대화 히스토리를 Responses API의 input 배열 형식으로 변환"""
    input_array = []
    
    # 시스템 프롬프트를 첫 번째 사용자 메시지에 포함
    system_text = "당신은 친절한 AI 어시스턴트입니다. 답변은 아주 자세히 해주세요. 출처가 모호하거나 애매한 부분은 대답하지 않거나 다시 물어봐야 합니다. 논리적으로 대답하세요.\n\n"
    
    # 대화 히스토리를 텍스트로 변환
    conversation_text = system_text
    for chat in chat_history:
        if chat["role"] == "user":
            conversation_text += f"사용자: {chat['content']}\n\n"
        elif chat["role"] == "assistant":
            conversation_text += f"어시스턴트: {chat['content']}\n\n"
    
    # 파일이 있으면 content 배열 형식 사용, 없으면 문자열 사용
    if file_ids:
        # 파일이 있을 때: 파일과 텍스트를 content 배열에 포함
        # content 배열의 모든 요소는 객체 형식이어야 함
        user_content = []
        
        # 먼저 파일 추가
        for file_id in file_ids:
            user_content.append({
                "type": "input_file",
                "file_id": file_id
            })
        
        # 텍스트는 객체 형식으로 추가
        # 사용자 예제 형식: type과 text 필드 사용
        user_content.append({
            "type": "input_text",
            "text": user_input
        })
        
        # 대화 히스토리가 있으면 별도 메시지로 추가
        if chat_history:
            for chat in chat_history:
                input_array.append({
                    "role": chat["role"],
                    "content": chat["content"]
                })
        
        # 현재 사용자 메시지 추가 (파일 포함)
        input_array.append({
            "role": "user",
            "content": user_content
        })
        
        # 시스템 프롬프트는 첫 번째로 추가
        if system_text.strip():
            input_array.insert(0, {
                "role": "system",
                "content": system_text.strip()
            })
    else:
        # 파일이 없을 때: 단순 문자열 사용
        full_text = conversation_text + f"사용자: {user_input}"
        input_array.append({
            "role": "user",
            "content": full_text
        })
    
    return input_array

def main_chat():
    # st.title("무엇이든 물어보세요")
    # st.write("---")

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    user_input = st.chat_input("메시지를 입력하세요...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("답변 생성 중..."):
            # 파일 ID 가져오기 (디버깅 정보 포함)
            file_ids = None
            rag_mode = st.session_state.get("rag_mode", False)
            uploaded_file_ids = st.session_state.get("uploaded_file_ids", [])
            
            # 디버깅 정보
            if st.session_state.get("debug_mode", False):
                st.write(f"🔍 디버그: rag_mode={rag_mode}, uploaded_file_ids={uploaded_file_ids}")
            
            if rag_mode and uploaded_file_ids:
                file_ids = uploaded_file_ids
                if st.session_state.get("debug_mode", False):
                    st.write(f"🔍 디버그: file_ids={file_ids}")
            
            # 대화 히스토리를 Responses API의 input 배열 형식으로 변환
            input_array = format_chat_history(st.session_state.chat_history[:-1], user_input, file_ids)
            
            # Responses API 호출 준비
            request_params = {
                "model": st.session_state.selected_model,
                "input": input_array,
                "reasoning": {"effort": "medium"},
                "text": {"verbosity": "medium"}
            }
            
            # Web Search가 활성화되어 있으면 tools에 추가
            if st.session_state.get("web_search_enabled", False):
                request_params["tools"] = [
                    {"type": "web_search"}
                ]
            
            # previous_response_id가 있으면 추가
            if "previous_response_id" in st.session_state and st.session_state.previous_response_id:
                request_params["previous_response_id"] = st.session_state.previous_response_id
            
            try:
                # Responses API 호출
                response = client.responses.create(**request_params)
                
                # 응답 파싱
                reply = response.output_text
                
                # response_id 저장 (다음 요청에 사용)
                if hasattr(response, 'id'):
                    st.session_state.previous_response_id = response.id

                st.chat_message("assistant").markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                error_msg = str(e)
                # 파일 관련 오류인 경우
                if "Files" in error_msg and "were not found" in error_msg:
                    # 세션 상태 확인
                    current_rag_mode = st.session_state.get("rag_mode", False)
                    current_file_ids = st.session_state.get("uploaded_file_ids", [])
                    
                    if current_file_ids:
                        st.error(f"❌ 업로드된 파일을 찾을 수 없습니다.")
                        st.error(f"파일 ID: {current_file_ids}")
                        st.info("💡 파일이 업로드된 직후에는 처리 시간이 필요할 수 있습니다. 잠시 후 다시 시도해주세요.")
                        st.info("💡 또는 파일을 다시 업로드해주세요.")
                    else:
                        st.error("❌ 파일 ID가 세션 상태에 없습니다.")
                        st.error(f"rag_mode: {current_rag_mode}, uploaded_file_ids: {current_file_ids}")
                        st.info("💡 파일을 다시 업로드해주세요.")
                        st.session_state.rag_mode = False
                else:
                    st.error(f"❌ 오류가 발생했습니다: {error_msg}")
                    # 디버깅 정보
                    if st.session_state.get("debug_mode", False):
                        st.write(f"🔍 디버그: file_ids={file_ids}, rag_mode={st.session_state.get('rag_mode')}, uploaded_file_ids={st.session_state.get('uploaded_file_ids')}")
