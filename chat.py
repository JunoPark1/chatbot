import streamlit as st
from openai import OpenAI
import tempfile
import os
import time
import base64
import html
import re

client = OpenAI(api_key=st.secrets["openai_api_key"])


def render_message(role, content):
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    normalized_content = re.sub(r"\n{3,}", "\n\n", content or "")

    if role == "assistant":
        # 어시스턴트 응답은 Markdown 렌더링을 유지해 가독성을 살린다.
        st.markdown(normalized_content)
    else:
        safe_content = html.escape(normalized_content).replace("\n", "<br>")
        st.markdown(
            f'<div class="chat-row {bubble_class}">{safe_content}</div>',
            unsafe_allow_html=True
        )


def render_generated_image(role, image_data_url, caption="생성된 이미지"):
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    safe_caption = html.escape(caption or "")
    st.markdown(
        f'<div class="chat-row {bubble_class}">{safe_caption}</div>',
        unsafe_allow_html=True
    )
    st.image(image_data_url, use_container_width=True)

def format_chat_history(chat_history, user_input, file_ids=None, image_inputs=None):
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
    
    # 파일/이미지가 있으면 content 배열 형식 사용, 없으면 문자열 사용
    if file_ids or image_inputs:
        user_content = []
        
        # 문서 파일은 input_file로 추가
        for file_id in (file_ids or []):
            user_content.append({"type": "input_file", "file_id": file_id})

        # 이미지는 input_image로 추가
        for image_url in (image_inputs or []):
            user_content.append({"type": "input_image", "image_url": image_url})
        
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


def upload_attachments(attached_files):
    supported_file_types = {
        "txt", "md", "markdown", "pdf", "html", "htm", "rtf",
        "doc", "docx", "ppt", "pptx", "xls", "xlsx",
        "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml",
        "py", "js", "ts", "tsx", "jsx", "java", "c", "cpp", "h", "hpp",
        "cs", "go", "rs", "rb", "php", "swift", "kt", "kts",
        "sql", "sh", "bash", "ps1", "toml", "ini", "cfg", "conf",
        "png", "jpg", "jpeg", "webp", "gif",
    }
    max_file_size_mb = 20
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    image_exts = {"jpg", "jpeg", "png", "gif", "webp"}

    valid_files = []
    for f in attached_files:
        ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        if ext in image_exts:
            # 이미지는 input_image로 별도 처리
            continue
        if ext not in supported_file_types:
            st.warning(f"❌ {f.name}: 지원되지 않는 파일 형식입니다.")
            continue
        if f.size > max_file_size_bytes:
            st.warning(f"❌ {f.name}: 파일 크기가 {max_file_size_mb}MB를 초과합니다.")
            continue
        valid_files.append(f)

    if not valid_files:
        return []

    uploaded_file_ids = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in valid_files:
            try:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
                with open(path, "rb") as fp:
                    file_obj = client.files.create(file=fp, purpose="user_data")
                    uploaded_file_ids.append(file_obj.id)
            except Exception as e:
                st.error(f"❌ {f.name}: 업로드 실패 - {str(e)}")

    # 업로드 직후 처리 지연으로 "file not found"가 날 수 있어 상태 확인 후 사용
    ready_file_ids = []
    pending_file_ids = []
    for file_id in uploaded_file_ids:
        is_ready = False
        for _ in range(8):  # 최대 약 8초 대기
            try:
                file_info = client.files.retrieve(file_id)
                status = getattr(file_info, "status", None)
                if status in ("processed", "ready", "completed", None):
                    is_ready = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if is_ready:
            ready_file_ids.append(file_id)
        else:
            pending_file_ids.append(file_id)

    if pending_file_ids:
        st.warning("일부 첨부 파일 처리가 지연되어 이번 질문에서는 제외되었습니다. 잠시 후 다시 시도해주세요.")

    return ready_file_ids


def extract_image_inputs(attached_files):
    image_ext_to_mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    image_inputs = []
    for f in attached_files:
        ext = f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        mime_type = image_ext_to_mime.get(ext)
        if not mime_type:
            continue
        try:
            encoded = base64.b64encode(f.getvalue()).decode("utf-8")
            image_inputs.append(f"data:{mime_type};base64,{encoded}")
        except Exception as e:
            st.error(f"❌ {f.name}: 이미지 처리 실패 - {str(e)}")
    return image_inputs


def filter_existing_file_ids(file_ids):
    existing = []
    missing = []
    for file_id in file_ids:
        try:
            client.files.retrieve(file_id)
            existing.append(file_id)
        except Exception:
            missing.append(file_id)
    return existing, missing


def clear_uploaded_files():
    current_file_ids = st.session_state.get("uploaded_file_ids", [])
    for file_id in current_file_ids:
        try:
            client.files.delete(file_id)
        except Exception:
            pass
    st.session_state.uploaded_file_ids = []
    st.session_state.rag_mode = False


def extract_generated_images(response):
    image_urls = []
    for item in getattr(response, "output", []) or []:
        # image_generation_call 출력 처리
        if getattr(item, "type", "") == "image_generation_call":
            result = getattr(item, "result", None)
            if isinstance(result, str) and result:
                image_urls.append(f"data:image/png;base64,{result}")
        # 일부 SDK 포맷 호환 처리
        if hasattr(item, "image_base64") and getattr(item, "image_base64"):
            image_urls.append(f"data:image/png;base64,{item.image_base64}")
    return image_urls


def generate_images_with_api(prompt, model):
    image_urls = []
    response = None
    used_model = model
    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
        )
    except Exception as e:
        error_text = str(e)
        if "must be verified to use the model gpt-image-2" in error_text and model.startswith("gpt-image-2"):
            # 조직 인증 전에는 gpt-image-1로 자동 폴백
            used_model = "gpt-image-1"
            response = client.images.generate(
                model=used_model,
                prompt=prompt,
                size="1024x1024",
            )
            st.info("ℹ️ 현재 조직 인증이 완료되지 않아 `gpt-image-1`으로 자동 전환해 생성했습니다.")
        else:
            raise
    for item in getattr(response, "data", []) or []:
        b64 = getattr(item, "b64_json", None)
        url = getattr(item, "url", None)
        if b64:
            image_urls.append(f"data:image/png;base64,{b64}")
        elif url:
            image_urls.append(url)
    return image_urls


def main_chat():
    # st.title("무엇이든 물어보세요")
    # st.write("---")

    for chat in st.session_state.chat_history:
        if chat.get("image_data_url"):
            render_generated_image(chat["role"], chat["image_data_url"], chat.get("content", "생성된 이미지"))
        else:
            render_message(chat["role"], chat["content"])

    current_file_ids = st.session_state.get("uploaded_file_ids", [])
    if current_file_ids:
        st.caption(f"첨부 파일 {len(current_file_ids)}개가 활성화되어 있습니다.")
        if st.button("🗑️ 첨부파일 제거", key="chat_clear_attachments"):
            clear_uploaded_files()
            st.success("첨부 파일을 제거했습니다.")
            st.rerun()

    if "image_mode_enabled" not in st.session_state:
        st.session_state.image_mode_enabled = False
    if "image_model" not in st.session_state:
        st.session_state.image_model = "gpt-image-1"

    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    supported_file_types = [
        "txt", "md", "markdown", "pdf", "html", "htm", "rtf",
        "doc", "docx", "ppt", "pptx", "xls", "xlsx",
        "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml",
        "py", "js", "ts", "tsx", "jsx", "java", "c", "cpp", "h", "hpp",
        "cs", "go", "rs", "rb", "php", "swift", "kt", "kts",
        "sql", "sh", "bash", "ps1", "toml", "ini", "cfg", "conf",
        "png", "jpg", "jpeg", "webp", "gif",
    ]

    if st.session_state.is_generating:
        if st.button("⏹️ 답변 중지", use_container_width=True):
            st.session_state.stop_requested = True

    try:
        # 입력창 바로 위에서 이미지 생성 모드 토글
        icon_col, mode_text_col = st.columns([0.10, 0.90], vertical_alignment="center")
        with icon_col:
            icon_label = "✅ 🖼️" if st.session_state.image_mode_enabled else "🖼️"
            if st.button(icon_label, key="toggle_image_mode", help="이미지 생성 모드 전환"):
                st.session_state.image_mode_enabled = not st.session_state.image_mode_enabled
                st.rerun()
        with mode_text_col:
            if st.session_state.image_mode_enabled:
                st.caption("이미지 생성 (`gpt-image-1`)")

        user_input_payload = st.chat_input(
            "메시지를 입력하세요...",
            accept_file="multiple",
            file_type=supported_file_types,
        )
    except TypeError:
        user_input_payload = st.chat_input("메시지를 입력하세요...")

    attached_files = []
    user_input = None
    if isinstance(user_input_payload, str):
        user_input = user_input_payload
    elif user_input_payload is not None:
        user_input = user_input_payload.text
        attached_files = user_input_payload.files or []

    should_submit = user_input is not None and (str(user_input).strip() != "" or len(attached_files) > 0)

    if should_submit:
        st.session_state.is_generating = True
        st.session_state.stop_requested = False
        visible_user_text = user_input if str(user_input).strip() else "(파일 첨부)"
        render_message("user", visible_user_text)
        st.session_state.chat_history.append({"role": "user", "content": visible_user_text})

        with st.spinner("답변 생성 중..."):
            # 파일 ID 가져오기 (디버깅 정보 포함)
            file_ids = []
            image_inputs = []
            uploaded_file_ids = st.session_state.get("uploaded_file_ids", [])

            if attached_files and not st.session_state.image_mode_enabled:
                image_inputs = extract_image_inputs(attached_files)
                newly_uploaded = upload_attachments(attached_files)
                file_ids.extend(newly_uploaded)
                st.session_state.uploaded_file_ids = newly_uploaded
                st.session_state.rag_mode = bool(newly_uploaded)
            elif attached_files and st.session_state.image_mode_enabled:
                st.info("이미지 생성 모드에서는 첨부 파일을 참고하지 않고 프롬프트 텍스트로 생성합니다.")
            elif uploaded_file_ids:
                file_ids = uploaded_file_ids

            file_ids, missing_file_ids = filter_existing_file_ids(file_ids)
            if missing_file_ids:
                st.warning("일부 첨부 파일을 찾지 못해 제외했습니다.")
                st.session_state.uploaded_file_ids = file_ids
                st.session_state.rag_mode = bool(file_ids)
            
            # 디버깅 정보
            if st.session_state.get("debug_mode", False):
                st.write(f"🔍 디버그: uploaded_file_ids={uploaded_file_ids}, current_file_ids={file_ids}, image_count={len(image_inputs)}")
            
            # 대화 히스토리를 Responses API의 input 배열 형식으로 변환
            input_array = format_chat_history(
                st.session_state.chat_history[:-1],
                user_input or "",
                file_ids,
                image_inputs,
            )
            
            # Responses API 호출 준비
            model_name = st.session_state.selected_model
            reasoning_by_model = st.session_state.get("reasoning_by_model", {})
            reasoning_effort = reasoning_by_model.get(model_name, "medium")
            verbosity_by_model = st.session_state.get("verbosity_by_model", {})
            text_verbosity = verbosity_by_model.get(model_name, "medium")

            request_params = {
                "model": model_name,
                "input": input_array,
                "reasoning": {"effort": reasoning_effort},
                "text": {"verbosity": text_verbosity}
            }
            
            # Web Search가 활성화되어 있으면 tools에 추가
            if not st.session_state.image_mode_enabled and st.session_state.get("web_search_enabled", False):
                request_params["tools"] = [
                    {"type": "web_search"}
                ]
            
            # previous_response_id가 있으면 추가
            if (
                not st.session_state.image_mode_enabled
                and "previous_response_id" in st.session_state
                and st.session_state.previous_response_id
            ):
                request_params["previous_response_id"] = st.session_state.previous_response_id
            
            try:
                if st.session_state.stop_requested:
                    st.info("⏹️ 답변 생성을 중지했습니다.")
                else:
                    if st.session_state.image_mode_enabled:
                        image_urls = generate_images_with_api(
                            user_input or "",
                            st.session_state.image_model,
                        )
                        if image_urls:
                            for idx, image_url in enumerate(image_urls, start=1):
                                render_generated_image("assistant", image_url, f"생성된 이미지 {idx}")
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"[이미지 생성 {idx}]",
                                    "image_data_url": image_url,
                                })
                        else:
                            fallback_msg = "이미지 생성 결과를 받지 못했습니다."
                            render_message("assistant", fallback_msg)
                            st.session_state.chat_history.append({"role": "assistant", "content": fallback_msg})
                    else:
                        # Responses API 호출
                        response = client.responses.create(**request_params)
                        reply = response.output_text
                        if hasattr(response, 'id'):
                            st.session_state.previous_response_id = response.id
                        render_message("assistant", reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                error_msg = str(e)
                # 파일 관련 오류인 경우
                if (
                    not st.session_state.image_mode_enabled
                    and "Files" in error_msg
                    and "were not found" in error_msg
                ):
                    # 이전 응답 체인에 사라진 파일이 연결된 경우 재시도
                    retry_params = dict(request_params)
                    retry_params.pop("previous_response_id", None)

                    retry_file_ids, retry_missing = filter_existing_file_ids(file_ids)
                    if retry_missing:
                        st.warning("일부 파일이 만료/삭제되어 제외 후 재시도합니다.")
                    retry_input = format_chat_history(
                        st.session_state.chat_history[:-1],
                        user_input or "",
                        retry_file_ids,
                        image_inputs,
                    )
                    retry_params["input"] = retry_input

                    try:
                        response = client.responses.create(**retry_params)
                        reply = response.output_text
                        st.session_state.previous_response_id = getattr(response, "id", None)
                        st.session_state.uploaded_file_ids = retry_file_ids
                        st.session_state.rag_mode = bool(retry_file_ids)
                        render_message("assistant", reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    except Exception:
                        current_file_ids = st.session_state.get("uploaded_file_ids", [])
                        st.error("❌ 업로드된 파일을 찾을 수 없습니다.")
                        st.error(f"파일 ID: {current_file_ids}")
                        st.info("💡 파일이 업로드된 직후에는 처리 시간이 필요할 수 있습니다. 잠시 후 다시 시도해주세요.")
                        st.info("💡 또는 파일을 다시 업로드해주세요.")
                else:
                    st.error(f"❌ 오류가 발생했습니다: {error_msg}")
                    # 디버깅 정보
                    if st.session_state.get("debug_mode", False):
                        st.write(f"🔍 디버그: file_ids={file_ids}, uploaded_file_ids={st.session_state.get('uploaded_file_ids')}")
            finally:
                st.session_state.is_generating = False
                st.session_state.stop_requested = False
