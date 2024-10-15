import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from openai import OpenAI
import io
from io import StringIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# # RAG 구성 함수
# def create_rag(files):
#     documents = []
#     for file in files:
#         if file.type == "application/pdf":
#             loader = PyPDFLoader(file.path)
#         elif file.type == "text/plain":
#             content = file.read().decode("utf-8")
#             loader = TextLoader(io.StringIO(content))
#         else:
#             st.error("Unsupported file type.")
#             return None
        
#         docs = loader.load()
#         print(len(docs))
#         # documents.extend(loader.load())
#         # for doc in loader.lazy_load():
#         #     print(doc.metadata)
#         #     documents.extend(doc)
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_documents(documents, embeddings)
#     retriever = vector_store.as_retriever()
#     return RetrievalQA(llm=OpenAI(), retriever=retriever)

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

# RAG 구성 함수
def create_rag(files):

    raw_text = pdf_read(files)
    text_chunks = get_chunks(raw_text)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    retriever=vector_store.as_retriever()

    prompt = PromptTemplate.from_template(
        """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
    검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
    한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

    #Question:
    {question}

    #Context:
    {context}

    #Answer:"""
    )

    llm = ChatOpenAI(model_name=st.session_state.selected_model, temperature=0)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# 메인 채팅 화면 구성
def main_chat():
    # st.sidebar.write("\n".join([f"Conversation {i+1}: {msg['content'][:20]}..." for i, msg in enumerate(st.session_state['messages'][-10:])]))
    selected_model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = selected_model
    temperature = st.sidebar.slider("Temperature", min_value=0., max_value=1., step=0.1)
    # 채팅 대화 기록 저장
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 대화 기록 출력
    # st.title("Chat with GPT")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message('user', avatar="👩‍🎓"):
                st.write(msg['content'])
        else:
            with st.chat_message('assistant', avatar="🧠"):
                st.write(msg['content'])

    # 파일 업로드 및 RAG 구성
    uploaded_files = st.sidebar.file_uploader("Upload PDF or TXT files", accept_multiple_files=True, type=["pdf", "txt"])
    if uploaded_files:
        rag_chain = create_rag(uploaded_files)
    else:
        rag_chain = None

    user_input = st.chat_input("Message GPT")

    if user_input:
        with st.spinner("Thinking..."):
            llm = OpenAI()
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            if rag_chain:
                assistant_message = rag_chain.invoke(user_input)
            else:
                answer = llm.chat.completions.create(
                    model=selected_model,
                    messages=st.session_state.messages,
                    temperature=temperature,
                )
                assistant_message = answer.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})

            with st.chat_message('user', avatar="👩‍🎓"):
                st.write(user_input)
            with st.chat_message('assistant', avatar="🧠"):
                st.write(assistant_message)
