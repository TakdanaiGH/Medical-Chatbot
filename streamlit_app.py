import streamlit as st
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from urllib.parse import urlparse
from dotenv import load_dotenv
import os
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()

embedding_model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")  # fallback if not found
embedding_port = os.getenv("EMBEDDING_PORT", "http://localhost:8555/v1")
model_name = os.getenv("LLM_MODEL", "unsloth/Qwen3-8B-FP8")
model_port = os.getenv("LLM_PORT", "http://localhost:8444/v1")
json_path = os.getenv("RAG_JSON", "agnos_threads.json")
key_api = os.getenv("KEY_API", "EMTRY")
# system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")  # fallback if not set

@st.cache_data
def load_documents():
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    embedding_texts = []
    metadatas = []

    for item in data:
        content = item.get("content", "").strip()
        doc_answer = "\n\nคำตอบจากแพทย์:\n" + "\n\n".join(item["doctor_answers"]) if item.get("doctor_answers") else "ยังไม่มีคำตอบจากแพทย์"
        full_text = content + doc_answer

        url = item.get("url", f"https://www.agnoshealth.com/forums")  # fallback if missing

        documents.append(Document(
            page_content=full_text,
            metadata={
                "id": item["id"],
                "embedding_text": content,
                "url": url
            }
        ))

        embedding_texts.append(content)
        metadatas.append({
            "id": item["id"],
            "url": url
        })

    return documents, embedding_texts, metadatas

documents, embedding_texts, metadatas = load_documents()

# --- Step 2: Setup embeddings ---
@st.cache_resource(show_spinner=False)
def create_vectorstore():
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=key_api,
        openai_api_base=embedding_port
    )
    vectorstore = FAISS.from_texts(embedding_texts, embeddings, metadatas=metadatas)
    for i, doc in enumerate(documents):
        vectorstore.docstore._dict[i] = doc
    return vectorstore

vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# --- Step 3: Streaming callback handler for displaying tokens in Streamlit ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.buffer = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.buffer += token
        # Update the "Answer" tab dynamically
        if "answer_output" in st.session_state:
            st.session_state["answer_output"].markdown(self.buffer)

callback_handler = StreamlitCallbackHandler()
callback_manager = CallbackManager([callback_handler])

# --- Step 4: Setup LLM ---
@st.cache_resource(show_spinner=False)
def create_llm():
    return ChatOpenAI(
        openai_api_key=key_api,
        openai_api_base=model_port,
        model_name=model_name,
        max_tokens=8192,
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )

llm = create_llm()

# --- Step 5: Create RAG chain ---
@st.cache_resource(show_spinner=False)
def create_rag_chain():
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

rag_chain = create_rag_chain()

def main():

    st.set_page_config("Medical Chatbot", layout="wide")

    # Two columns: FAQ left (1/4 width), chat right (3/4 width)
    col1, col2 = st.columns([1, 3])

    with col1:
        # FAQ on the left column
        st.header("📋 FAQ (Click to ask)")
        faq_questions = [
            "ปวดหัว รักษาเบื้องต้นอย่างไร",
            "ปฐมพยาบาลเบื้องต้นอย่างไร",
            "มีเลือดกำเดาไหล หยุดอย่างไร",
            "มีอาการซึมเศร้า รักษาอย่างไร",
            "วิธีป้องกันโรคติดต่อ",
            "อาหารช่วยลด อาการไอ มีเสมหะ",
            "กินข้าวไม่ตรงเวลามีผลอะไรไหม"
        ]
        for q in faq_questions:
            if st.button(q, key=f"faq_{q}"):
                st.session_state.pending_query = q
                st.session_state.streaming_answer = ""
                st.session_state.stream_container = None
                st.rerun()

    with col2:
        # Put image + title at the top of the right column (chat area)
        col_img, col_title = st.columns([1, 6])
        with col_img:
            st.image("agnos-images.png", width=40)
        with col_title:
            st.markdown("### Medical Chatbot")

        # Initialize session state variables
        if "history" not in st.session_state:
            st.session_state.history = []
        if "pending_query" not in st.session_state:
            st.session_state.pending_query = None
        if "streaming_answer" not in st.session_state:
            st.session_state.streaming_answer = ""
        if "stream_container" not in st.session_state:
            st.session_state.stream_container = None
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                return_messages=True,
            )


        # --- Streaming callback handler ---
        class StreamlitCallbackHandler(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs):
                st.session_state.streaming_answer += token
                if st.session_state.stream_container:
                    st.session_state.stream_container.markdown(st.session_state.streaming_answer)

        callback_handler = StreamlitCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        # --- LLM setup ---
        llm = ChatOpenAI(
            openai_api_key=key_api,
            openai_api_base=model_port,
            model_name=model_name,
            temperature=0.7,
            max_tokens=8192,
            streaming=True,
            callback_manager=callback_manager,
            verbose=True,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,  # Make sure `retriever` is defined globally
            memory=st.session_state.memory,
            return_source_documents=True,
            output_key="answer",
        )

        # --- Display previous chat history ---
        for turn in st.session_state.history:
            with st.container():
                st.markdown(f"#### ❓ You: {turn['question']}")
                tab1, tab2 = st.tabs(["💬 Answer", "📚 Sources"])
                with tab1:
                    st.markdown(turn["answer"])
                with tab2:
                    st.markdown(turn["sources"], unsafe_allow_html=True)

        # --- Process pending query ---
        if st.session_state.pending_query:
            query = st.session_state.pending_query

            st.markdown(f"#### ❓ You: {query}")
            tab1, tab2 = st.tabs(["💬 Answer", "📚 Sources"])

            # Create containers for streaming
            st.session_state.stream_container = tab1.empty()
            st.session_state.stream_container.markdown("🔍 Generating...")

            source_container = tab2.empty()
            source_container.markdown("⌛ Waiting for sources...")

            try:
                result = conv_chain({"question": query})

                st.session_state.stream_container.markdown(result["answer"])

                source_md = ""
                for doc in result["source_documents"]:
                    preview = doc.page_content[:200].replace("\n", " ").strip()
                    url = doc.metadata.get("url", "#")
                    parsed_url = urlparse(url)
                    short_url = f"{parsed_url.netloc}{parsed_url.path[:20]}..."

                    source_md += f"""
                    <div style="
                        background-color: #d3d3d3;
                        color: #000000;
                        padding: 10px;
                        border-radius: 6px;
                        margin-bottom: 10px;">
                        <strong>🔗 <a href="{url}" target="_blank" style="color: #1a0dab;">{short_url}</a></strong><br>
                        {preview}...
                    </div>
                    """
                source_container.markdown(source_md, unsafe_allow_html=True)

                st.session_state.history.append({
                    "question": query,
                    "answer": result["answer"],
                    "sources": source_md
                })

            except Exception as e:
                st.error(f"❌ Error: {e}")

            # Cleanup
            st.session_state.pending_query = None
            st.session_state.streaming_answer = ""
            st.session_state.stream_container = None
            st.rerun()

        # --- Input form ---
        st.markdown("---")
        st.markdown("#### 💬 Ask a new question")
        with st.form("input_form", clear_on_submit=True):
            user_input = st.text_input("Enter your question:")
            submitted = st.form_submit_button("Ask")

        if submitted and user_input.strip():
            st.session_state.pending_query = user_input.strip()
            st.rerun()

if __name__ == "__main__":
    main()
