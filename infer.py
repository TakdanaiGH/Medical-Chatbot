import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
import readline  # optional: enables arrow key history in input()

### STEP 1: Load JSON data ###
with open("agnos_threads.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []
embedding_texts = []
metadatas = []

for item in data:
    content = item.get("content", "").strip()
    doc_answer = "\n\nคำตอบจากแพทย์:\n" + "\n\n".join(item["doctor_answers"]) if item.get("doctor_answers") else "ยังไม่มีคำตอบจากแพทย์"
    full_text = content + doc_answer

    url = item.get("url", f"https://www.agnoshealth.com/forums")  # fallback to constructed URL if missing

    # Store full text for LLM, but only content for embedding
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

print(f"✅ Loaded and prepared {len(documents)} documents from agnos_threads.json.")

### STEP 2: Use local embedding model via OpenAI-compatible API ###
embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-0.6B",           # Must match model served by vLLM
    openai_api_key="not-needed",                # vLLM doesn't check this
    openai_api_base="http://localhost:8555/v1"  # Embedding model server
)

### STEP 3: Build FAISS vectorstore using only the content for embedding ###
vectorstore = FAISS.from_texts(embedding_texts, embeddings, metadatas=metadatas)

# Reattach full documents to FAISS for LLM context
for i, doc in enumerate(documents):
    vectorstore.docstore._dict[i] = doc

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})  # Return top 20 relevant

### STEP 4: Streaming callback handler for LLM ###
class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

### STEP 5: Setup Chat LLM (vLLM completion server) ###
llm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8444/v1",
    model_name="unsloth/Qwen3-8B-FP8",
    temperature=0.7,
    max_tokens=8192,
    top_p=0.8,
    streaming=True,
    callback_manager=callback_manager,
    verbose=True,
)

### STEP 6: Create RAG chain ###
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

### STEP 7: Interactive CLI loop ###
print("\n💬 Ready! Type your question (or 'exit' to quit):")
while True:
    try:
        query = input("🧠 Enter question (Thai OK): ").strip()
        if query.lower() in ("exit", "quit"):
            print("👋 Exiting.")
            break

        print("🔍 Answer:")
        result = rag_chain.invoke({"query": query})

        print("\n\n📚 Source Documents:")
        for doc in result["source_documents"]:
            preview = doc.page_content[:200].replace("\n", " ")
            url = doc.metadata.get("url", "URL not available")
            print(f"- ID: {doc.metadata['id']} | URL: {url} | {preview}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
