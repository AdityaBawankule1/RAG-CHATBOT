import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Local Free RAG Chatbot", page_icon="🤖")
st.title("🤖 Local RAG Chatbot (Ollama + FAISS)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Force HuggingFace to use CPU (prevents Metal GPU driver conflicts on Mac)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = load_embeddings()

with st.sidebar:
    st.header("📄 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Indexing document..."):
                # Safely handle temp file on macOS
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    loader = TextLoader(tmp_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
                    chunks = text_splitter.split_documents(docs)

                    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                    st.success("Document indexed successfully!")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_store is None:
        with st.chat_message("assistant"):
            warning_msg = "Please upload and process a text document in the sidebar first!"
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    else:
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})
        llm = ChatOllama(model="llama3.2", temperature=0)

        template = """Answer the question based only on the following context:
{context}

Question: {question}
Answer:"""
        prompt_template = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})