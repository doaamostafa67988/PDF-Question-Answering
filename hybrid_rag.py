import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
#from langchain.retrievers import EnsembleRetriever
#from langchain.retrievers import EnsembleRetriever
#from langchain_community.retrievers import EnsembleRetriever
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class EnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float]
    
    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        # Get documents from all retrievers
        all_documents = []
        for retriever in self.retrievers:
            #docs = retriever.get_relevant_documents(query)
            docs = retriever._get_relevant_documents(query=query, run_manager=run_manager)
            all_documents.append(docs)
        
        # Use Reciprocal Rank Fusion to combine results
        doc_scores = {}
        for i, docs in enumerate(all_documents):
            for rank, doc in enumerate(docs):
                doc_id = doc.page_content
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0}
                # RRF formula: score += weight / (rank + 60)
                doc_scores[doc_id]["score"] += self.weights[i] / (rank + 60)
        
        # Sort by score and return documents
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = '/tmp'

model = OllamaLLM(model="llama3.2:3b")

def upload_pdf(file):
    #pdfs_directory = "Hybird_Rag/pdfs"

    #  Create directory if it doesn't exist
    os.makedirs(pdfs_directory, exist_ok=True)

    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
        
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True
    )

    return text_splitter.split_documents(documents)

def build_semantic_retriever(documents):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)

    return vector_store.as_retriever()

def build_bm25_retriever(documents):
    return BM25Retriever.from_documents(documents, preprocess_func=word_tokenize)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("Hybrid PDF Question-Answering")
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    #upload_pdf(uploaded_file)
    #documents = load_pdf(pdfs_directory + uploaded_file.name)
    pdf_path = upload_pdf(uploaded_file)
    if not os.path.exists(pdf_path):
        st.error("PDF was not saved correctly.")
        st.stop()
    documents = load_pdf(pdf_path)
    chunked_documents = split_text(documents)

    semantic_retriever = build_semantic_retriever(chunked_documents)
    bm25_retriever = build_bm25_retriever(chunked_documents)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    question = st.chat_input()

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        related_documents = hybrid_retriever.invoke(question)
        answer = answer_question(question, related_documents)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    # Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").write(chat["content"])
    else:
        st.chat_message("assistant").write(chat["content"])