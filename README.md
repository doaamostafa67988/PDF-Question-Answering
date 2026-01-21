
---

# Chat with PDF using Hybrid RAG

A hybrid Retrieval-Augmented Generation (RAG) system built with **Nomic**, **Llama 3.2**, LangChain, and Streamlit that enables you to chat with PDFs and answer complex questions about your local documents. This project improves RAG accuracy by leveraging a hybrid approach combining semantic embeddings and BM25 retrieval.

## Features

* Upload PDF files and ask questions interactively.
* Hybrid retrieval combining semantic search with BM25 to improve answer relevance.
* Uses **Llama 3.2 (3B)** model for language understanding and generation.
* Integrates **Nomic** for efficient embeddings and vector search.
* User-friendly chat interface built with Streamlit.
* Works seamlessly with local PDF documents.

## Prerequisites

* Install **Ollama** on your local machine. Download it from the [official website](https://ollama.com/).
* Pull the Llama 3.2 model via Ollama:

```bash
ollama pull llama3.2:3b
```

* Install required Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app locally:

```bash
streamlit run hybrid_pdf_rag.py
```


## Project Structure

* `hybrid_pdf_rag.py` — Main Streamlit app implementing the hybrid RAG pipeline.
* `requirements.txt` — Python dependencies.


## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/doaamostafa67988/Chat_with_pdf/blob/main/LICENSE) file for details.

---
