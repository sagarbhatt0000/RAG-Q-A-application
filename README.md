
**RAG-Based Q&A Application with Streamlit and PostgreSQL**

This project is a Retrieval-Augmented Generation (RAG) based Question-Answering (Q&A) application. It uses Streamlit for the frontend, PostgreSQL with pgvector for storing document embeddings, and Hugging Face models for generating accurate and context-aware answers.

**🚀 Features**

Supports input from PDF, DOCX, TXT, Web Links, and plain text.
Utilizes LangChain for document chunking and management.
Performs efficient vector similarity search using pgvector in PostgreSQL.
Generates precise answers using Meta-Llama-3-8B-Instruct via Hugging Face.
Provides summarized responses using the BART model for clarity.
Simple and interactive UI with Streamlit.

**🛠 Tech Stack**

Frontend: Streamlit
Backend: Python, LangChain
Database: PostgreSQL with pgvector extension
LLM: Meta-Llama-3-8B-Instruct (Hugging Face)
Embeddings: Sentence Transformers (all-mpnet-base-v2)
Summarization: Facebook BART-Large-CNN

**📦 How It Works**

Upload Documents or Provide Links: Process documents using PDF, DOCX, or text files.
Generate Embeddings: Convert document chunks to vector embeddings using Hugging Face.
Store in Database: Store embeddings and text in PostgreSQL for retrieval.
Ask Questions: Input a query using the Streamlit interface.
Retrieve Relevant Context: Perform similarity search using vector embeddings.
Generate Answer: Use LLM to generate a concise and contextually accurate answer.
Summarize (if needed): Provide a short summary using BART if the response is lengthy.

**🧑‍💻 Future Enhancements**

Implement authentication and user management.
Add multilingual support for input and output.
Expand to support real-time data sources.
