import streamlit as st
import os
import numpy as np
import psycopg2
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from secret_api_keys import huggingface_api_key
from transformers import pipeline
import torch

os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key

def connect_db():
    try:
        connection = psycopg2.connect(
            dbname="rag_db",
            user="postgres",
            password="1234",
            host="localhost",
            port="5432"
        )
        return connection
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def store_document(doc_name, content, embedding):
    connection = connect_db()
    if connection:
        try:
            cursor = connection.cursor()
            embedding_array = np.array(embedding, dtype=np.float32).tolist()
            cursor.execute(
                "INSERT INTO documents (doc_name, content, embedding) VALUES (%s, %s, %s)",
                (doc_name, content, embedding_array)
            )
            connection.commit()
            cursor.close()
        except Exception as e:
            st.error(f"Error storing document: {e}")
        finally:
            connection.close()

def retrieve_documents(query_embedding, limit=3):
    connection = connect_db()
    if connection:
        try:
            cursor = connection.cursor()
            query_embedding_str = f"[{','.join(map(str, query_embedding))}]"
            cursor.execute(
                """
                SELECT doc_name, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY similarity DESC
                LIMIT %s;
                """,
                (query_embedding_str, limit)
            )
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
        finally:
            connection.close()
    return []

def process_input(input_type, input_data):
    text = ""
    if input_type == "Link":
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(input_data)
        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
    elif input_type == "PDF":
        pdf_reader = PdfReader(BytesIO(input_data.read()))
        text = "".join(page.extract_text() for page in pdf_reader.pages)
    elif input_type == "Text":
        text = input_data
    elif input_type == "DOCX":
        doc = Document(BytesIO(input_data.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
    elif input_type == "TXT":
        text = input_data.read().decode('utf-8')
    else:
        raise ValueError("Unsupported input type")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = hf_embeddings.embed_documents(texts)

    for i, chunk in enumerate(texts):
        store_document(f"doc_chunk_{i}", chunk, embeddings[i])

    st.success("Document processed and stored successfully!")

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def answer_question(query):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    query_embedding = hf_embeddings.embed_query(query)

    documents = retrieve_documents(query_embedding)
    if not documents:
        return "No relevant documents found."

    context = "\n\n".join([doc[1] for doc in documents])[:5000]

    llm = HuggingFaceEndpoint(
        repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
        token=huggingface_api_key,
        temperature=0.3,
        max_new_tokens=150
    )

    prompt = f"""
    You are a helpful assistant. Answer the following question concisely using the provided context.
    
    Context: {context}
    Question: {query}
    
    Provide a brief and precise answer:
    """

    try:
        answer = llm.predict(prompt)
        return summarize_text(answer)
    except Exception as e:
        return f"Error generating answer: {e}"

def main():
    st.title("RAG Q&A App with PostgreSQL")
    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    if input_type == "Link":
        num_links = st.number_input("Enter the number of links", min_value=1, max_value=20, step=1)
        input_data = [st.text_input(f"URL {i+1}") for i in range(num_links)]
    elif input_type == "Text":
        input_data = st.text_area("Enter your text")
    else:
        input_data = st.file_uploader(f"Upload a {input_type} file", type=input_type.lower())

    if st.button("Proceed"):
        if input_data:
            process_input(input_type, input_data)
        else:
            st.error("Please provide input data.")

    query = st.text_input("Ask your question")
    if st.button("Submit"):
        if query:
            answer = answer_question(query)
            st.write(answer)
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()