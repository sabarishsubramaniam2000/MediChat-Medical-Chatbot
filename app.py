import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ==============================
# Streamlit Page Configuration
# ==============================
st.set_page_config(
    page_title="MediChat 🏥 - AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
)

# Load environment variables
load_dotenv()

# Load GROQ API key
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY not found! Please set it in your .env file.")

# Streamlit App Header
st.markdown("<h1 style='text-align: center; color: #00897B;'>MediChat 🏥 - AI Medical Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #00796B;'>Ask medical questions based on uploaded documents</h5>", unsafe_allow_html=True)

# ==============================
# Load Fine-Tuned LLaMA-2 Model
# ==============================
model_path = "./finetuned_llama"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    st.success("Fine-tuned medical model loaded successfully.")
except Exception as e:
    st.error(f"Error loading fine-tuned model: {e}")

# ==============================
# Initialize Llama3 model using ChatGroq (Backup Option)
# ==============================
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# ==============================
# Define Prompt Template
# ==============================
prompt = ChatPromptTemplate.from_template(
    """
    You are MediChat, an AI Medical Assistant. 
    Answer medical questions accurately based on the provided context only. 
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# ==============================
# Initialize Embedding Model
# ==============================
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ==============================
# Function to Process PDFs and Create Vector Embeddings
# ==============================
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("./Document")  # Load PDFs
        st.session_state.docs = st.session_state.loader.load()  # Load documents

        if not st.session_state.docs:
            st.error("No medical documents found in './Document'. Please upload PDFs.")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        if not st.session_state.final_documents:
            st.error("No valid document chunks created. Check your medical PDFs.")
            return

        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            embedding_model
        )
        st.success("Medical Knowledge Base is Ready!")

# ==============================
# User Input Section
# ==============================
user_query = st.text_input("Enter your medical question based on the uploaded documents:")

# Button to Process Medical Documents
if st.button("Process Medical Documents"):
    vector_embedding()

# ==============================
# Handle Query Processing
# ==============================
if user_query:
    if "vectors" not in st.session_state:
        st.error("Please process the medical documents first.")
    else:
        retriever = st.session_state.vectors.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()

        try:
            response = retrieval_chain.invoke({'input': user_query})
            elapsed_time = time.process_time() - start_time

            st.write(f"Response Time: {elapsed_time:.2f} seconds")
            st.subheader("Medical Answer:")
            st.write(response.get('answer', "No relevant medical information found."))
        except Exception as e:
            st.error(f"Error processing query: {e}")

        with st.expander("Relevant Medical References"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("-------------------")
