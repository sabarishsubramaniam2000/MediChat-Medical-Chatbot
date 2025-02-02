# 🏥 MediChat - AI-Powered Medical Assistant

MediChat is an **AI-powered medical chatbot** that answers medical questions using **fine-tuned LLaMA-2** and **retrieval-augmented generation (RAG)**. It allows users to **upload medical documents** (e.g., *Gale Encyclopedia of Medicine*, *Current Essential Medicine*) and **search within them** for relevant answers.

---
## **🚀 Project Overview**
This project leverages:
- **Fine-Tuned LLaMA-2 Chat Model** (using `finetune.py`)
- **LangChain & FAISS for Document Search** (used in `app.py`)
- **Streamlit UI** for an easy-to-use chatbot interface
- **Custom Medical Dataset** (extracted from *Gale Encyclopedia of Medicine Volume 1 & 2* and *Current Essential Medicine*)

---

## **📂 Project Structure**
📂 MediChat/
│── 📂 Document/                  # Folder to store uploaded PDFs
│── 📝 app.py                     # Chatbot app (Streamlit UI + FAISS search)
│── 📝 finetune.py                 # Fine-tuning script for LLaMA-2
│── 📝 custom_data.json            # Training dataset (medical Q&A)
│── 📝 custom_data_test.json       # Validation dataset (testing AI's accuracy)
│── 📝 requirements.txt            # List of required Python libraries
│── 📝 .env                        # Stores API keys (Hugging Face & Groq)
│── 📝 README.md    