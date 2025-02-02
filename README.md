# 🏥 MediChat - AI-Powered Medical Assistant

MediChat is an **AI-powered medical chatbot** that answers medical questions using **fine-tuned LLaMA-2** and **Retrieval-augmented generation (RAG)**. It allows users find answers related to medical domain based on the upload medical documents (e.g., *Gale Encyclopedia of Medicine*, *Current Essential Medicine*) and search within them for relevant answers.

---
## **Project Overview**
This project leverages:
- **LangChain & FAISS for Document Search** (used in `app.py`)
- **Fine-Tuned LLaMA-2 Chat Model** (using `finetune.py`)
- **Streamlit UI** for an easy-to-use chatbot interface
- **Custom Medical Dataset** (extracted from *Gale Encyclopedia of Medicine Volume 1 & 2* and *Current Essential Medicine*)

## 📊 System Workflow

Below is a step-by-step flowchart of how the **MediChat AI Medical Chatbot** processes user queries using **LLaMA-2 and FAISS vector search**.

```mermaid
graph TD;
  A[User asks a medical question] -->|Check if documents are embedded| B{Are documents available?};
  B -- Yes --> C[Retrieve relevant document chunks];
  B -- No --> D[Prompt user to embed documents first];
  C --> E[Generate response using Llama2 & vector database];
  E --> F[Display medical answer & reference documents];
  F --> G[Allow user to ask another question];
  G --> A;

---

## **Project Structure**
📂 MediChat/ │── 📂 finetuned_llama/ # Fine-tuned LLaMA-2 model (output of finetune.py) │── 📂 Document/ # Folder to store uploaded PDFs │── 📝 app.py # Chatbot app (Streamlit UI + FAISS search) │── 📝 finetune.py # Fine-tuning script for LLaMA-2 │── 📝 custom_data.json # Training dataset (medical Q&A) │── 📝 custom_data_test.json # Validation dataset (testing AI's accuracy) │── 📝 requirements.txt # List of required Python libraries │── 📝 .env # Stores API keys (Hugging Face & Groq) │── 📝 README.md # Project documentation

---

## **How the Chatbot Works (`app.py`)**
`app.py` is responsible for:
1. **Loading the Fine-Tuned LLaMA-2 Model** → Uses the AI model trained in `finetune.py`.
2. **Processing PDFs into Searchable Chunks** → Uses FAISS to enable fast document retrieval.
3. **Retrieval-Augmented Generation (RAG)** → Combines AI-generated responses with document-based evidence.
4. **User-Friendly Chat Interface** → Allows users to ask questions and get answers.

## **How the Fine-Tuning Works (`finetune.py`)**
`finetune.py` is responsible for training LLaMA-2 using **LoRA (Low-Rank Adaptation)** to improve its understanding of medical queries.

### **What are we fine-tuning?**
- Instead of training **all** LLaMA-2 parameters (which would require massive GPUs), we **only fine-tune select attention layers**.
- This makes the model **faster, cheaper, and more memory-efficient**.
- The model **learns from `custom_data.json`**, which contains **medical Q&A pairs**.

### **Fine-Tuning Process**
1. **Load Pre-Trained LLaMA-2** from Hugging Face.
2. **Prepare Custom Medical Dataset (`custom_data.json`)**.
3. **Apply LoRA** to modify only essential model layers.
4. **Train for Multiple Epochs** (evaluating using `custom_data_test.json`).
5. **Save the Fine-Tuned Model** for use in `app.py`.

## **📚 Why We Use a Custom JSON Dataset?**
The default LLaMA-2 model **isn’t specialized in medicine**, so we need to teach it using **domain-specific data**.

### **✅ Custom Dataset Sources:**
- *Gale Encyclopedia of Medicine Volume 1 & 2*
- *Current Essential Medicine*