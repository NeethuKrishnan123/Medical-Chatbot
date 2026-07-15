# Medical-Chatbot

An AI-powered Medical Chatbot that answers medical questions using Retrieval-Augmented Generation (RAG). The chatbot retrieves relevant medical information from a vector database and generates context-aware responses using a Large Language Model (LLM).

---

##  Features

*  AI-powered medical question answering
*  PDF document ingestion
*  Automatic text chunking
*  Semantic search using Pinecone Vector Database
*  Retrieval-Augmented Generation (RAG)
*  Interactive chatbot interface
*  Fast responses with Groq LLM
*  Flask-based web application
*  Ready for cloud deployment

---

##  Tech Stack

### Programming Language

* Python 3.10

### Frameworks & Libraries

* Flask
* LangChain
* LangChain Community
* LangChain Groq
* LangChain Pinecone

### AI & LLM

* Groq API
* HuggingFace Embeddings

### Vector Database

* Pinecone

### Document Processing

* PyPDF
* RecursiveCharacterTextSplitter

### Frontend

* HTML
* CSS

### Deployment

* AWS
* GitHub

---

#  Project Structure

```
Medical-Chatbot/
│
├── data/
│   └── Medical_Book.pdf
|
├── research/
|   └── trials.ipynb
│
├── src/
│   ├── helper.py
│   ├── prompt.py
│   └── __init__.py
│
├── static/
│   └── style.css
│
├── templates/
│   └── chat.html
|
├── .env
├── .gitignore
├── app.py
├── store_index.py
├── requirements.txt
├── setup.py
├── README.md
└── template.sh
```

---

#  Installation

Clone the repository

```bash
git clone https://github.com/NeethuKrishnan123/Medical-Chatbot
```

Move into the project

```bash
cd Medical-Chatbot
```

Create a Conda environment

```bash
conda create -n medibot python=3.10 -y
```

Activate the environment

```bash
conda activate medibot
```

Install dependencies

```bash
pip install -r requirements.txt
```

Install the project

```bash
pip install -e .
```

---

#  Environment Variables

Create a `.env` file in the project root.

```
PINECONE_API_KEY
GROQ_API_KEY
```

---

#  Store PDF Embeddings

Run:

```bash
python store_index.py
```

This will:

* Load medical PDF documents
* Split documents into chunks
* Generate embeddings
* Store embeddings in Pinecone

---

#  Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:8080/
```

---

#  Workflow

```
Medical PDF
      │
      ▼
Document Loader
      │
      ▼
Text Splitter
      │
      ▼
HuggingFace Embeddings
      │
      ▼
Pinecone Vector Database
      │
      ▼
Retriever
      │
      ▼
Groq LLM
      │
      ▼
Medical AI Chatbot
```


---





