# 🤖 Local RAG Chatbot — Ollama + FAISS

A **fully local Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit, LangChain, FAISS, Hugging Face embeddings, and Ollama**.

The application allows you to upload a `.txt` document, convert it into searchable vector embeddings, and ask questions about the document using a locally running LLM. Your document and conversations remain on your machine.

## ✨ Features

* 📄 Upload `.txt` documents
* 🔍 Split documents into smaller chunks for retrieval
* 🧠 Generate embeddings using `all-MiniLM-L6-v2`
* 🗂️ Store embeddings locally using FAISS
* 🤖 Generate answers using Ollama and `llama3.2`
* 🔒 No external API required for LLM inference
* 💬 Chat-style Streamlit interface
* 💾 Conversation history maintained during the session
* 🖥️ CPU-based Hugging Face embeddings for better compatibility on macOS
* ⚡ Lightweight and easy to run locally

## 🏗️ Architecture

```text
                 ┌──────────────────┐
                 │   Text Document  │
                 │      (.txt)      │
                 └────────┬─────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    TextLoader         │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Text Splitter        │
              │ chunk_size = 300      │
              │ overlap = 50          │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Hugging Face          │
              │ Embeddings             │
              │ all-MiniLM-L6-v2      │
              └───────────┬───────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │      FAISS     │
                 │ Vector Store   │
                 └───────┬────────┘
                         │
                    User Question
                         │
                         ▼
                 ┌────────────────┐
                 │   Retriever    │
                 │    Top K = 2   │
                 └───────┬────────┘
                         │
                         ▼
                 ┌────────────────┐
                 │    Ollama      │
                 │    llama3.2    │
                 └───────┬────────┘
                         │
                         ▼
                    Final Answer
```

## 🛠️ Tech Stack

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Application development |
| Streamlit    | Web UI                  |
| LangChain    | RAG pipeline            |
| FAISS        | Vector database         |
| Hugging Face | Text embeddings         |
| Ollama       | Local LLM inference     |
| Llama 3.2    | Language model          |

## 📋 Requirements

* Python 3.10+
* Ollama
* Git
* At least 8 GB RAM recommended
* Internet connection for the initial installation/model downloads

After the models and dependencies have been downloaded, the chatbot can run locally without sending documents to an external LLM API.

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <PROJECT_FOLDER>
```

### 2. Install dependencies

```bash
pip install streamlit langchain langchain-community langchain-text-splitters langchain-huggingface langchain-ollama faiss-cpu sentence-transformers
```

## 🦙 Set Up Ollama

Install Ollama on your computer and download the required model.

Then run:

```bash
ollama pull llama3.2
```

Verify that the model is available:

```bash
ollama list
```

You should see:

```text
llama3.2
```

Make sure Ollama is running before starting the Streamlit application.

## ▶️ Run the Application

Start the application with:

```bash
streamlit run app.py
```

Replace `app.py` with the actual filename if your Python file has a different name.

Streamlit will provide a local address such as:

```text
http://localhost:8501
```

Open it in your browser.

## 📄 How to Use

### Step 1 — Upload a document

Use the **Knowledge Base** section in the sidebar and upload a `.txt` file.

### Step 2 — Process the document

Click:

```text
Process Document
```

The application will:

1. Load the document.
2. Split it into chunks.
3. Generate embeddings.
4. Create a FAISS vector store.
5. Store the vector store in the current Streamlit session.

### Step 3 — Ask questions

Enter your question in the chat box.

Here are some of the questions I asked to the chatbot from my document.

```text
Staring out the window, Clara watched the thick grey fog roll across the cobblestone path. The old brass lantern by the garden gate flickered, its tiny flame struggling against the damp chill of autumn. Nobody walked down this lane after midnight, yet she distinctly heard the slow, measured tap of a wooden cane approaching the door. She froze, her tea growing cold in her hands.
```

### Comprehension Questions
```text
What weather condition is described at the beginning of the story?
Response:- The weather condition described at the beginning of the story is fog. Specifically, it's "thick grey fog" that is rolling across the cobblestone path.
```
```text
Where is the flickering brass lantern located?
Response - The flickering brass lantern is located by the garden gate.
```
```text
What sound did Clara hear coming down the lane?
Response:- The sound that Clara heard coming down the lane was the slow, measured tap of a wooden cane.
```
```text
How does Clara react to the sound?
Response:- She freezes.
```
```text
What do you think will happen next at the door?
Response:- Based on the context, I think what will happen next is that someone (or something) will knock on the door, and Clara will likely be surprised or even frightened by the sudden arrival of a visitor at such an unusual hour. The slow, measured tap of a wooden cane suggests that the person approaching may be elderly or have some kind of disability, which could add to the sense of unease or curiosity.
```

## 🔄 RAG Pipeline

The application follows this workflow:

```text
Document
   ↓
Load
   ↓
Chunk
   ↓
Embedding
   ↓
FAISS Vector Store
   ↓
Similarity Search
   ↓
Top 2 Relevant Chunks
   ↓
Prompt + Context
   ↓
Llama 3.2
   ↓
Answer
```

The prompt instructs the model to answer based only on the retrieved context.

## 🔐 Privacy

This project is designed for **local/private document question answering**.

The LLM inference is performed locally through Ollama rather than through an external API.

The uploaded document is temporarily written to a local temporary file while it is being processed and is then deleted.

The FAISS vector store is maintained in Streamlit session state and is not uploaded to a cloud database.

### Important

Privacy depends on your local environment and Ollama/model configuration. Installing the application initially requires downloading dependencies and models from the internet, but document inference itself does not require an OpenAI or other cloud LLM API.

## ⚙️ Configuration

### Embedding Model

The project uses:

```python
all-MiniLM-L6-v2
```

Embeddings are explicitly configured to use the CPU:

```python
model_kwargs={'device': 'cpu'}
```

This helps avoid Metal GPU-related driver issues on macOS.

### Chunking

Current configuration:

```python
chunk_size=300
chunk_overlap=50
```

### Retrieval

The application retrieves the top 2 relevant chunks:

```python
search_kwargs={"k": 2}
```

### LLM

The application uses:

```python
ChatOllama(
    model="llama3.2",
    temperature=0
)
```

A temperature of `0` makes responses more deterministic.

## 📁 Suggested Project Structure

```text
local-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

You should **not** commit the virtual environment.

Example `.gitignore`:

```gitignore
venv/
__pycache__/
*.pyc
.env
.DS_Store
```

## 📦 requirements.txt

Create a `requirements.txt` file containing:

```text
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-ollama
langchain-core
faiss-cpu
sentence-transformers
```

Then install everything with:

```bash
pip install -r requirements.txt
```

## 🧪 Example

Suppose you upload:

```text
company_policy.txt
```

Containing information about employee policies.

You could ask:

```text
What is the annual leave policy?
```

The system retrieves the relevant sections from the document and provides an answer using the local Llama model.

## ⚠️ Current Limitations

* Only `.txt` files are supported.
* The vector store exists only for the current Streamlit session.
* Uploaded documents need to be processed again after restarting the application.
* Only the top 2 chunks are retrieved for each question.
* The application currently handles one active knowledge base at a time.
* There is no persistent FAISS index.
* There is no document citation/source display in the UI.
* Ollama must be installed and running locally.

## 🚀 Future Improvements

Possible upgrades include:

* 📚 Support PDF, DOCX and Markdown files
* 🗃️ Persistent FAISS indexes
* 📑 Display source documents and retrieved chunks
* 🔎 Adjustable similarity search
* 📂 Multiple document collections
* 🧹 Automatic document management
* 💾 Persistent chat history
* 🎛️ Model selection from the UI
* 🌐 Streaming LLM responses
* 📊 RAG evaluation metrics
* 🔐 Local authentication
* 🖥️ Package as a standalone desktop application
* 🧠 Add reranking for better retrieval
* 📝 Add conversation export
* 🔄 Add document re-indexing

## 🎯 Why This Project?

This project demonstrates the core components of a modern **Retrieval-Augmented Generation system** without relying on a paid cloud LLM API.

It is useful for learning and demonstrating:

* Large Language Models
* RAG architecture
* Vector databases
* Semantic search
* Embeddings
* LangChain
* Local AI
* Ollama
* Streamlit
* FAISS

## 👨‍💻 Author

**Aditya Bawankule**