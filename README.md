# 📊 Business Intelligence + Financial RAG Assistant

A Streamlit-based **Business Intelligence, Data Analyst, and Financial Report RAG Chatbot** for uploading business documents and datasets and asking natural-language questions about revenue, expenses, profit, products, customers, inventory, policies, and financial performance.

## ✨ Features

- PDF, CSV, Excel (`.xlsx`, `.xls`), Word (`.docx`, `.doc`), TXT, Markdown, and common image files
- **ChromaDB** persistent vector database for semantic retrieval
- **Sentence Transformers** embeddings using `all-MiniLM-L6-v2`
- **Groq** LLM integration
- **Pandas** calculations for structured CSV/Excel analysis
- Configurable chunk size, overlap, Top-K retrieval, and temperature
- Source/file/page references for retrieved content
- Financial, sales, product, customer, inventory, and policy questions
- Cross-document analysis

## 🏗️ Recommended GitHub Structure

```text
Business-Intelligence-RAG-Chatbot/
│
├── Business_Analyst_Chroma.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

### File purposes

| File | Purpose |
|---|---|
| `Business_Analyst_Chroma.py` | Main Streamlit RAG chatbot |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `.gitignore` | Prevents private/unnecessary files from GitHub |
| `.env.example` | Example environment-variable configuration |

## 🔐 API Key Security

**Never upload your real Groq API key to GitHub.**

For local development, create a `.env` file:

```text
GROQ_API_KEY=your_actual_groq_api_key
```

Upload only `.env.example`:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Make sure `.env` is included in `.gitignore`.

## 💻 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Important packages include:

```text
streamlit
pandas
numpy
chromadb
sentence-transformers
groq
python-dotenv
pypdf
openpyxl
xlrd
python-docx
Pillow
pytesseract
```

### 4. Configure the Groq API key

Create `.env`:

```text
GROQ_API_KEY=your_actual_groq_api_key
```

### 5. Run the application

```bash
streamlit run Business_Analyst_Chroma.py
```

If the Streamlit command is not recognized:

```bash
python -m streamlit run Business_Analyst_Chroma.py
```

## 🧠 RAG Architecture

```text
Upload Files
     ↓
Text / Table / OCR Extraction
     ↓
Chunking
     ↓
Sentence Transformer Embeddings
     ↓
Persistent ChromaDB
     ↓
User Question
     ↓
Semantic Top-K Retrieval
     ↓
Pandas Calculations + Retrieved Context
     ↓
Groq LLM
     ↓
Grounded Answer + Sources
```

## 🗄️ ChromaDB

The application uses a persistent local ChromaDB directory:

```text
chroma_db/
```

This stores the vector database generated from processed documents.

**Do not upload `chroma_db/` to GitHub.**

Add this to `.gitignore`:

```text
chroma_db/
```

## 📂 Supported Uploads

The application supports:

- PDF
- CSV
- Excel
- Word
- TXT
- Markdown
- PNG
- JPG / JPEG
- WEBP

CSV and Excel files can also be analyzed directly with Pandas for numerical business questions.

## 📊 Example Questions

### Financial

```text
What was total revenue in 2025?
What were the major expenses?
Summarize the financial results.
Compare 2024 and 2025.
Why did profit decrease?
```

### Sales / Products

```text
Which product generated the most revenue?
Which product categories are performing poorly?
Which products have low inventory?
Which customers purchased the most?
```

### Business / Policy

```text
What does the company policy say about discounts?
Compare sales performance with the pricing policy.
Which products had strong sales but low inventory?
```

### Cross-document

```text
Compare the annual report with the sales CSV.
Does the sales data support the financial report?
Which category contributed most to revenue?
What business factors appear to explain the change in profit?
```

## ⚙️ RAG Configuration

The sidebar provides controls for:

- **Chunk Size** — controls document chunk length
- **Chunk Overlap** — controls overlap between chunks
- **Top-K Retrieved Chunks** — controls how many ChromaDB results are retrieved
- **LLM Temperature** — controls response randomness

For financial/business analysis, a lower temperature is generally preferable.

## ☁️ Deploying on Streamlit

1. Push the project to GitHub.
2. Create a Streamlit app connected to the GitHub repository.
3. Select `Business_Analyst_Chroma.py` as the main application file.
4. Make sure `requirements.txt` is in the repository.
5. Add the Groq key in Streamlit Secrets.

Example:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

Do **not** upload `.env` or your actual API key to GitHub.

## 🚫 Do Not Upload

```text
.env
chroma_db/
```

Also avoid committing confidential:

- Customer information
- Financial records
- Company reports
- Passwords
- API keys
- Private business documents

## 🛠️ Troubleshooting

### ChromaDB

```bash
pip install chromadb
```

### Sentence Transformers

```bash
pip install sentence-transformers
```

### Groq

```bash
pip install groq
```

### Word files

```bash
pip install python-docx
```

### Excel files

```bash
pip install openpyxl xlrd
```

### PDF files

```bash
pip install pypdf
```

## 🎯 Project Purpose

This project demonstrates a practical **Retrieval-Augmented Generation (RAG)** architecture for business intelligence and financial reporting.

It combines:

**Document Retrieval + Vector Database + Structured Data Analysis + LLM Reasoning**

to provide a business-friendly conversational interface for exploring uploaded company information.

## 🚀 Future Improvements

Possible enhancements include:

- Hybrid keyword + vector retrieval
- Reranking retrieved documents
- User-specific Chroma collections
- Persistent document management
- Table-aware retrieval
- Automatic chart generation
- Advanced metadata filtering
- Conversation memory
- Authentication
- Cloud vector database deployment

## 👨‍💻 Author

**Business Intelligence + Financial RAG Assistant**

Built with Python, Streamlit, ChromaDB, Sentence Transformers, Pandas, and Groq.
