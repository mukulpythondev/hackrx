# 📄 Legal-HR-Insurance RAG Chatbot – Bajaj Finserv Hackathon

## 💻 GitHub Repository
**Demo API:** `https://hackrx-submission.up.railway.app/hackrx/run`

---

## 🚀 Problem Statement
In **insurance, HR, legal, and compliance** domains, retrieving **exact clause-level answers** from lengthy documents is time-consuming.  
Our solution enables:
- Parsing **PDF/DOCX/email** documents.  
- Retrieving **precise clauses** relevant to a natural language query.  
- Providing **traceable metadata** (page, section).  
- Outputting answers in **structured JSON** format for seamless integration.

---

## 📌 Key Features
✅ Multi-format document ingestion (PDF, DOCX, email)  
✅ Clause-level retrieval with page & section reference  
✅ Semantic search powered by **OpenAI Embeddings + Pinecone**  
✅ Strict context-based answering to reduce hallucinations  
✅ Structured JSON responses  
✅ Parallel Q&A handling  

---

## 🛠️ Tech Stack
| Layer            | Technology |
|------------------|------------|
| Backend API      | FastAPI |
| LLM              | OpenAI GPT-4o-mini |
| Embeddings       | OpenAI `text-embedding-3-large` |
| Vector DB        | Pinecone |
| Graph DB         | Neo4j |
| PDF Parsing      | LangChain PyPDFLoader |
| Chunking         | RecursiveCharacterTextSplitter |
| Deployment       | Uvicorn / Docker |
| Utilities        | python-dotenv, aiohttp, hashlib |

---

## ⚙️ Architecture
<img width="924" height="738" alt="image" src="https://github.com/user-attachments/assets/f9e1261e-ae28-4b2a-9b9b-9777922b6f8f" />


### **Document Flow**
1. **Ingestion** – Document URL or file upload.  
2. **Parsing & Chunking** – PDF text extraction → chunks of 1000 characters (200 overlap).  
3. **Embedding Generation** – OpenAI embeddings for semantic representation.  
4. **Storage** – Pinecone for vector search + Neo4j for graph storage.

### **Query Flow**
1. User submits natural language questions.  
2. Retriever fetches top-k relevant chunks (filtered by `doc_id`).  
3. LLM generates answers strictly from retrieved context.  
4. Output returned as **JSON with metadata**.

---

Here’s the markdown version you can directly paste:

```
📂 Project Structure
├── app/                         # Core application logic
│   ├── prompt.py                 # LLM prompt templates
│   ├── rag_graph.py              # Neo4j-based graph retrieval
│   ├── rag_vector.py             # Pinecone vector retrieval
│   ├── utils.py                  # Helper utilities
│
├── graph_main.py                 # Graph-based RAG FastAPI endpoint
├── main.py                       # Main entry point for API
├── requirements.txt              # Python dependencies
```


## ⚙️ Installation & Setup

```bash
# 1️⃣ Clone Repository
git clone https://github.com/mukulpythondev/hackrx
cd hackrx

# 2️⃣ Install Dependencies
pip install -r requirements.txt

# 3️⃣ Create .env File
NEO4J_URL=bolt://<your-neo4j-url>
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
OPENAI_API_KEY=your_openai_key

# 4️⃣ Run App
uvicorn graph2_main:app --reload
````

---

📡 **Request Format**

```json
POST /hackrx/run
Authorization: Bearer <token>
Content-Type: application/json

{
    "documents": "https://your-storage.com/policy.pdf",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}
```

📡 **Response Format**

```json
{
    "answers": [
        "A grace period of thirty days is provided...",
        "There is a waiting period of thirty-six months...",
        "Yes, the policy covers maternity expenses..."
    ]
}
```

---

## 💡 Implementation Highlights

* **Document Hashing** – MD5 to uniquely tag docs (`doc_id`).
* **Multi-query generation** for better recall.
* **Chunk deduplication** before LLM processing.
* **Async PDF download** for performance.
* **Strict context mode** to avoid hallucination.

---

## 🏆 Unique Selling Points (USP)

* Works across **Insurance, Legal, HR, Compliance**.
* Answers are **evidence-backed** with metadata.
* Supports **multi-question batch processing**.
* Combines **vector search + graph relationships**.

---

## 📈 Business Impact

* **Insurance** – Faster claims eligibility check.
* **Legal** – Clause extraction in contracts.
* **HR** – Quick employee policy clarification.
* **Compliance** – Regulation lookup with audit trail.

---

## 🔮 Future Enhancements

* Cross-document clause linking.
* Web UI for file upload & search.
* Offline FAISS fallback.
* Role-based access control.

