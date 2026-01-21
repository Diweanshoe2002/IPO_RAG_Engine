# IPO RAG Research Assistant (Production Architecture)

This project implements a **production-grade Retrieval-Augmented Generation (RAG) system**
for analyzing Indian IPO documents such as **DRHP** and **RHP**.

The system is designed to handle:
- Long legal documents
- OCR-heavy PDFs
- Financial tables
- Multi-section reasoning
- Zero hallucination guarantees

The architecture explicitly separates **ingestion**, **retrieval**, and **reasoning**.

---

## 1. High-Level Architecture

### Runtime Flow
1. User selects a company and asks a question via Streamlit
2. Company → document mapping is resolved using the IPO Registry
3. Relevant chunks are retrieved from a **persistent FAISS vector store**
4. Retrieved context is passed to **DSPy Chain-of-Thought reasoning**
5. Final answer is generated strictly from IPO documents

### One-Time Ingestion Flow
1. PDF is parsed using **Docling**
2. OCR, layout detection, and table extraction are applied
3. HybridChunker generates structure-aware chunks
4. Chunks are embedded and stored in FAISS

---

## 2. Docling: Why It Matters

IPO documents are **not normal text**:
- Tables span multiple pages
- Headers define legal context
- OCR quality varies across scans

### Docling Responsibilities
- OCR extraction (scanned PDFs)
- Layout-aware parsing
- Table cell matching
- Section preservation

### HybridChunker Advantages
- Size-aware (token safe)
- Structure-aware (sections & tables)
- Reduced over-chunking
- Improved retrieval precision

Docling + HybridChunker ensures that embeddings represent **semantic units**, not arbitrary splits.

---

## 3. RAG Layer (FAISS)

- Each IPO has an **isolated FAISS index**
- Embeddings are generated once and persisted
- Retrieval is similarity-based (Top-K)
- No cross-company leakage

This guarantees deterministic, repeatable answers.

---

## 4. DSPy: Chain-of-Thought Reasoning

DSPy is used **after retrieval**, never before.

### Why DSPy?
- Explicit reasoning modules (not prompt hacks)
- Chain-of-Thought enforced programmatically
- Separation of retrieval and reasoning

### DSPy Flow
- Input: Retrieved IPO chunks + user question
- Output:
  - Internal reasoning steps
  - Final structured analyst answer

DSPy never accesses PDFs or embeddings directly.
It only reasons over retrieved context.

---

## 5. Project Structure

```
ipo-rag/
├── app/
│   ├── config/        # LLM & embedding setup
│   ├── ingestion/     # Docling + HybridChunker
│   ├── vectorstore/   # FAISS persistence
│   ├── rag/           # Retrieval logic
│   ├── reasoning/     # DSPy reasoning
│   └── streamlit_app.py
├── data/
│   ├── raw/           # IPO PDFs
│   └── vectorstore/   # FAISS indexes
├── requirements.txt
├── .env.example
└── README.md
```

---

## 6. Run Instructions

```bash
pip install -r requirements.txt
export CEREBRAS_API_KEY=your_key
streamlit run app/streamlit_app.py
```

---

## 7. Design Guarantees

- No hallucinations
- No external knowledge
- No prompt-only reasoning
- Production-safe modular architecture

---

## Disclaimer
This system is for research purposes only and is not investment advice.
