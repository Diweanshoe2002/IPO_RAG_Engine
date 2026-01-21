# IPO RAG Research Assistant

A production-grade Retrieval-Augmented Generation (RAG) system for analyzing Indian IPO documents including DRHP (Draft Red Herring Prospectus) and RHP (Red Herring Prospectus) with zero-hallucination guarantees.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Interface                           │
│                    (User Query Input)                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │   IPO Registry       │
                    │   (Company → Doc)    │
                    └───────┬──────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
    ┌─────────▼────────┐      ┌─────────▼──────────┐
    │ FAISS Retrieval  │      │  Document Ingestion│
    │ (Isolated Index) │      │  (One-time)        │
    └─────────┬────────┘      └─────────┬──────────┘
              │                          │
              │                ┌─────────▼──────────┐
              │                │  Docling Parser    │
              │                │  - OCR             │
              │                │  - Layout Analysis │
              │                │  - Table Extract   │
              │                └─────────┬──────────┘
              │                          │
              │                ┌─────────▼──────────┐
              │                │  HybridChunker     │
              │                │  - Structure-aware │
              │                │  - Token-safe      │
              │                └─────────┬──────────┘
              │                          │
              │                ┌─────────▼──────────┐
              │                │  Embeddings        │
              │                │  → FAISS Index     │
              │                └────────────────────┘
              │
    ┌─────────▼────────┐
    │ Retrieved Chunks │
    │ (Top-K Context)  │
    └─────────┬────────┘
              │
    ┌─────────▼────────┐
    │ DSPy Reasoning   │
    │ (Chain-of-Thought)│
    └─────────┬────────┘
              │
    ┌─────────▼────────┐
    │ Structured Answer│
    │ (With Citations) │
    └──────────────────┘
```

## Key Features

### 1. **Production-Grade Document Processing**
- **Docling Integration**: Advanced OCR and layout-aware parsing
- **HybridChunker**: Structure-aware, token-safe chunking
- **Table Preservation**: Multi-page table cell matching
- **Section Context**: Maintains legal document hierarchy

### 2. **Zero-Hallucination Architecture**
- **Isolated Vector Stores**: Per-company FAISS indexes
- **Deterministic Retrieval**: Similarity-based Top-K retrieval
- **Source-Grounded Answers**: All responses cite IPO documents
- **No Cross-Contamination**: Company data never mixed

### 3. **Intelligent Reasoning**
- **DSPy Chain-of-Thought**: Programmatic reasoning modules
- **Multi-Section Analysis**: Connects information across document sections
- **Financial Table Understanding**: Extracts structured financial data
- **Legal Context Awareness**: Understands IPO document structure

### 4. **Scalable Design**
- **One-Time Ingestion**: Process documents once, query forever
- **Persistent Storage**: FAISS indexes cached on disk
- **Fast Retrieval**: Optimized vector similarity search
- **Modular Architecture**: Easy to extend and maintain

## Project Structure

```
ipo-rag-assistant/
│
├── app/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py              # Configuration management
│   │   └── llm_config.py            # LLM & embedding setup
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── docling_parser.py        # Docling PDF processor
│   │   ├── hybrid_chunker.py        # Structure-aware chunking
│   │   └── pipeline.py              # Complete ingestion pipeline
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── faiss_manager.py         # FAISS index management
│   │   └── embedding_generator.py   # Embedding creation
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py             # Document retrieval logic
│   │   └── ipo_registry.py          # Company-to-document mapping
│   │
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── dspy_chain.py            # DSPy reasoning modules
│   │   └── answer_formatter.py      # Response structuring
│   │
│   └── streamlit_app.py             # Main Streamlit interface
│
├── data/
│   ├── raw/                         # IPO PDF documents
│   │   ├── DRHP/
│   │   └── RHP/
│   ├── vectorstore/                 # Persistent FAISS indexes
│   └── registry/                    # Company metadata
│
├── scripts/
│   ├── ingest_documents.py          # Batch document ingestion
│   ├── rebuild_index.py             # Index rebuilding utility
│   └── validate_setup.py            # Environment validation
│
├── config/
│   ├── .env.example                 # Environment template
│   └── companies.yaml               # IPO company registry
│
├── docs/
│   ├── architecture.md              # Detailed architecture
│   ├── docling_guide.md             # Docling integration guide
│   ├── dspy_reasoning.md            # DSPy reasoning explanation
│   └── api_reference.md             # API documentation
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_reasoning.py
│
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (gitignored)
├── .gitignore
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- Cerebras API key (for LLM inference)
- Sufficient disk space for PDF storage and FAISS indexes

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ipo-rag-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp config/.env.example .env
   # Edit .env with your API keys
   ```

5. **Validate setup**
   ```bash
   python scripts/validate_setup.py
   ```

### Configuration

Edit `.env` file with your credentials:

```env
# Cerebras API
CEREBRAS_API_KEY=your-cerebras-api-key

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# LLM Configuration
LLM_MODEL=llama3.1-8b
LLM_TEMPERATURE=0.1
MAX_TOKENS=2048

# Retrieval Configuration
TOP_K_RETRIEVAL=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Storage Paths
RAW_DATA_DIR=./data/raw
VECTORSTORE_DIR=./data/vectorstore
REGISTRY_PATH=./config/companies.yaml
```

## Usage

### 1. Ingest IPO Documents

Place your IPO PDF documents in `data/raw/` directory:

```bash
data/raw/
├── DRHP/
│   ├── company_a_drhp.pdf
│   └── company_b_drhp.pdf
└── RHP/
    └── company_a_rhp.pdf
```

Run the ingestion pipeline:

```bash
python scripts/ingest_documents.py --input-dir data/raw/DRHP
```

The system will:
- Parse PDFs using Docling (OCR + layout analysis)
- Generate structure-aware chunks with HybridChunker
- Create embeddings
- Build and persist FAISS indexes

### 2. Launch Streamlit Interface

```bash
streamlit run app/streamlit_app.py
```

Access the interface at `http://localhost:8501`

### 3. Query IPO Documents

**Example Queries:**

- "What is the company's revenue growth over the last 3 years?"
- "Who are the key promoters and their shareholding patterns?"
- "What are the risk factors mentioned in the IPO?"
- "Explain the use of proceeds from this IPO"
- "Compare the P/E ratio with industry peers"

### 4. Programmatic Usage

```python
from app.rag.retriever import IPORetriever
from app.reasoning.dspy_chain import IPOAnalystChain

# Initialize retriever
retriever = IPORetriever(company="TechCorp")

# Retrieve relevant chunks
chunks = retriever.retrieve("What is the company's revenue?", top_k=5)

# Generate answer with DSPy reasoning
analyst = IPOAnalystChain()
answer = analyst.analyze(question="What is the company's revenue?", context=chunks)

print(answer.response)
print(answer.reasoning_steps)
```

## 🔧 Advanced Features

### Document Processing Pipeline

#### Docling: Why It Matters

IPO documents are **not normal text**:
- ✅ Tables span multiple pages
- ✅ Headers define legal context
- ✅ OCR quality varies across scans
- ✅ Complex layouts with mixed content

**Docling Responsibilities:**
- OCR extraction from scanned PDFs
- Layout-aware parsing (columns, headers, footers)
- Table cell matching across pages
- Section hierarchy preservation
- Figure and diagram detection

#### HybridChunker Advantages

Traditional chunking methods fail on IPO documents. HybridChunker provides:

- **Size-aware**: Respects token limits for embedding models
- **Structure-aware**: Preserves section and table boundaries
- **Context preservation**: Maintains parent-child relationships
- **Reduced over-chunking**: Minimizes information fragmentation
- **Improved retrieval**: Better semantic unit representation

**Chunking Strategy:**

```python
# Example: Financial table chunking
Table: "Revenue Breakdown"
├─ Chunk 1: Table header + Year 2021-2022 data
├─ Chunk 2: Table header + Year 2022-2023 data
└─ Chunk 3: Table header + Year 2023-2024 data

# Each chunk includes context for standalone understanding
```

### RAG Layer (FAISS)

**Design Principles:**

1. **Isolated Indexes**: Each IPO has its own FAISS index
   - No cross-company data leakage
   - Deterministic retrieval
   - Independent updates

2. **Persistent Storage**: Indexes cached on disk
   - Fast startup times
   - No re-ingestion needed
   - Version control friendly

3. **Similarity-Based Retrieval**: Top-K semantic search
   - Cosine similarity metrics
   - Configurable K values
   - Relevance scoring

4. **Metadata Enrichment**: Each chunk includes:
   - Source document
   - Page numbers
   - Section context
   - Table references

### DSPy: Chain-of-Thought Reasoning

DSPy operates **after retrieval**, never before.

**Why DSPy?**

- ✅ Explicit reasoning modules (not prompt engineering)
- ✅ Chain-of-Thought enforced programmatically
- ✅ Separation of retrieval and reasoning
- ✅ Reproducible outputs
- ✅ Easy to test and validate

**DSPy Flow:**

```
Input:
├─ Retrieved IPO chunks (context)
└─ User question

↓

DSPy Chain-of-Thought:
├─ Step 1: Identify relevant information
├─ Step 2: Extract key financial metrics
├─ Step 3: Analyze trends and patterns
├─ Step 4: Formulate answer
└─ Step 5: Generate citations

↓

Output:
├─ Reasoning steps (visible to user)
├─ Structured analyst answer
└─ Source citations
```

**Example DSPy Signature:**

```python
class IPOAnalysis(dspy.Signature):
    """Analyze IPO documents with chain-of-thought reasoning"""
    
    question: str = dspy.InputField(desc="User question")
    context: str = dspy.InputField(desc="Retrieved IPO chunks")
    
    reasoning: str = dspy.OutputField(desc="Step-by-step analysis")
    answer: str = dspy.OutputField(desc="Final structured answer")
    sources: List[str] = dspy.OutputField(desc="Source citations")
```

## Workflows

### Workflow 1: Financial Analysis

```
User: "What is the company's EBITDA margin trend?"

System:
1. Retrieves financial tables from last 3 years
2. DSPy identifies EBITDA and revenue figures
3. Calculates margins: 2021: 22%, 2022: 25%, 2023: 28%
4. Analyzes trend: "Improving margin, +6% over 3 years"
5. Cites: Pages 45-47, Financial Statements section
```

### Workflow 2: Risk Assessment

```
User: "What are the top 3 business risks?"

System:
1. Retrieves "Risk Factors" section
2. DSPy categorizes risks by severity
3. Identifies: Regulatory, Competition, Technology
4. Ranks by frequency and prominence
5. Cites: Pages 12-18, Risk Factors section
```

### Workflow 3: Competitive Positioning

```
User: "How does the company compare to competitors?"

System:
1. Retrieves market position and peer data
2. DSPy extracts competitor names and metrics
3. Compares market share, growth rates, margins
4. Analyzes: "Top 3 player, 15% market share, fastest growth"
5. Cites: Pages 8-10, Industry Overview section
```

# 🗺️ Roadmap

- [ ] Multi-language support (Hindi, regional)
- [ ] Comparative analysis across multiple IPOs
- [ ] Financial model generation
- [ ] Automated red flag detection
- [ ] Excel export of extracted data
- [ ] REST API for programmatic access
- [ ] Real-time IPO document monitoring

