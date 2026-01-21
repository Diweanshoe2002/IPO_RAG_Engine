import streamlit as st
import nest_asyncio
nest_asyncio.apply()

from app.config.settings import init_settings
from app.ingestion.metadata_registry import get_documents_for_company
from app.ingestion.document_loader import load_documents
from app.vectorstore.index_manager import build_or_load_index
from app.rag.query_engine import retrieve_context
from app.reasoning.dspy_reasoner import run_reasoning
from llama_index.core import Settings
import dspy

st.set_page_config(page_title="IPO RAG Assistant", layout="wide")
st.title("IPO RAG Research Assistant")

company = st.selectbox("Select Company", ["shadowfax", "tata_capital"])
query = st.text_area("Ask a question from IPO documents")

if st.button("Analyze"):
    init_settings()
    dspy.settings.configure(lm=Settings.llm)

    paths = get_documents_for_company(company)
    docs = load_documents(paths)

    index = build_or_load_index(docs, f"data/vectorstore/{company}")
    context = retrieve_context(index, query)

    answer = run_reasoning(context, query)
    st.markdown("### Answer")
    st.write(answer)
