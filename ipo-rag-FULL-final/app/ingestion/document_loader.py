from app.ingestion.docling_chunker import docling_chunk_pdf
from llama_index.core import Settings

def load_documents(paths):
    tokenizer = Settings.embed_model._tokenizer
    docs = []
    for p in paths:
        docs.extend(docling_chunk_pdf(p, tokenizer))
    return docs
