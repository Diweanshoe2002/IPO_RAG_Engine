from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

def build_or_load_index(docs, persist_dir):
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage
    )

    index.storage_context.persist(persist_dir)
    return index
