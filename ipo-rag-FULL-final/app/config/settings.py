import os
from llama_index.core import Settings
from llama_index.llms.cerebras import Cerebras
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def init_settings():
    Settings.llm = Cerebras(
        model="gpt-oss-120b",
        api_key=os.getenv("CEREBRAS_API_KEY")
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
