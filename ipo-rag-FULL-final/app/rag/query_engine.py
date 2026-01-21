def retrieve_context(index, query, top_k=5):
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [n.node.text for n in nodes]
