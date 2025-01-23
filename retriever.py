import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from embeddings import load_embeddings


def retrieve_documents(query, top_k=3):
    """
    Retrieves top-k documents relevant to the query.

    Args:
        query (str): User query.
        top_k (int): Number of top documents to retrieve.

    Returns:
        list: Top-k documents relevant to the query.
    """
    embeddings_data = load_embeddings()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])

    index = faiss.IndexFlatL2(query_embedding.shape[1])
    index.add(np.array(embeddings_data["embeddings"]))

    distances, indices = index.search(query_embedding, top_k)
    return [embeddings_data["documents"][i] for i in indices[0]]

    