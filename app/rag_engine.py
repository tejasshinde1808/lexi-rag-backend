from sentence_transformers import SentenceTransformer
import faiss
import pickle
from app.utils import load_faiss_index, retrieve_docs, generate_answer
import json

def answer_query(query):
    # Load components
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, metadata = load_faiss_index()

    # Embed query and retrieve top-k docs
    query_vector = model.encode([query])
    top_k = 3
    D, I = index.search(query_vector, top_k)
    top_chunks = [metadata[i] for i in I[0]]

    # Get raw texts
    context_texts = [chunk["text"] for chunk in top_chunks]
    sources = [{"text": c["text"], "source": c["source"]} for c in top_chunks]

    # Generate answer
    answer = generate_answer(query, context_texts)

    return {
        "answer": answer,
        "citations": sources
    }
