import os
from sentence_transformers import SentenceTransformer
from app.utils import chunk_text
import faiss
import pickle

def build_index(doc_dir='data/legal_docs'):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts, metadata = [], []

    for file in os.listdir(doc_dir):
        path = os.path.join(doc_dir, file)
        chunks = chunk_text(path)
        texts.extend(chunks)
        metadata.extend([{"text": c, "source": file} for c in chunks])

    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save
    faiss.write_index(index, "vector_store/index.faiss")
    with open("vector_store/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
