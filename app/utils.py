import fitz  # PyMuPDF
import os
import pickle
import faiss

def chunk_text(file_path, chunk_size=300):
    text = ""
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        text = " ".join([page.get_text() for page in doc])
    # Add DOCX support as needed
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def load_faiss_index():
    index = faiss.read_index("vector_store/index.faiss")
    with open("vector_store/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def generate_answer(query, context_chunks):
    # Simulate basic LLM behavior (mock if no GPU model used)
    return f"Based on the retrieved documents, here's the answer to: {query}"
