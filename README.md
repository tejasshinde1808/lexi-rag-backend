# Lexi.sg RAG Backend Test
Setup Instructions

```bash
git clone https://github.com/yourname/lexi.sg-rag-backend-test
cd lexi.sg-rag-backend-test
pip install -r requirements.txt
python app/document_loader.py  # Index documents
uvicorn app.main:app --reload

