from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.rag_engine import answer_query

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_endpoint(payload: QueryRequest):
    return answer_query(payload.query)
