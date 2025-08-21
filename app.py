# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from rag import answer_question  

app = FastAPI()

class Question(BaseModel):
    query: str
    history: List[Tuple[str, str]] = []  

@app.post("/ask")
def ask_question(q: Question):
    answer = answer_question(q.query, q.history)
    return {"answer": answer}

