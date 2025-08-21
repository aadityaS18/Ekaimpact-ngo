# rag.py
import os
import requests
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pathlib import Path

RAG_PROMPT = """You are an FAQ BOT with access to EkaImpact NGO information. 
Use the provided context to answer the user’s question.

If the answer is not in the context, say: "I don’t know based on EkaImpact’s data."

Context:
{context}

Question: {question}

Answer:"""

# --- Load FAISS index ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("data/faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- Custom Prompt ---
prompt_tmpl = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT
)

def answer_question(question: str, mistral_api_key: str, model: str = "mistral-small-latest") -> str:
    """
    Ask a question using Mistral API directly (no client lib).
    """
    # 1. Retrieve relevant docs
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Format prompt
    prompt = RAG_PROMPT.format(context=context, question=question)
    

    # 3. Call mistral API directly
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {mistral_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an assistant answering FAQs."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400
    }

    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")

    result = resp.json()
    return result["choices"][0]["message"]["content"]

# --- Script entry ---
if __name__ == "__main__":
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
    key= os.getenv("MISTRAL_API_KEY") 
    
    q = input("Question: ")
    print("Answer:", answer_question(q, mistral_api_key=key))
