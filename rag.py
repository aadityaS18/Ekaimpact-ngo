# rag.py
import os
from typing import List, Tuple, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline


RAG_PROMPT = """You are a helpful assistant with access to EkaImpact NGO information. 
Use the provided context to answer the user’s question.

If the answer is not in the context, say: "I don’t know based on EkaImpact’s data."

Context:
{context}

Question: {question}

Answer:"""

# --- Embeddings (must match build_index.py) ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Load FAISS index ---
vectorstore = FAISS.load_local("data/faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- Hugging Face model (BART or FLAN) ---
generator = pipeline(
    "text2text-generation",
    model="facebook/bart-large-cnn",   
    max_new_tokens=400
)
llm = HuggingFacePipeline(pipeline=generator)

# --- Custom Prompt ---
prompt_tmpl = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT
)

# --- Build QA chain ---
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_tmpl}
)

# --- Main function ---
def answer_question(question: str) -> str:
    """
    Ask a question to the RAG pipeline.
    - `question`: user query
    """
    result = qa.invoke({"query": question})
    return result["result"]

# --- Script entry ---
if __name__ == "__main__":
    q = input("Question: ")
    print("Answer:", answer_question(q))







