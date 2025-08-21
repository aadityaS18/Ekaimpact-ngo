

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR  = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE_DIR, "data")
SITE_TXT  = os.path.join(DATA_DIR, "ekai_cleaned.txt")  # scraped/cleaned file
FAQ_TXT   = os.path.join(DATA_DIR, "faq.txt")           # optional FAQ file
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
os.makedirs(DATA_DIR, exist_ok=True)

def parse_faq(raw: str):
    # blank-line separated blocks; Q: / A: lines
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    qa = []
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        q = next((l[2:].strip() for l in lines if l.lower().startswith("q:")), None)
        a = next((l[2:].strip() for l in lines if l.lower().startswith("a:")), None)
        if q and a:
            qa.append((q, a))
    return qa

def load_docs():
    docs = []

    # 1) Site content → chunk
    if os.path.exists(SITE_TXT):
        with open(SITE_TXT, "r", encoding="utf-8") as f:
            site_text = f.read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        for ch in splitter.split_text(site_text):
            docs.append(Document(page_content=ch, metadata={"source": "site", "priority": 1}))
    else:
        print(f"  missing: {SITE_TXT}")

    # 2) FAQ → index Q and A
    if os.path.exists(FAQ_TXT):
        with open(FAQ_TXT, "r", encoding="utf-8") as f:
            pairs = parse_faq(f.read())

        for q, a in pairs:
            docs.append(Document(
                page_content=f"[FAQ Answer]\n{a}",
                metadata={"source": "faq", "priority": 10, "question": q}
            ))
            docs.append(Document(
                page_content=f"[FAQ Question]\n{q}",
                metadata={"source": "faq_q", "priority": 8, "answer": a}
            ))
    else:
        print(f"ℹ  missing: {FAQ_TXT} (optional)")

    if not docs:
        raise RuntimeError("No documents to index. Provide data/ekai_cleaned.txt and/or data/faq.txt")
    return docs

def main():
    docs = load_docs()

    # 🔹 Use Hugging Face sentence-transformer for embeddings
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vs = FAISS.from_documents(docs, embedding)
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"✅ Indexed {len(docs)} chunks → {INDEX_DIR}")

if __name__ == "__main__":
    main()


