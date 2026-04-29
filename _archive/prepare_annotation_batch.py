"""
prepare_annotation_batch.py
----------------------------
Selects richest parsed documents and prepares them for annotation.
Outputs annotation/raw_qa/label_studio_import.json

Usage:
    python scripts/prepare_annotation_batch.py
"""
import os, json, random

MIN_WORDS = 80
MAX_WORDS = 200
PASSAGES_PER_DOC = 5

def load_top_docs(n=40):
    rows = []
    for source, folder in [("sebi","data/parsed/sebi"),("rbi","data/parsed/rbi")]:
        for fname in os.listdir(folder):
            if not fname.endswith(".txt"): continue
            path = os.path.join(folder, fname)
            with open(path, encoding="utf-8") as f: text = f.read()
            words = len(text.split())
            if words > 400:
                rows.append({"source":source,"file":fname,"path":path,"words":words,"text":text})
    rows.sort(key=lambda x: x["words"], reverse=True)
    return rows[:n]

def extract_passages(text, source_file):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    passages = []
    for i, para in enumerate(paragraphs):
        words = para.split()
        if MIN_WORDS <= len(words) <= MAX_WORDS:
            passages.append({"context": para, "document": source_file})
        elif len(words) > MAX_WORDS:
            chunk = " ".join(words[:MAX_WORDS])
            if len(chunk.split()) >= MIN_WORDS:
                passages.append({"context": chunk, "document": source_file})
    return passages[:PASSAGES_PER_DOC]

def main():
    os.makedirs("annotation/raw_qa", exist_ok=True)
    docs = load_top_docs(n=40)
    all_passages = []
    for doc in docs:
        all_passages.extend(extract_passages(doc["text"], doc["file"]))
    random.shuffle(all_passages)
    tasks = [{"id": i+1, "data": {"context": p["context"], "document": p["document"]}}
             for i, p in enumerate(all_passages)]
    path = "annotation/raw_qa/label_studio_import.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"Done. {len(tasks)} passages → {path}")

if __name__ == "__main__":
    main()
