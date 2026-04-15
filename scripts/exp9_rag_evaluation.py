"""
Experiment 9: RAG vs Oracle Context Evaluation
Builds FAISS index over source documents.
Tests 3 models under RAG condition vs. provided oracle context.
Key finding: If TMP accuracy drops only slightly under RAG,
failure is reasoning-driven not retrieval-driven.

Field fix: dataset uses 'answer' (not 'reference_answer').
"""
import sys, os, json
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_dataset, load_all_results, get_task_accuracies, call_gemini, call_groq,
    OUTPUT_DIR, FIGURES_DIR, REPO_ROOT, TASK_MAP, _task_col
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss

OUTPUT = OUTPUT_DIR / "rag_evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUTPUT / "faiss_index"
CHUNKS_PATH = OUTPUT / "doc_chunks.json"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TOP_K = 3

RAG_EVAL_PROMPT = """You are an expert in Indian financial regulatory text.

Use ONLY the following retrieved regulatory text to answer the question.
Do not use any external knowledge.

Retrieved Context:
{retrieved_context}

Question: {question}

Answer concisely in 1-3 sentences. Give only the answer, no preamble."""


def build_document_chunks():
    """Chunk all source documents into retrieval units."""
    parsed_dir = REPO_ROOT / "data/parsed"
    chunks = []
    chunk_id = 0

    text_files = list(parsed_dir.glob("**/*.txt")) + list(parsed_dir.glob("**/*.md"))

    if not text_files:
        print(f"No text files found in {parsed_dir}")
        print("Falling back to dataset contexts as document chunks...")
        dataset = load_dataset()
        for i, row in dataset.iterrows():
            context = str(row.get('context', '') or '')
            source = str(row.get('document', f'item_{i}') or f'item_{i}')
            chunks.append({'chunk_id': chunk_id, 'source': source, 'text': context, 'item_id': i})
            chunk_id += 1
        return chunks

    print(f"Found {len(text_files)} document files. Chunking...")
    for fpath in text_files:
        try:
            text = fpath.read_text(encoding='utf-8', errors='ignore')
            words = text.split()
            i = 0
            while i < len(words):
                chunk_words = words[i:i + CHUNK_SIZE]
                chunks.append({
                    'chunk_id': chunk_id,
                    'source': str(fpath.name),
                    'text': ' '.join(chunk_words),
                    'char_start': i
                })
                chunk_id += 1
                i += (CHUNK_SIZE - CHUNK_OVERLAP)
        except Exception as e:
            print(f"  Error reading {fpath}: {e}")

    print(f"Created {len(chunks)} chunks from {len(text_files)} documents")
    return chunks


def build_faiss_index(chunks):
    """Build FAISS index from document chunks."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c['text'] for c in chunks]

    print(f"Encoding {len(texts)} chunks...")
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = embedder.encode(batch, show_progress_bar=False,
                               convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype('float32')
    dim = embeddings.shape[1]

    print(f"Building FAISS index (dim={dim}, n={len(embeddings)})...")
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH / "index.faiss"))
    print(f"Index saved with {index.ntotal} vectors")
    return index, embedder


def retrieve(query, index, embedder, chunks, top_k=TOP_K):
    """Retrieve top-k chunks for a query."""
    query_emb = embedder.encode([query], normalize_embeddings=True).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    retrieved = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(chunks):
            retrieved.append({'text': chunks[idx]['text'], 'source': chunks[idx]['source'],
                               'score': float(dist)})
    return retrieved


def run_rag_evaluation(dataset, index, embedder, chunks, models_to_eval):
    """Run evaluation under RAG condition for specified models."""
    results = []
    sample_items = []
    for task in dataset['task_type'].unique():
        task_items = dataset[dataset['task_type'] == task]
        n = min(30, len(task_items))
        sample_items.append(task_items.sample(n, random_state=42))
    sample_df = pd.concat(sample_items).reset_index(drop=True)

    print(f"Running RAG evaluation on {len(sample_df)} items...")

    for model_name in models_to_eval:
        print(f"\n  Model: {model_name}")
        for idx, item in sample_df.iterrows():
            question = str(item.get('question', ''))
            # Use 'answer' field (verified dataset field name)
            reference = str(item.get('answer', ''))
            task_type = str(item.get('task_type', ''))

            retrieved = retrieve(
                question + ' ' + str(item.get('context', ''))[:200],
                index, embedder, chunks
            )
            retrieved_context = '\n\n---\n\n'.join([r['text'] for r in retrieved])

            prompt = RAG_EVAL_PROMPT.format(
                retrieved_context=retrieved_context[:3000],
                question=question
            )

            if 'Gemini' in model_name:
                prediction = call_gemini(prompt)
            elif model_name in ('LLaMA-3.3-70B', 'Qwen3-32B', 'DeepSeek R1 70B'):
                groq_model = {
                    'LLaMA-3.3-70B': 'llama-3.3-70b-versatile',
                    'Qwen3-32B': 'qwen-qwq-32b',
                    'DeepSeek R1 70B': 'deepseek-r1-distill-llama-70b'
                }[model_name]
                prediction = call_groq(prompt, model=groq_model)
            else:
                continue

            from rapidfuzz import fuzz
            ref_l = reference.lower().strip()
            pred_l = prediction.lower().strip()
            exact = int(ref_l in pred_l or pred_l == ref_l)
            fuzzy_score = fuzz.token_set_ratio(ref_l, pred_l) / 100.0
            is_correct = exact or fuzzy_score >= 0.72

            results.append({
                'item_idx': idx,
                'model': model_name,
                'task_type': task_type,
                'condition': 'RAG',
                'reference': reference,
                'prediction': prediction[:200],
                'is_correct': int(is_correct),
                'retrieved_sources': [r['source'] for r in retrieved]
            })

        pd.DataFrame(results).to_csv(OUTPUT / "rag_results_partial.csv", index=False)

    return pd.DataFrame(results)


def main():
    print("=== EXPERIMENT 9: RAG EVALUATION ===\n")
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    index_file = INDEX_PATH / "index.faiss"

    if index_file.exists() and CHUNKS_PATH.exists():
        print("Loading existing FAISS index...")
        with open(CHUNKS_PATH) as f:
            chunks = json.load(f)
        index = faiss.read_index(str(index_file))
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Loaded index with {index.ntotal} vectors and {len(chunks)} chunks")
    else:
        print("Building document chunks...")
        chunks = build_document_chunks()
        with open(CHUNKS_PATH, 'w') as f:
            json.dump(chunks, f)
        index, embedder = build_faiss_index(chunks)

    print("\nLoading oracle (existing) results...")
    task_accs_oracle = get_task_accuracies()

    rag_models = ["Gemini 2.5 Flash", "LLaMA-3.3-70B", "DeepSeek R1 70B"]

    rag_cache = OUTPUT / "rag_results_full.csv"
    if rag_cache.exists():
        print("Loading cached RAG results...")
        rag_df = pd.read_csv(rag_cache)
    else:
        dataset = load_dataset()
        rag_df = run_rag_evaluation(dataset, index, embedder, chunks, rag_models)
        rag_df.to_csv(rag_cache, index=False)

    print("\nComputing RAG vs Oracle comparison...")
    comparison_data = []
    for model in rag_models:
        model_rag = rag_df[rag_df['model'] == model]
        for task in model_rag['task_type'].unique():
            task_rag = model_rag[model_rag['task_type'] == task]
            rag_acc = task_rag['is_correct'].mean() if len(task_rag) > 0 else np.nan
            task_short = TASK_MAP.get(task, task[:3])
            oracle_acc = (task_accs_oracle.loc[model, task_short]
                          if model in task_accs_oracle.index and task_short in task_accs_oracle.columns
                          else np.nan)
            comparison_data.append({
                'model': model, 'task': task_short,
                'oracle_accuracy': oracle_acc, 'rag_accuracy': rag_acc,
                'accuracy_drop': oracle_acc - rag_acc if pd.notna(oracle_acc) and pd.notna(rag_acc) else np.nan
            })

    comp_df = pd.DataFrame(comparison_data)
    comp_df.to_csv(OUTPUT / "rag_oracle_comparison.csv", index=False)
    print("\nRAG vs Oracle Accuracy:")
    print(comp_df.to_string(index=False))

    # ── Figure ────────────────────────────────────────────────────────────
    tasks = ['REG', 'NUM', 'CON', 'TMP']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    x = np.arange(len(tasks))
    width = 0.12
    colors_list = ['#2196F3', '#4CAF50', '#F44336']
    for i, model in enumerate(rag_models):
        model_data = comp_df[comp_df['model'] == model]
        oracle_vals = [model_data[model_data['task'] == t]['oracle_accuracy'].mean() for t in tasks]
        rag_vals = [model_data[model_data['task'] == t]['rag_accuracy'].mean() for t in tasks]
        offset = (i - 1) * width * 2.5
        ax.bar(x + offset - width / 2, oracle_vals, width,
               color=colors_list[i], alpha=0.85, label=f'{model.split()[0]} (Oracle)')
        ax.bar(x + offset + width / 2, rag_vals, width,
               color=colors_list[i], alpha=0.45, hatch='//', label=f'{model.split()[0]} (RAG)')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_xlabel('Task Type', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Oracle vs RAG Context: Per-Task Accuracy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    drop_data = comp_df.dropna(subset=['accuracy_drop'])
    task_drops = drop_data.groupby('task')['accuracy_drop'].mean().reindex(tasks)
    colors_drop = ['#F44336' if d > 0.1 else '#FF9800' if d > 0.05 else '#4CAF50'
                   for d in task_drops.fillna(0)]
    ax2.bar(tasks, task_drops.fillna(0), color=colors_drop, alpha=0.85, edgecolor='white')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Task Type', fontsize=11)
    ax2.set_ylabel('Accuracy Drop (Oracle → RAG)', fontsize=11)
    ax2.set_title('Retrieval Sensitivity by Task\n(Low TMP drop = reasoning failure, not retrieval)',
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, (task, val) in enumerate(zip(tasks, task_drops.fillna(0))):
        ax2.text(i, val + 0.005, f'{val:.1%}', ha='center', fontsize=10)

    plt.tight_layout()
    out_path = FIGURES_DIR / "exp9_rag_evaluation.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    print("=== EXPERIMENT 9 COMPLETE ===")
    tmp_drops = comp_df[comp_df['task'] == 'TMP']['accuracy_drop'].mean()
    num_drops = comp_df[comp_df['task'] == 'NUM']['accuracy_drop'].mean()
    print(f"Key finding: TMP accuracy drop under RAG = {tmp_drops:.1%} vs NUM = {num_drops:.1%}")


if __name__ == "__main__":
    main()
