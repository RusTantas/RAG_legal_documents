#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/home/sda/projects/RAG_legal_documents/data')
OUTPUT_DIR = Path('/home/sda/projects/RAG_legal_documents')

print("Loading data...")
docs = pd.read_csv(DATA_DIR / 'documents.csv')
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

print(f"Documents: {len(docs)}, Train: {len(train)}, Test: {len(test)}")

STOPWORDS = set('и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему когда даже ну ли если уже или ни быть был него до вас уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам без чего раз тоже себе под будет тогда кто этот того потому этого какой ним здесь этом один мой тем чтобы нее были куда зачем всех при два об другой хоть после над больше тот через эти нас про всего них какая много три эту перед лучше том такой им более всю между'.split())
TOKEN = re.compile(r'[а-яёa-z0-9]+', re.I)

def tokenize(text):
    if not isinstance(text, str):
        text = str(text)
    tokens = TOKEN.findall(text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]

def preprocess(text):
    if not isinstance(text, str):
        return str(text)
    return text.lower().strip()

print("Setting up BM25...")
from rank_bm25 import BM25Okapi

tokenized_docs = [tokenize(text) for text in tqdm(docs['text'], desc="Tokenizing")]
bm25 = BM25Okapi(tokenized_docs)
doc_ids = docs['doc_id'].tolist()
doc_texts = docs['text'].tolist()
doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

def bm25_search(query, top_k=50):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_indices]

print("Loading e5-large...")
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('intfloat/e5-large')
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

print("Encoding documents...")
doc_embeddings = []
batch_size = 8

for i in tqdm(range(0, len(docs), batch_size), desc="Encoding"):
    batch = [preprocess(text) for text in docs['text'].iloc[i:i+batch_size].tolist()]
    batch = [f"passage: {text[:2000]}" for text in batch]
    batch_embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
    doc_embeddings.append(batch_embs)

doc_embeddings = np.vstack(doc_embeddings).astype('float32')
print(f"Document embeddings: {doc_embeddings.shape}")

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_embeddings)

def dense_search(query, top_k=50):
    query_text = f"query: {preprocess(query)}"
    query_emb = model.encode([query_text], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(query_emb, top_k)
    return [(doc_ids[idx], float(scores[0][i])) for i, idx in enumerate(indices[0])]

def weighted_rrf(bm25_results, dense_results, weights=(0.4, 0.6), top_k=5):
    scores = {}
    
    bm25_max = max(s for _, s in bm25_results) if bm25_results else 1
    for rank, (doc_id, score) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + weights[0] * (score / bm25_max) * (1 / (rank + 1))
    
    for rank, (doc_id, score) in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + weights[1] * score * (1 / (rank + 1))
    
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:top_k]]

def search(query, top_k=20):
    bm25_results = bm25_search(query, top_k=top_k)
    dense_results = dense_search(query, top_k=top_k)
    candidates = weighted_rrf(bm25_results, dense_results, weights=(0.35, 0.65), top_k=top_k)
    return candidates[:top_k]

def recall_at_5(gold_list, preds_list):
    hits = 0
    for gold, preds in zip(gold_list, preds_list):
        if gold in preds[:5]:
            hits += 1
    return hits / len(gold_list)

print("\nEvaluating on training set...")
train_preds = []
for question in tqdm(train['question'], desc="Train search"):
    train_preds.append(search(question, top_k=20))

r5 = recall_at_5(train['gold_doc_id'].tolist(), train_preds)
print(f"Recall@5 on FULL train: {r5:.4f}")

print("\nGenerating test predictions...")
test_preds = []
for question in tqdm(test['question'], desc="Test inference"):
    top5 = search(question, top_k=5)
    test_preds.append(top5)

submission_rows = []
for qid, top5 in zip(test['qid'], test_preds):
    for doc_id in top5:
        submission_rows.append({'qid': qid, 'doc_id': doc_id})

submission = pd.DataFrame(submission_rows)
output_path = OUTPUT_DIR / 'submission.csv'
submission.to_csv(output_path, index=False)

print(f"\nSubmission saved to {output_path}")
print(f"Shape: {submission.shape}")
print(f"Unique qid: {submission['qid'].nunique()} (expected: {len(test)})")