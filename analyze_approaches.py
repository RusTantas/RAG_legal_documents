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

print("Setting up BM25 (k1=1.5, b=0.75)...")
from rank_bm25 import BM25Okapi, BM25Plus

tokenized_docs = [tokenize(text) for text in docs['text']]
bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
doc_ids = docs['doc_id'].tolist()

def bm25_search(query, top_k=50):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_indices]

print("\nTesting different approaches...")

def recall_at_5(gold_list, preds_list):
    hits = 0
    for gold, preds in zip(gold_list, preds_list):
        if gold in preds[:5]:
            hits += 1
    return hits / len(gold_list)

sample = train.head(100)

print("\n1. Pure BM25:")
preds = [[doc_id for doc_id, _ in bm25_search(q, 5)] for q in tqdm(sample['question'])]
r5 = recall_at_5(sample['gold_doc_id'].tolist(), preds)
print(f"   Recall@5: {r5:.4f}")

print("\n2. Pure BM25 with more candidates:")
preds = []
for q in tqdm(sample['question']):
    results = bm25_search(q, 100)
    preds.append([doc_id for doc_id, _ in results])
r5 = recall_at_5(sample['gold_doc_id'].tolist(), preds)
print(f"   Recall@5: {r5:.4f}")

print("\n3. Testing different k1, b parameters...")
best_r5 = 0
best_params = None
for k1 in [0.5, 1.0, 1.5, 2.0]:
    for b in [0.5, 0.75, 1.0]:
        bm25_test = BM25Okapi(tokenized_docs, k1=k1, b=b)
        preds = [[doc_id for doc_id, _ in bm25_search(q, 5)] for q in sample['question']]
        r5 = recall_at_5(sample['gold_doc_id'].tolist(), preds)
        if r5 > best_r5:
            best_r5 = r5
            best_params = (k1, b)
            print(f"   k1={k1}, b={b}: {r5:.4f} *")
        else:
            print(f"   k1={k1}, b={b}: {r5:.4f}")

print(f"\nBest params: k1={best_params[0]}, b={best_params[1]} with Recall@5={best_r5:.4f}")

print("\n4. Dense embeddings only (m-E5-small)...")
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('intfloat/multilingual-e5-small')
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

def preprocess(text):
    return text.lower().strip()[:1500]

print("Encoding documents...")
doc_embeddings = []
for i in tqdm(range(0, len(docs), 32)):
    batch = [f"passage: {preprocess(text)}" for text in docs['text'].iloc[i:i+32].tolist()]
    batch_embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
    doc_embeddings.append(batch_embs)
doc_embeddings = np.vstack(doc_embeddings).astype('float32')

index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)

def dense_search(query, top_k=5):
    q_emb = model.encode([f"query: {preprocess(query)}"], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(q_emb, top_k)
    return [(doc_ids[idx], float(scores[0][i])) for i, idx in enumerate(indices[0])]

preds = [[doc_id for doc_id, _ in dense_search(q, 5)] for q in tqdm(sample['question'])]
r5 = recall_at_5(sample['gold_doc_id'].tolist(), preds)
print(f"   Dense-only Recall@5: {r5:.4f}")

print("\n5. Trying to use gold_evidence_text as query augmentation...")
augmented_queries = []
for idx, row in sample.iterrows():
    q = row['question']
    aug = q + " " + str(row.get('gold_evidence_text', ''))[:200]
    augmented_queries.append(aug)

preds = [[doc_id for doc_id, _ in bm25_search(q, 5)] for q in tqdm(augmented_queries)]
r5 = recall_at_5(sample['gold_doc_id'].tolist(), preds)
print(f"   Augmented query Recall@5: {r5:.4f}")