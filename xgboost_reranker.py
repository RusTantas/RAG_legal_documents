#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
from rank_bm25 import BM25Okapi, BM25Plus
from pathlib import Path
import xgboost as xgb

DATA_DIR = Path('data')
OUTPUT_DIR = Path('.')
docs = pd.read_csv(DATA_DIR / 'documents.csv')
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

TOKEN = re.compile(r'[а-яёa-z0-9]+', re.I)

def tokenize(text):
    if not isinstance(text, str):
        text = str(text)
    return TOKEN.findall(text.lower())

doc_ids = docs['doc_id'].tolist()
N_DOCS = len(doc_ids)

print("Building BM25 indices...")
tokenized_docs = [tokenize(t) for t in docs['text']]
bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
bm25plus = BM25Plus(tokenized_docs, k1=1.5, b=0.75)

def recall_at_k(gold_list, preds_list, k):
    return sum(1 for g, p in zip(gold_list, preds_list) if g in p[:k]) / len(gold_list)

print("Precomputing BM25 scores for all train queries...")
queries = [f"{row['question']} {row['topic']}" for _, row in train.iterrows()]
tokenized_queries = [tokenize(q) for q in queries]
gold_docs = train['gold_doc_id'].tolist()

bm25_scores = np.array([bm25.get_scores(tq) for tq in tokenized_queries], dtype=np.float32)
bm25plus_scores = np.array([bm25plus.get_scores(tq) for tq in tokenized_queries], dtype=np.float32)

def compute_features_batch(queries_tokens, docs_tokens_list, bm25_s, bm25plus_s, top_k=50):
    n_queries = len(queries_tokens)
    features = np.zeros((n_queries * top_k, 12), dtype=np.float32)
    labels = np.zeros(n_queries * top_k, dtype=np.int32)
    
    gold_indices = [doc_ids.index(g) for g in gold_docs]
    
    for i in range(n_queries):
        q_set = set(queries_tokens[i])
        q_len = len(queries_tokens[i])
        top_indices = np.argsort(-bm25_s[i])[:top_k]
        
        for j, doc_idx in enumerate(top_indices):
            d_tokens = docs_tokens_list[doc_idx]
            d_set = set(d_tokens)
            d_len = len(d_tokens)
            
            base_idx = i * top_k + j
            bm25_norm = bm25_s[i, doc_idx]
            bm25plus_norm = bm25plus_s[i, doc_idx]
            jaccard = len(q_set & d_set) / max(len(q_set | d_set), 1) if q_set or d_set else 0
            overlap_ratio = len(q_set & d_set) / max(q_len, 1)
            doc_cover = len(q_set & d_set) / max(d_len, 1)
            
            features[base_idx] = [
                np.log1p(bm25_norm), np.log1p(bm25plus_norm),
                jaccard, overlap_ratio, doc_cover,
                q_len, d_len, j,
                bm25_norm * jaccard, bm25plus_norm * overlap_ratio,
                np.log1p(bm25_norm) * (j + 1), 1.0 / (j + 1),
            ]
            labels[base_idx] = 1 if doc_idx == gold_indices[i] else 0
    
    return features, labels

print("Building features...")
X_train, y_train = compute_features_batch(tokenized_queries, tokenized_docs, bm25_scores, bm25plus_scores)
print(f"X_train: {X_train.shape}, Positive: {sum(y_train)} ({100*sum(y_train)/len(y_train):.2f}%)")

print("Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
    scale_pos_weight=len(y_train) / sum(y_train), tree_method='hist', n_jobs=-1,
    use_label_encoder=False, eval_metric='logloss'
)
model.fit(X_train, y_train, verbose=False)
print("Training complete!")

print("Predicting on train...")
train_probs = model.predict_proba(X_train)[:, 1]

train_preds = []
for i in range(len(queries)):
    query_probs = train_probs[i * 50:(i + 1) * 50]
    top50_idx = np.argsort(-bm25_scores[i])[:50]
    sorted_local_idx = np.argsort(-query_probs)
    sorted_doc_idx = top50_idx[sorted_local_idx]
    train_preds.append([doc_ids[idx] for idx in sorted_doc_idx[:5]])

r5 = recall_at_k(gold_docs, train_preds, 5)
print(f"\n{'='*60}")
print(f"XGBoost Enhanced Reranker Recall@5: {r5:.4f}")
print(f"{'='*60}")

print("Generating submission...")
test_queries = [row['question'] for _, row in test.iterrows()]
test_tokenized = [tokenize(q) for q in test_queries]

test_bm25 = np.array([bm25.get_scores(tq) for tq in test_tokenized], dtype=np.float32)
test_bm25plus = np.array([bm25plus.get_scores(tq) for tq in test_tokenized], dtype=np.float32)

X_test, _ = compute_features_batch(test_tokenized, tokenized_docs, test_bm25, test_bm25plus)
test_probs = model.predict_proba(X_test)[:, 1]

test_preds = []
for i in range(len(test_queries)):
    query_probs = test_probs[i * 50:(i + 1) * 50]
    top50_idx = np.argsort(-test_bm25[i])[:50]
    sorted_local_idx = np.argsort(-query_probs)
    sorted_doc_idx = top50_idx[sorted_local_idx]
    test_preds.append([doc_ids[idx] for idx in sorted_doc_idx[:5]])

submission_rows = [{'qid': qid, 'doc_id': doc_id} for qid, top5 in zip(test['qid'], test_preds) for doc_id in top5]
pd.DataFrame(submission_rows).to_csv(OUTPUT_DIR / 'submission.csv', index=False)
print("Submission saved!")