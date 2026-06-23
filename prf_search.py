#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi, BM25Plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/home/sda/projects/RAG_legal_documents/data')
OUTPUT_DIR = Path('/home/sda/projects/RAG_legal_documents')

print("Loading data...")
docs = pd.read_csv(DATA_DIR / 'documents.csv')
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

STOPWORDS = set('и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему когда даже ну ли если уже или ни быть был него до вас уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам без чего раз тоже себе под будет тогда кто этот того потому этого какой ним здесь этом один мой тем чтобы нее были куда зачем всех при два об другой хоть после над больше тот через эти нас про всего них какая много три эту перед лучше том такой им более всю между'.split())
TOKEN = re.compile(r'[а-яёa-z0-9]+', re.I)

def tokenize(text):
    if not isinstance(text, str):
        text = str(text)
    tokens = TOKEN.findall(text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]

doc_ids = docs['doc_id'].tolist()
doc_texts = docs['text'].tolist()
doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

print("Setting up BM25 and TF-IDF...")
tokenized_docs = [tokenize(text) for text in docs['text']]
bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
bm25plus = BM25Plus(tokenized_docs, k1=1.5, b=0.75)

tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
D = tfidf.fit_transform(docs['text'])

def recall_at_5(gold_list, preds_list):
    hits = 0
    for gold, preds in zip(gold_list, preds_list):
        if gold in preds[:5]:
            hits += 1
    return hits / len(gold_list)

def get_combined_scores(query, doc_idx):
    bm25_s = bm25.get_scores(tokenize(query))[doc_idx]
    bm25plus_s = bm25plus.get_scores(tokenize(query))[doc_idx]
    tfidf_s = linear_kernel(tfidf.transform([query]), D)[0][doc_idx]
    return 0.5 * bm25plus_s + 0.3 * bm25_s + 0.2 * tfidf_s

def prf_search(query, top_k=50, n_prf=5, alpha=0.5):
    """Pseudo-relevance feedback: expand query with top documents"""
    bm25_scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    
    if n_prf > 0:
        expanded_query = query
        for i in range(n_prf):
            if i < len(top_indices):
                doc_text = doc_texts[top_indices[i]]
                expanded_query = f"{expanded_query} {doc_text[:500]}"
    
    expanded_scores = bm25plus.get_scores(tokenize(expanded_query))
    final_scores = alpha * expanded_scores + (1 - alpha) * bm25_scores
    
    final_indices = np.argsort(final_scores)[::-1][:5]
    return [doc_ids[i] for i in final_indices]

print("\nTesting PRF approaches on 200 samples...")
sample = train.head(200)

print("\n1. Pure BM25Plus + topic:")
preds = []
for idx, row in sample.iterrows():
    q = f"{row['question']} {row['topic']}"
    bm25plus_scores = bm25plus.get_scores(tokenize(q))
    top_idx = np.argsort(bm25plus_scores)[::-1][:5]
    preds.append([doc_ids[i] for i in top_idx])
r5 = recall_at_5(sample['gold_doc_id'].tolist(), preds)
print(f"   Recall@5: {r5:.4f}")

print("\n2. BM25Plus + topic + PRF (n_prf=1, alpha=0.7):")
preds = []
for idx, row in sample.iterrows():
    q = f"{row['question']} {row['topic']}"
    preds.append(prf_search(q, n_prf=1, alpha=0.7))
r5 = recall_at_5(sample['gold_doc_id'].tolist(), preds)
print(f"   Recall@5: {r5:.4f}")

print("\n3. BM25Plus + topic + PRF (n_prf=2, alpha=0.6):")
preds = []
for idx, row in sample.iterrows():
    q = f"{row['question']} {row['topic']}"
    preds.append(prf_search(q, n_prf=2, alpha=0.6))
r5 = recall_at_5(sample['gold_doc_id'].tolist(), preds)
print(f"   Recall@5: {r5:.4f}")

print("\n4. Combined scores (BM25Plus + BM25 + TF-IDF):")
preds = []
for idx, row in sample.iterrows():
    q = f"{row['question']} {row['topic']}"
    combined = []
    for i in range(len(doc_ids)):
        combined.append((doc_ids[i], get_combined_scores(q, i)))
    combined.sort(key=lambda x: x[1], reverse=True)
    preds.append([d for d, _ in combined[:5]])
r5 = recall_at_5(sample['gold_doc_id'].tolist(), preds)
print(f"   Recall@5: {r5:.4f}")

print("\n5. Best: Combined + PRF:")
best_r5 = 0
best_params = None
for n_prf in [1, 2, 3]:
    for alpha in [0.5, 0.6, 0.7, 0.8]:
        preds = []
        for idx, row in sample.iterrows():
            q = f"{row['question']} {row['topic']}"
            
            bm25_scores = bm25.get_scores(tokenize(q))
            top_indices = np.argsort(bm25_scores)[::-1][:50]
            
            expanded_query = q
            for i in range(n_prf):
                if i < len(top_indices):
                    expanded_query = f"{expanded_query} {doc_texts[top_indices[i]][:300]}"
            
            expanded_scores = bm25plus.get_scores(tokenize(expanded_query))
            final_scores = alpha * expanded_scores + (1 - alpha) * bm25_scores
            final_indices = np.argsort(final_scores)[::-1][:5]
            preds.append([doc_ids[i] for i in final_indices])
        
        r5 = recall_at_5(sample['gold_doc_id'].tolist(), preds)
        print(f"   n_prf={n_prf}, alpha={alpha}: {r5:.4f}")
        if r5 > best_r5:
            best_r5 = r5
            best_params = (n_prf, alpha)

print(f"\nBest params: n_prf={best_params[0]}, alpha={best_params[1]} with {best_r5:.4f}")

if best_r5 >= 0.6:
    print("\n" + "="*60)
    print("GENERATING SUBMISSION WITH BEST PARAMS")
    print("="*60)
    
    n_prf, alpha = best_params
    
    print(f"\nEvaluating on FULL train...")
    train_preds = []
    for idx, row in tqdm(train.iterrows(), total=len(train)):
        q = f"{row['question']} {row['topic']}"
        
        bm25_scores = bm25.get_scores(tokenize(q))
        top_indices = np.argsort(bm25_scores)[::-1][:50]
        
        expanded_query = q
        for i in range(n_prf):
            if i < len(top_indices):
                expanded_query = f"{expanded_query} {doc_texts[top_indices[i]][:300]}"
        
        expanded_scores = bm25plus.get_scores(tokenize(expanded_query))
        final_scores = alpha * expanded_scores + (1 - alpha) * bm25_scores
        final_indices = np.argsort(final_scores)[::-1][:5]
        train_preds.append([doc_ids[i] for i in final_indices])
    
    r5_full = recall_at_5(train['gold_doc_id'].tolist(), train_preds)
    print(f"Final Recall@5 on FULL train: {r5_full:.4f}")
    
    print("\nGenerating test predictions...")
    test_preds = []
    for idx, row in tqdm(test.iterrows(), total=len(test)):
        q = f"{row['question']}"
        
        bm25_scores = bm25.get_scores(tokenize(q))
        top_indices = np.argsort(bm25_scores)[::-1][:50]
        
        expanded_query = q
        for i in range(n_prf):
            if i < len(top_indices):
                expanded_query = f"{expanded_query} {doc_texts[top_indices[i]][:300]}"
        
        expanded_scores = bm25plus.get_scores(tokenize(expanded_query))
        final_scores = alpha * expanded_scores + (1 - alpha) * bm25_scores
        final_indices = np.argsort(final_scores)[::-1][:5]
        test_preds.append([doc_ids[i] for i in final_indices])
    
    submission_rows = []
    for qid, top5 in zip(test['qid'], test_preds):
        for doc_id in top5:
            submission_rows.append({'qid': qid, 'doc_id': doc_id})
    
    submission = pd.DataFrame(submission_rows)
    submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
    print(f"\nSubmission saved! Shape: {submission.shape}")
    print(f"Recall@5: {r5_full:.4f}")
else:
    print(f"\nBest Recall@5 = {best_r5:.4f} < 0.6, not generating submission yet")