#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi, BM25Plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from catboost import CatBoostClassifier
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

def get_features(query, doc_idx):
    bm25_s = bm25.get_scores(tokenize(query))[doc_idx]
    bm25plus_s = bm25plus.get_scores(tokenize(query))[doc_idx]
    tfidf_s = linear_kernel(tfidf.transform([query]), D)[0][doc_idx]
    return [bm25_s, bm25plus_s, tfidf_s, np.log1p(bm25_s), np.log1p(bm25plus_s), np.log1p(tfidf_s)]

print("\nBuilding training data...")

X_train_list = []
y_train_list = []
group_train = []

for idx, row in tqdm(train.iterrows(), total=len(train), desc="Processing train"):
    query = row['question']
    topic = str(row.get('topic', ''))
    query = f"{query} {topic}" if topic else query
    gold = row['gold_doc_id']
    
    bm25_scores = bm25.get_scores(tokenize(query))
    top50_idx = np.argsort(bm25_scores)[::-1][:50]
    
    for rank, i in enumerate(top50_idx):
        X_train_list.append(get_features(query, i) + [rank])
        y_train_list.append(1 if doc_ids[i] == gold else 0)
        group_train.append(idx)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Positive samples: {sum(y_train)} ({100*sum(y_train)/len(y_train):.2f}%)")

print("\nTraining CatBoost classifier...")
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    random_seed=42,
    verbose=50,
    class_weights={0: 1, 1: 50}  # Boost positive class
)

model.fit(X_train, y_train)

print("\nEvaluating on training set...")
train_preds = []
for idx, row in tqdm(train.iterrows(), total=len(train), desc="Predicting"):
    query = row['question']
    topic = str(row.get('topic', ''))
    query = f"{query} {topic}" if topic else query
    
    bm25_scores = bm25.get_scores(tokenize(query))
    top50_idx = np.argsort(bm25_scores)[::-1][:50]
    
    X_query = np.array([get_features(query, i) + [rank] for rank, i in enumerate(top50_idx)])
    probs = model.predict_proba(X_query)[:, 1]
    
    sorted_idx = np.argsort(probs)[::-1][:5]
    train_preds.append([doc_ids[top50_idx[i]] for i in sorted_idx])

r5 = recall_at_5(train['gold_doc_id'].tolist(), train_preds)
print(f"\nCatBoost Recall@5 on train: {r5:.4f}")

if r5 < 0.6:
    print("\nTrying ensemble with BM25 scores...")
    # Try combining model probs with BM25 scores
    train_preds2 = []
    for idx, row in tqdm(train.iterrows(), total=len(train), desc="Predicting ensemble"):
        query = row['question']
        topic = str(row.get('topic', ''))
        query = f"{query} {topic}" if topic else query
        
        bm25_scores = bm25.get_scores(tokenize(query))
        top50_idx = np.argsort(bm25_scores)[::-1][:50]
        
        X_query = np.array([get_features(query, i) + [rank] for rank, i in enumerate(top50_idx)])
        probs = model.predict_proba(X_query)[:, 1]
        
        # Combine with BM25 scores (normalized)
        bm25_top = bm25_scores[top50_idx]
        bm25_norm = (bm25_top - bm25_top.min()) / (bm25_top.max() - bm25_top.min() + 1e-8)
        combined = 0.6 * probs + 0.4 * bm25_norm
        
        sorted_idx = np.argsort(combined)[::-1][:5]
        train_preds2.append([doc_ids[top50_idx[i]] for i in sorted_idx])
    
    r5_ens = recall_at_5(train['gold_doc_id'].tolist(), train_preds2)
    print(f"Ensemble Recall@5 on train: {r5_ens:.4f}")
    
    if r5_ens > r5:
        train_preds = train_preds2
        r5 = r5_ens
        print("Using ensemble!")

print(f"\nFinal Recall@5: {r5:.4f}")

print("\nGenerating test predictions...")
test_preds = []
for idx, row in tqdm(test.iterrows(), total=len(test), desc="Predicting test"):
    query = row['question']
    topic = str(row.get('topic', ''))
    query = f"{query} {topic}" if topic else query
    
    bm25_scores = bm25.get_scores(tokenize(query))
    top50_idx = np.argsort(bm25_scores)[::-1][:50]
    
    X_query = np.array([get_features(query, i) + [rank] for rank, i in enumerate(top50_idx)])
    probs = model.predict_proba(X_query)[:, 1]
    
    # Combine with BM25 scores
    bm25_top = bm25_scores[top50_idx]
    bm25_norm = (bm25_top - bm25_top.min()) / (bm25_top.max() - bm25_top.min() + 1e-8)
    combined = 0.6 * probs + 0.4 * bm25_norm
    
    sorted_idx = np.argsort(combined)[::-1][:5]
    test_preds.append([doc_ids[top50_idx[i]] for i in sorted_idx])

submission_rows = []
for qid, top5 in zip(test['qid'], test_preds):
    for doc_id in top5:
        submission_rows.append({'qid': qid, 'doc_id': doc_id})

submission = pd.DataFrame(submission_rows)
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
print(f"\nSubmission saved! Shape: {submission.shape}")