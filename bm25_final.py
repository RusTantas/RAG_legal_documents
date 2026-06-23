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

print("Setting up BM25...")
from rank_bm25 import BM25Okapi

tokenized_docs = [tokenize(text) for text in docs['text']]
bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
doc_ids = docs['doc_id'].tolist()

def bm25_search(query, top_k=50):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_indices]

def recall_at_5(gold_list, preds_list):
    hits = 0
    for gold, preds in zip(gold_list, preds_list):
        if gold in preds[:5]:
            hits += 1
    return hits / len(gold_list)

print("\nTesting BM25 on full train...")
train_preds = []
for q in tqdm(train['question'], desc="Train"):
    train_preds.append([doc_id for doc_id, _ in bm25_search(q, 5)])

r5 = recall_at_5(train['gold_doc_id'].tolist(), train_preds)
print(f"Pure BM25 Recall@5 on FULL train: {r5:.4f}")

print("\nGenerating test predictions...")
test_preds = []
for q in tqdm(test['question'], desc="Test"):
    test_preds.append([doc_id for doc_id, _ in bm25_search(q, 5)])

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

print("\nNow let's try query expansion using topic...")
train_preds_aug = []
for idx, row in tqdm(train.iterrows(), total=len(train), desc="Train with topic"):
    q = row['question']
    topic = str(row.get('topic', ''))
    aug_query = f"{q} {topic}" if topic else q
    train_preds_aug.append([doc_id for doc_id, _ in bm25_search(aug_query, 5)])

r5_aug = recall_at_5(train['gold_doc_id'].tolist(), train_preds_aug)
print(f"BM25 + topic Recall@5 on FULL train: {r5_aug:.4f}")

if r5_aug > r5:
    print("\nTopic augmentation improved! Using it for submission...")
    test_preds_aug = []
    for idx, row in tqdm(test.iterrows(), total=len(test), desc="Test with topic"):
        q = row['question']
        topic = str(row.get('topic', ''))
        aug_query = f"{q} {topic}" if topic else q
        test_preds_aug.append([doc_id for doc_id, _ in bm25_search(aug_query, 5)])

    submission_rows = []
    for qid, top5 in zip(test['qid'], test_preds_aug):
        for doc_id in top5:
            submission_rows.append({'qid': qid, 'doc_id': doc_id})

    submission = pd.DataFrame(submission_rows)
    submission.to_csv(output_path, index=False)
    print(f"Updated submission saved to {output_path}")
else:
    print(f"\nTopic did not help ({r5_aug:.4f} vs {r5:.4f}). Keeping original submission.")