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

print(f"Documents: {len(docs)}, Train: {len(train)}, Test: {len(test)}")

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

print("Setting up BM25...")
tokenized_docs = [tokenize(text) for text in docs['text']]
bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
bm25plus = BM25Plus(tokenized_docs, k1=1.5, b=0.75)

print("Setting up TF-IDF...")
tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
D = tfidf.fit_transform(docs['text'])

def get_bm25_scores(query):
    tokenized = tokenize(query)
    return bm25.get_scores(tokenized)

def get_bm25plus_scores(query):
    tokenized = tokenize(query)
    return bm25plus.get_scores(tokenized)

def get_tfidf_scores(query):
    Q = tfidf.transform([query])
    return linear_kernel(Q, D)[0]

def recall_at_5(gold_list, preds_list):
    hits = 0
    for gold, preds in zip(gold_list, preds_list):
        if gold in preds[:5]:
            hits += 1
    return hits / len(gold_list)

print("\nBuilding training data for Learning-to-Rank...")

def build_feature_matrix(queries, doc_ids_list):
    """Build feature matrix for a list of queries and candidate doc_ids"""
    features = []
    labels = []
    
    for q_idx, (qid, query) in enumerate(tqdm(queries, desc="Building features")):
        bm25_scores = get_bm25_scores(query)
        bm25plus_scores = get_bm25plus_scores(query)
        tfidf_scores = get_tfidf_scores(query)
        
        for doc_id in doc_ids_list:
            doc_idx = doc_id_to_idx[doc_id]
            
            feat = [
                bm25_scores[doc_idx],
                bm25plus_scores[doc_idx],
                tfidf_scores[doc_idx],
                np.log1p(bm25_scores[doc_idx]),
                np.log1p(bm25plus_scores[doc_idx]),
                np.log1p(tfidf_scores[doc_idx]),
                bm25_scores[doc_idx] * tfidf_scores[doc_idx],
            ]
            features.append(feat)
    
    return np.array(features)

# Create training data
# For each query, get top-50 candidates from BM25 and build pairs with gold doc
print("\nGenerating candidate sets and building features...")

train_queries = []
train_candidates = []
train_labels = []

for idx, row in tqdm(train.iterrows(), total=len(train), desc="Processing train"):
    qid = row['qid']
    query = row['question']
    gold_doc = row['gold_doc_id']
    topic = str(row.get('topic', ''))
    aug_query = f"{query} {topic}" if topic else query
    
    # Get top-50 candidates from BM25+topic
    bm25_scores = get_bm25_scores(aug_query)
    top_indices = np.argsort(bm25_scores)[::-1][:50]
    candidates = [doc_ids[i] for i in top_indices]
    
    # Only keep if gold is in candidates
    if gold_doc not in candidates:
        # Add gold manually at position 25
        candidates = candidates[:24] + [gold_doc] + candidates[24:]
    
    train_queries.append((qid, aug_query))
    train_candidates.append(candidates)
    
    # Labels: 1 for gold, 0 for others
    labels = [1 if doc_id == gold_doc else 0 for doc_id in candidates]
    train_labels.extend(labels)

print(f"Total training samples: {len(train_labels)}")
print(f"Positive samples: {sum(train_labels)}")

# Build feature matrix
print("\nBuilding feature matrix...")
all_doc_ids = []
for candidates in train_candidates:
    all_doc_ids.extend(candidates)

X_train = build_feature_matrix(train_queries, [doc_id for candidates in train_candidates for doc_id in candidates])
y_train = np.array(train_labels)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Train CatBoost Ranker
print("\nTraining CatBoost Ranker...")
from catboost import CatBoostRanker, Pool

# Group sizes (number of candidates per query)
group_sizes = [len(candidates) for candidates in train_candidates]

# Create group weights
group_weights = np.ones(len(group_sizes))

# Create train pool with groups
train_pool = Pool(
    data=X_train,
    label=y_train,
    group_id=[i for i, size in enumerate(group_sizes) for _ in range(size)]
)

# CatBoost Ranker
model = CatBoostRanker(
    loss_function='QueryRMSE',
    iterations=200,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=50
)

model.fit(train_pool)

print("\nEvaluating on training set...")
train_preds = []
for (qid, query), candidates in zip(train_queries, train_candidates):
    # Get features for this query's candidates
    X_query = []
    bm25_scores = get_bm25_scores(query)
    bm25plus_scores = get_bm25plus_scores(query)
    tfidf_scores = get_tfidf_scores(query)
    
    for doc_id in candidates:
        doc_idx = doc_id_to_idx[doc_id]
        feat = [
            bm25_scores[doc_idx],
            bm25plus_scores[doc_idx],
            tfidf_scores[doc_idx],
            np.log1p(bm25_scores[doc_idx]),
            np.log1p(bm25plus_scores[doc_idx]),
            np.log1p(tfidf_scores[doc_idx]),
            bm25_scores[doc_idx] * tfidf_scores[doc_idx],
        ]
        X_query.append(feat)
    
    X_query = np.array(X_query)
    preds = model.predict(X_query)
    
    # Sort by prediction score
    sorted_indices = np.argsort(preds)[::-1][:5]
    train_preds.append([candidates[i] for i in sorted_indices])

r5 = recall_at_5(train['gold_doc_id'].tolist(), train_preds)
print(f"\nCatBoost Ranker Recall@5 on train: {r5:.4f}")

# Generate test predictions
print("\nGenerating test predictions...")
test_queries = []
test_candidates = []

for idx, row in tqdm(test.iterrows(), total=len(test), desc="Processing test"):
    qid = row['qid']
    query = row['question']
    topic = str(row.get('topic', ''))
    aug_query = f"{query} {topic}" if topic else query
    
    bm25_scores = get_bm25_scores(aug_query)
    top_indices = np.argsort(bm25_scores)[::-1][:50]
    candidates = [doc_ids[i] for i in top_indices]
    
    test_queries.append((qid, aug_query))
    test_candidates.append(candidates)

test_preds = []
for (qid, query), candidates in zip(test_queries, test_candidates):
    X_query = []
    bm25_scores = get_bm25_scores(query)
    bm25plus_scores = get_bm25plus_scores(query)
    tfidf_scores = get_tfidf_scores(query)
    
    for doc_id in candidates:
        doc_idx = doc_id_to_idx[doc_id]
        feat = [
            bm25_scores[doc_idx],
            bm25plus_scores[doc_idx],
            tfidf_scores[doc_idx],
            np.log1p(bm25_scores[doc_idx]),
            np.log1p(bm25plus_scores[doc_idx]),
            np.log1p(tfidf_scores[doc_idx]),
            bm25_scores[doc_idx] * tfidf_scores[doc_idx],
        ]
        X_query.append(feat)
    
    X_query = np.array(X_query)
    preds = model.predict(X_query)
    
    sorted_indices = np.argsort(preds)[::-1][:5]
    test_preds.append([candidates[i] for i in sorted_indices])

# Create submission
submission_rows = []
for qid, top5 in zip(test['qid'], test_preds):
    for doc_id in top5:
        submission_rows.append({'qid': qid, 'doc_id': doc_id})

submission = pd.DataFrame(submission_rows)
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)

print(f"\nSubmission saved!")
print(f"Shape: {submission.shape}")
print(f"Recall@5 on train: {r5:.4f}")