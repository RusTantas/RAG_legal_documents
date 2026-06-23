#!/usr/bin/env python3
"""
Improved RAG Retrieval for Legal Documents
Key improvements:
1. e5-large instead of e5-small for better embeddings
2. Weighted RRF combining BM25 + Dense + Lexical overlap
3. Russian text preprocessing
4. Proper Russian cross-encoder for reranking
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
DATA_DIR = Path('/home/sda/projects/RAG_legal_documents/data')
OUTPUT_DIR = Path('/home/sda/projects/RAG_legal_documents')
EMBEDDING_MODEL = 'intfloat/e5-large'  # Better than e5-small
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'  # Will try Russian one

# ============ DATA LOADING ============
print("Loading data...")
docs = pd.read_csv(DATA_DIR / 'documents.csv')
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

print(f"Documents: {len(docs)}, Train: {len(train)}, Test: {len(test)}")

# ============ PREPROCESSING ============
# Russian stopwords
STOPWORDS = set('и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему когда даже ну ли если уже или ни быть был него до вас уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам без чего раз тоже себе под будет тогда кто этот того потому этого какой ним здесь этом один мой тем чтобы нее были куда зачем всех при два об другой хоть после над больше тот через эти нас про всего них какая много три эту перед лучше том такой им более всю между'.split())

TOKEN = re.compile(r'[а-яёa-z0-9]+', re.I)

def tokenize(text):
    """Tokenize with Russian stopwords"""
    if not isinstance(text, str):
        text = str(text)
    tokens = TOKEN.findall(text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]

def preprocess(text):
    """Light preprocessing"""
    if not isinstance(text, str):
        return str(text)
    return text.lower().strip()

# ============ BM25 SETUP ============
print("Setting up BM25...")
from rank_bm25 import BM25Okapi

tokenized_docs = [tokenize(text) for text in tqdm(docs['text'], desc="Tokenizing")]
bm25 = BM25Okapi(tokenized_docs)
doc_ids = docs['doc_id'].tolist()
doc_texts = docs['text'].tolist()

# Create doc_id to index mapping
doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

def bm25_search(query, top_k=50):
    """BM25 search returning (doc_id, score)"""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_indices]

# ============ DENSE SEARCH SETUP ============
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer(EMBEDDING_MODEL)
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Encode documents
print("Encoding documents...")
doc_embeddings = []
batch_size = 16  # Smaller batch for larger model

for i in tqdm(range(0, len(docs), batch_size), desc="Encoding"):
    batch = [preprocess(text) for text in docs['text'].iloc[i:i+batch_size].tolist()]
    # e5 models require "query: " or "passage: " prefix
    batch = [f"passage: {text[:2000]}" for text in batch]  # Truncate long docs
    batch_embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
    doc_embeddings.append(batch_embs)

doc_embeddings = np.vstack(doc_embeddings).astype('float32')
print(f"Document embeddings shape: {doc_embeddings.shape}")

# Create FAISS index
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_embeddings)

def dense_search(query, top_k=50):
    """Dense semantic search returning (doc_id, score)"""
    query_text = f"query: {preprocess(query)}"
    query_emb = model.encode([query_text], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(query_emb, top_k)
    return [(doc_ids[idx], float(scores[0][i])) for i, idx in enumerate(indices[0])]

# ============ LEXICAL OVERLAP ============
def lexical_overlap(query, doc_text, top_k=50):
    """Calculate word overlap between query and document"""
    query_tokens = set(tokenize(query))
    doc_tokens = set(tokenize(doc_text))
    
    if not query_tokens:
        return []
    
    overlap_scores = []
    for i, text in enumerate(doc_texts):
        doc_token_set = set(tokenize(text))
        # Jaccard-like score
        intersection = len(query_tokens & doc_token_set)
        union = len(query_tokens | doc_token_set)
        score = intersection / union if union > 0 else 0
        overlap_scores.append((doc_ids[i], score))
    
    # Sort by overlap score
    overlap_scores.sort(key=lambda x: x[1], reverse=True)
    return overlap_scores[:top_k]

# ============ RECIPROCAL RANK FUSION ============
def weighted_rrf(bm25_results, dense_results, lex_results=None, weights=(0.4, 0.6, 0.0), top_k=5):
    """Weighted Reciprocal Rank Fusion"""
    scores = {}
    
    # BM25 scores (normalized)
    bm25_max = max(s for _, s in bm25_results) if bm25_results else 1
    for rank, (doc_id, score) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + weights[0] * (score / bm25_max) * (1 / (rank + 1))
    
    # Dense scores (already normalized via cosine)
    for rank, (doc_id, score) in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + weights[1] * score * (1 / (rank + 1))
    
    # Lexical overlap
    if lex_results and weights[2] > 0:
        for rank, (doc_id, score) in enumerate(lex_results):
            scores[doc_id] = scores.get(doc_id, 0) + weights[2] * score * (1 / (rank + 1))
    
    # Sort by combined score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:top_k]]

# ============ RERANKER ============
print("Loading reranker...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

try:
    # Try to load a multilingual/Russian model first
    reranker_name = 'sberbank-ai/bert-base-multilingual-cased-sentence'
    tokenizer = AutoTokenizer.from_pretrained(reranker_name)
    reranker = AutoModelForSequenceClassification.from_pretrained(reranker_name)
    reranker.eval()
    USE_RERANKER = True
    print(f"Reranker loaded: {reranker_name}")
except Exception as e:
    print(f"Could not load multilingual reranker: {e}")
    print("Using reranker without Russian support")
    reranker_name = RERANKER_MODEL
    tokenizer = AutoTokenizer.from_pretrained(reranker_name)
    reranker = AutoModelForSequenceClassification.from_pretrained(reranker_name)
    reranker.eval()
    USE_RERANKER = True

def rerank(query, candidates, top_k=5):
    """Rerank candidates using cross-encoder"""
    if len(candidates) <= 1:
        return candidates[:top_k]
    
    # Get document texts for candidates
    candidate_texts = []
    for doc_id in candidates:
        idx = doc_id_to_idx[doc_id]
        text = doc_texts[idx][:1024]  # Truncate for reranker
        candidate_texts.append(text)
    
    # Create pairs [query, document]
    pairs = [[query, text] for text in candidate_texts]
    
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        outputs = reranker(**inputs)
        scores = outputs.logits.squeeze().tolist()
    
    if isinstance(scores, float):
        scores = [scores]
    
    # Sort by reranker score
    sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_pairs[:top_k]]

# ============ FULL PIPELINE ============
def search(query, top_k=20, use_rerank=True):
    """Full search pipeline: BM25 + Dense + RRF + Rerank"""
    # Get candidates from multiple methods
    bm25_results = bm25_search(query, top_k=top_k)
    dense_results = dense_search(query, top_k=top_k)
    lex_results = lexical_overlap(query, None, top_k=top_k)
    
    # Combine with weighted RRF
    candidates = weighted_rrf(bm25_results, dense_results, lex_results, weights=(0.35, 0.65, 0.0), top_k=top_k)
    
    # Rerank if enabled
    if use_rerank and len(candidates) > 1:
        candidates = rerank(query, candidates, top_k=top_k)
    
    return candidates[:top_k]

# ============ EVALUATION ============
def recall_at_5(gold_list, preds_list):
    """Calculate Recall@5"""
    hits = 0
    for gold, preds in zip(gold_list, preds_list):
        if gold in preds[:5]:
            hits += 1
    return hits / len(gold_list)

# ============ TRAIN EVALUATION ============
print("\nEvaluating on training set (first 200 samples for speed)...")
train_sample = train.head(200)

preds = []
for question in tqdm(train_sample['question'], desc="Searching"):
    preds.append(search(question, top_k=20, use_rerank=True))

r5 = recall_at_5(train_sample['gold_doc_id'].tolist(), preds)
print(f"Recall@5 on train sample (200): {r5:.4f}")

# ============ GENERATE SUBMISSION ============
print("\nGenerating predictions for test set...")
test_preds = []

for question in tqdm(test['question'], desc="Test inference"):
    top5 = search(question, top_k=5, use_rerank=True)
    test_preds.append(top5)

# Create submission DataFrame
submission_rows = []
for qid, top5 in zip(test['qid'], test_preds):
    for doc_id in top5:
        submission_rows.append({'qid': qid, 'doc_id': doc_id})

submission = pd.DataFrame(submission_rows)

# Save
output_path = OUTPUT_DIR / 'submission.csv'
submission.to_csv(output_path, index=False)
print(f"\nSubmission saved to {output_path}")
print(f"Submission shape: {submission.shape}")

# Verify format
print(f"\nSubmission verification:")
print(f"  Unique qid: {submission['qid'].nunique()} (expected: {len(test)})")
print(f"  Rows per qid: {submission.groupby('qid').size().unique()}")