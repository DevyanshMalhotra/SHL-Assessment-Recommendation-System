import json, pickle, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
R = json.load(open(BASE / "metadata.json"))
tf = pickle.load(open(BASE / "tfidf_vectorizer.pkl", "rb"))
tfm = sp.load_npz(str(BASE / "tfidf_matrix.npz"))
nn = pickle.load(open(BASE / "nn_model.pkl", "rb"))
dense_enc = SentenceTransformer("all-mpnet-base-v2")
cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
tests = pd.read_csv(BASE / "tests/test_queries.csv")
embs = np.load(BASE / "embeddings.npy")

def normalize(url):
    u = url.lower().strip().replace("https://www.shl.com", "")
    return u.replace("/products/product-catalog/view/", "/solutions/products/productcatalog/view/").rstrip("/")

records_norm = [normalize(r["url"]) for r in R]

def recall_at_k(rel, ret, k):
    return len(set(ret[:k]) & set(rel)) / len(rel) if rel else 0

def ap_at_k(rel, ret, k):
    if not rel: return 0.0
    hits = 0; score = 0.0
    for i, doc in enumerate(ret[:k], 1):
        if doc in rel:
            hits += 1
            score += hits / i
    return score / min(k, len(rel))

recalls = []; aps = []

for _, row in tests.iterrows():
    q = row["query"]
    rel = [normalize(u) for u in row["relevant_urls"].split("|") if u.strip()]

    v_sparse = tf.transform([q])
    sim_sparse = cosine_similarity(v_sparse, tfm).flatten()
    idx_sparse = np.argsort(-sim_sparse)[:50]

    q_emb = dense_enc.encode([q], convert_to_numpy=True)
    _, idx_dense = nn.kneighbors(q_emb, n_neighbors=50)
    idx_dense = idx_dense[0]

    scores = {}
    for rank, i in enumerate(idx_sparse, 1):
        scores[i] = scores.get(i, 0) + 1 / (60 + rank)
    for rank, i in enumerate(idx_dense, 1):
        scores[i] = scores.get(i, 0) + 1 / (60 + rank)

    candidates = sorted(scores, key=scores.get, reverse=True)[:20]
    pairs = [(q, R[i]["name"] + " " + R[i]["description"]) for i in candidates]
    ce_scores = cross_enc.predict(pairs)

    reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    topk = [idx for idx, _ in reranked][:3]
    ret_urls = [records_norm[i] for i in topk]

    recalls.append(recall_at_k(rel, ret_urls, 3))
    aps.append(ap_at_k(rel, ret_urls, 3))

print(f"Mean Recall@3: {np.mean(recalls):.4f}")
print(f"MAP@3:         {np.mean(aps):.4f}")
