import json, pickle, numpy as np, scipy.sparse as sp, requests
from pathlib import Path
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
META_PATH = BASE / "metadata.json"
EMBS_PATH = BASE / "embeddings.npy"
TFIDF_PATH = BASE / "tfidf_vectorizer.pkl"
TFM_PATH = BASE / "tfidf_matrix.npz"
NN_IDX_PATH = BASE / "nn_model.pkl"

DENSE_K = 50
SPARSE_K = 50
FINAL_K = 10
RRF_K = 10

with open(META_PATH, encoding="utf-8") as f:
    records = json.load(f)
with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)
tfidf_matrix = sp.load_npz(str(TFM_PATH))
emb_matrix = np.load(EMBS_PATH)
with open(NN_IDX_PATH, "rb") as f:
    dense_nn = pickle.load(f)

dense_encoder = SentenceTransformer("all-mpnet-base-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    query: str

def extract_text_from_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
    except:
        return ""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=list[dict])
def recommend(req: QueryRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(400, "Query must be non-empty")
    text = extract_text_from_url(q) if q.startswith(("http://", "https://")) else q

    q_sp = tfidf.transform([text])
    sparse_sim = cosine_similarity(q_sp, tfidf_matrix).flatten()
    top_sparse = np.argsort(-sparse_sim)[:SPARSE_K]

    q_emb = dense_encoder.encode([text], convert_to_numpy=True)
    _, top_dense = dense_nn.kneighbors(q_emb, n_neighbors=DENSE_K)
    top_dense = top_dense[0]

    scores = {}
    for rank, idx in enumerate(top_sparse, start=1):
        scores[idx] = scores.get(idx, 0) + 1.0 / (RRF_K + rank)
    for rank, idx in enumerate(top_dense, start=1):
        scores[idx] = scores.get(idx, 0) + 1.0 / (RRF_K + rank)

    candidates = sorted(scores, key=lambda i: scores[i], reverse=True)[:100]
    pairs = [(text, records[i]["name"] + " " + records[i]["description"]) for i in candidates]
    cross_scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, cross_scores), key=lambda x: x[1], reverse=True)
    topk = [i for i, _ in ranked][:FINAL_K]

    return [
        {
            "name": records[i]["name"],
            "url": records[i]["url"],
            "description": records[i]["description"],
            "remote_testing": records[i]["remote_testing"],
            "adaptive": records[i]["adaptive"],
            "duration": records[i]["duration"],
            "test_types": records[i]["test_types"],
            "type": records[i]["type"]
        }
        for i in topk
    ]
