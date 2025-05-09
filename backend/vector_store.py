import json, pickle, numpy as np, scipy.sparse as sp
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

BASE = Path(__file__).parent
DATA = BASE / "assessments.json"
META = BASE / "metadata.json"
EMBS = BASE / "embeddings.npy"
TFIDF = BASE / "tfidf_vectorizer.pkl"
TFM = BASE / "tfidf_matrix.npz"
NN_IDX = BASE / "nn_model.pkl"

with open(DATA, encoding="utf-8") as f:
    records = json.load(f)
with open(META, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

texts = [r["name"] + " " + r["description"] for r in records]

tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=75000)
tfidf_matrix = tfidf.fit_transform(texts)
sp.save_npz(TFM, tfidf_matrix)
with open(TFIDF, "wb") as f:
    pickle.dump(tfidf, f)

model = SentenceTransformer("all-mpnet-base-v2")
emb_matrix = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
np.save(EMBS, emb_matrix)

nn = NearestNeighbors(n_neighbors=100, metric="cosine", algorithm="brute")
nn.fit(emb_matrix)
with open(NN_IDX, "wb") as f:
    pickle.dump(nn, f)
