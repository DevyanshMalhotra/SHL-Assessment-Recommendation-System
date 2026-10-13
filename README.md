# SHL Assessment Recommendation System

A hybrid search and ranking prototype for recommending SHL assessments from a job description or skills query. It combines **TF-IDF retrieval**, **dense sentence embeddings**, **reciprocal rank fusion**, and **cross-encoder reranking** behind a FastAPI endpoint, with a Streamlit interface.

## How it works

1. `backend/data_ingestion.py` prepares assessment metadata in `assessments.json`.
2. `backend/vector_store.py` creates the sparse TF-IDF matrix and dense vector index from the assessment names and descriptions.
3. `POST /recommend` retrieves up to 50 candidates from each index, merges rankings using reciprocal rank fusion, and reranks candidates using a cross-encoder.
4. The API returns up to 10 assessments with names, links, descriptions, duration and testing metadata.

**Models:** `sentence-transformers/all-mpnet-base-v2` for dense retrieval and `cross-encoder/ms-marco-MiniLM-L-12-v2` for reranking. The code also accepts a URL as a query and attempts to extract its page text.

## Repository layout

```text
backend/
  app.py                  FastAPI application and hybrid retrieval
  assessments.json        Assessment catalog snapshot
  metadata.json           Metadata used by the indexer
  vector_store.py         Sparse and dense index construction
  data_ingestion.py       Catalog preparation
  eval.py                 Recall@3 and MAP@3 evaluation
  tests/test_queries.csv  Evaluation queries
  requirements.txt
frontend/
  streamlit_app.py        Browser interface
  requirements.txt
docs/approach.md          Retrieval and evaluation notes
Dockerfile                Backend image
```

## Run locally

Use Python 3.11 or a compatible Python environment. Install backend dependencies first:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt
```

**Build the indexes before starting the API.** The binary index artifacts (`tfidf_vectorizer.pkl`, `tfidf_matrix.npz`, `embeddings.npy` and `nn_model.pkl`) are generated locally and intentionally not committed.

```bash
python backend/vector_store.py
uvicorn app:app --app-dir backend --reload --port 8000
```

In another terminal, install frontend dependencies and start Streamlit:

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

Open the Streamlit URL shown in the terminal. The frontend connects to `http://localhost:8000` by default; use the `API_URL` environment variable to change it. Model downloads and first-time index creation require an internet connection.

### API

`GET /health`

```json
{"status": "ok"}
```

`POST /recommend`

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"Java and SQL backend developer assessment"}'
```

The response is a JSON array of up to ten assessments. An empty query returns HTTP 400.

### Docker (backend only)

```bash
docker build -t shl-recommender .
docker run --rm -p 8000:8000 shl-recommender
```

The Docker image builds the indexes from the checked-in assessment catalog. Model downloads make the initial image build heavier than a typical REST API.

## Evaluation

```bash
python backend/eval.py
```

This evaluates the top three recommendations against `backend/tests/test_queries.csv` using mean Recall@3 and MAP@3. The recorded results in [the approach notes](docs/approach.md) are **Recall@3 = 0.0500** and **MAP@3 = 0.0741**; they are baseline observations, not production-quality accuracy claims.

## Current limitations

- Search indexes and transformer models must be built/downloaded before serving requests outside Docker.
- URL-to-text extraction depends on the availability and structure of third-party pages.
- The catalog is a snapshot; it is not continuously synchronized with SHL.
- CORS is permissive in the prototype; add authentication, request limits and input restrictions before production deployment.
- No hosted-availability or response-latency guarantees are claimed.

This is an independent demonstration project, not an official SHL product.
