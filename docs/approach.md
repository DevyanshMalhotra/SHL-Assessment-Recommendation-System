# SHL Assessment Recommender – Technical Approach

## Problem Statement
Recruiters and hiring managers struggle to select the right SHL assessment(s) from a large catalog. This tool recommends the top assessments based on a natural language job description or competency requirement.

## Architecture Overview
- **Backend:** FastAPI REST API
- **Frontend:** Streamlit UI
- **Embeddings:** SentenceTransformers (`all-mpnet-base-v2`)
- **Similarity Models:**
  - TF-IDF + Nearest Neighbors (Recall-oriented)
  - Cross-Encoder Reranker (Relevance-oriented)
- **Data Sources:**
  - `assessments.json`: Raw SHL assessment metadata
  - `metadata.json`: Processed metadata for modeling

## Pipeline
1. **Data Ingestion (`data_ingestion.py`):**
   - Loads, cleans, and processes SHL catalog.
   - Generates `assessments.json`.

2. **TF-IDF Model (`vector_store.py`):**
   - Vectorizes all assessments using TF-IDF.
   - Saves the `tfidf_matrix.npz`,`metadata.json`, `tfidf_vectorizer.pkl`, and `embeddings.npy`.

3. **Dense Embeddings:**
   - Uses SentenceTransformer to create dense vector representations of assessment metadata.

4. **FastAPI Backend (`app.py`):**
   - `/health`: Health check
   - `/recommend`: Accepts a job description, returns top SHL assessment matches.

5. **Frontend (`streamlit_app.py`):**
   - Simple web UI to enter job requirements and view recommendations.

6. **Evaluation (`eval.py`):**
   - Computes MAP@3 and Recall@3 from a test CSV file generated via `generate_test_csv.py`.

## Results
- **Mean Recall@3:** 0.0500
- **MAP@3:** 0.0741

