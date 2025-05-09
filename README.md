# SHL Assessment Recommendation System

This project provides SHL assessment recommendations based on a natural language job description using a combination of TF-IDF, dense vector embeddings, and cross-encoder reranking.

---

## 🛠️ Features

- 🔍 Recommend SHL tests from a job description
- 🚀 FastAPI backend with `/health` and `/recommend` endpoints
- 🖥️ Streamlit frontend for easy interaction
- 📊 Evaluation via Recall@3 and MAP@3
- 🧠 Uses SentenceTransformers and TF-IDF
- 🐳 Docker-compatible

---

## 📁 Project Structure

```

SHL Assessment Recommendation System/
├── backend/
│   ├── app.py               # FastAPI server
│   ├── data\_ingestion.py    # Prepares SHL assessment data
│   ├── tests\test\_queries.csv
│   ├── eval.py              # Computes Recall\@3, MAP\@3
│   ├── vector\_store.py      # TF-IDF vector model
│   ├── generate\_test\_csv.py # Test(queries) data generator
│   ├── assessments.json     # Raw SHL catalog
│   ├── metadata.json        # Processed descriptions
│   └── nn\_model.pkl         # Pretrained NearestNeighbors model
|   └── requirements.txt
├── frontend/
│   └── streamlit\_app.py     # Web interface
|   └── requirements.txt
├── docs/
│   └── approach.md          # Project explanation
├── Dockerfile
├── README.md

````

---

## 📡 API Endpoints

### `/health` – `GET`
Check if the backend is running.
```json
{"status": "ok"}
````

### `/recommend` – `POST`

Get recommendations by sending a JSON payload with a job description.

```json
{
  "query": "mid-level Python SQL JavaScript <60min"
}
```

**Returns:** Top 10 SHL assessments with metadata and URLs.

---

## 🚀 Run Locally

```bash
# Backend
cd backend
python app.py

# Frontend
cd ../frontend
streamlit run streamlit_app.py
```

---

## 🧪 Evaluate

```bash
cd backend
python eval.py
```

---