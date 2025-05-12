set -e

python data_ingestion.py

python vector_store.py

uvicorn app:app --host 0.0.0.0 --port 8000
