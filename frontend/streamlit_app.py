import os
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.ok and r.json().get("status")=="ok"
    except:
        return False

def fetch_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(" ", strip=True)
    except:
        return ""

st.title("SHL Assessment Recommendation")
st.write("API status:", "✅" if get_health() else "❌")

inp = st.text_area("Enter JD URL, text, or query:")
if st.button("Recommend"):
    if not inp.strip():
        st.warning("Please enter input")
    else:
        payload = inp.strip()
        if payload.lower().startswith(("http://","https://")):
            txt = fetch_text(payload)
            if txt:
                payload = txt
        try:
            r = requests.post(f"{API_URL}/recommend", json={"query":payload}, timeout=10)
            if r.ok:
                df = pd.DataFrame(r.json())
                df["name"] = df.apply(lambda r: f"[{r['name']}]({r['url']})", axis=1)
                st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
            else:
                st.error(f"{r.status_code}: {r.json().get('detail',r.text)}")
        except Exception as e:
            st.error(f"Error: {e}")
