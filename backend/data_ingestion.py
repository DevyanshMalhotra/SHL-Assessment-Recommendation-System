import re, time, json, logging
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

BASE_CATALOG = "https://www.shl.com/solutions/products/product-catalog/"
PAGE_URL     = BASE_CATALOG + "?type={type}&start={start}"
HEADERS      = {"User-Agent": "Mozilla/5.0"}
OUTPUT_FILE  = "assessments.json"
PER_PAGE     = 12

TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger()

def make_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.2, 
                    status_forcelist=[500,502,503,504], 
                    allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    return s

def fetch_html(session, url):
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    return resp.text

def parse_catalog(html):
    soup = BeautifulSoup(html, "html.parser")
    # pick only product links
    anchors = soup.find_all("a", href=re.compile(r"^/products/product-catalog/view/"))
    out = []
    for a in anchors:
        name     = a.get_text(strip=True)
        href     = a["href"]
        url      = href if href.startswith("http") else "https://www.shl.com" + href
        codes    = [
            c for c in (a.next_sibling or "").strip().split()
            if c in TYPE_MAP
        ]
        out.append({"name": name, "url": url, "codes": codes})
    return out

def parse_detail(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")

    meta = soup.find("meta", {"name":"description"})
    if meta and meta.get("content"):
        description = meta["content"].strip()
    else:
        hdr = soup.find(["h1","h2"])
        p   = hdr.find_next_sibling("p") if hdr else None
        description = p.get_text(strip=True) if p else ""

    m = re.search(
        r"(?:Completion Time|Approximate Completion Time).*?=\s*(\d+)|(?:Completion Time|Duration).*?(\d+)\s*minutes",
        text, re.IGNORECASE
    )
    if m:
        dur = m.group(1) or m.group(2)
        duration = f"{dur} minutes"
    else:
        duration = ""

    remote = "Yes" if re.search(r"\b(remote|online|remotely)\b", text, re.IGNORECASE) else "No"

    adaptive = "Yes" if re.search(r"\b(adaptive|IRT|item response theory)\b", text, re.IGNORECASE) else "No"
   
    m3 = re.search(r"Test Type:\s*([A-Z](?:\s+[A-Z])*)", text)
    codes = m3.group(1).split() if m3 else []
    test_types = [TYPE_MAP[c] for c in codes if c in TYPE_MAP]

    return {
        "description":    description,
        "duration":       duration,
        "remote_testing": remote,
        "adaptive":       adaptive,
        "test_types":     test_types
    }

def scrape_all():
    sess = make_session()
    raw = []

    for t in (2,1):
        logger.info(f"Scraping catalog type={t}")
        start = 0
        while True:
            url   = PAGE_URL.format(type=t, start=start)
            html  = fetch_html(sess, url)
            batch = parse_catalog(html)
            if not batch:
                break
            raw.extend(batch)
            logger.info(f" → page start={start}: {len(batch)} items")
            start += PER_PAGE
            time.sleep(0.2)

    seen, results = set(), []
    for item in raw:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        try:
            detail_html = fetch_html(sess, item["url"])
            info        = parse_detail(detail_html)
        except Exception as e:
            logger.warning(f"Failed detail scrape for {item['url']}: {e}")
            info = {
                "description":"", "duration":"",
                "remote_testing":"No", "adaptive":"No",
                "test_types":[ TYPE_MAP[c] for c in item["codes"] ]
            }
        results.append({
            "name":           item["name"],
            "url":            item["url"],
            **info,
            "type":           info["test_types"][0] if info["test_types"] else ""
        })
        time.sleep(0.1)

    logger.info(f"Total unique assessments: {len(results)}")
    return results

if __name__=="__main__":
    data = scrape_all()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(data)} items to {OUTPUT_FILE}")
