
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()
BASE_URL = os.getenv("BASE_URL", "https://www.ekaimpact.org")
OUT = Path("data/ekai_full.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

visited = set()
collected = []

def is_internal(url: str) -> bool:
    p = urlparse(url)
    base_host = urlparse(BASE_URL).netloc
    return (p.netloc == "" or p.netloc == base_host)

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator="\n", strip=True)

def scrape_page(url: str):
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None, []
        html = resp.text
        text = clean_text(html)
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            full = urljoin(url, a["href"])
            if is_internal(full):
                links.append(full.split("#")[0])
        return text, links
    except Exception:
        return None, []

def crawl(start_url: str):
    queue = [start_url]
    pbar = tqdm(total=1, desc="Crawling")
    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        pbar.update(1)
        text, links = scrape_page(url)
        if text:
            collected.append(text)
        for link in links:
            if link not in visited and link.startswith(BASE_URL):
                queue.append(link)
        time.sleep(0.4)
    pbar.close()

if __name__ == "__main__":
    print(f"Start scraping: {BASE_URL}")
    crawl(BASE_URL)
    OUT.write_text("\n\n".join(collected), encoding="utf-8")
    print(f"✅ Saved raw site to {OUT} (pages: {len(visited)})")
