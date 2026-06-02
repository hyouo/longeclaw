"""
PubMed search via NCBI E-utilities.
No API key needed for <3 requests/second.
With NCBI_API_KEY env var: 10 requests/second.
"""

import json
import os
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from typing import Optional

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "longevityclaw"
EMAIL = "longevityclaw@localhost"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")

# Rate limit tracking
_last_request_time = 0.0
_MIN_DELAY = 0.35  # 3 req/sec without key


def _rate_limit():
    """Ensure minimum delay between requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_DELAY:
        time.sleep(_MIN_DELAY - elapsed)
    _last_request_time = time.time()


def _fetch_with_retry(url: str, timeout: int = 15, max_retries: int = 3) -> bytes:
    """Fetch URL with retry on 429 rate limit errors."""
    for attempt in range(max_retries):
        _rate_limit()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))  # exponential backoff
                continue
            raise
    return b""


def search_pubmed(query: str, max_results: int = 10) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "tool": TOOL,
        "email": EMAIL,
        "sort": "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    url = f"{BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    data = json.loads(_fetch_with_retry(url, timeout=15))
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch article metadata and abstracts for a list of PMIDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": TOOL,
        "email": EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    url = f"{BASE}/efetch.fcgi?{urllib.parse.urlencode(params)}"
    xml_data = _fetch_with_retry(url, timeout=30)

    root = ET.fromstring(xml_data)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article, ".//PMID")
        title = _text(article, ".//ArticleTitle")

        # Abstract — may have multiple sections
        abstract_parts = []
        for ab in article.findall(".//AbstractText"):
            label = ab.get("Label", "")
            text = "".join(ab.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = _text(author, "LastName")
            initials = _text(author, "Initials")
            if last:
                authors.append(f"{last} {initials}" if initials else last)

        # Journal and date
        journal = _text(article, ".//Journal/Title")
        year = _text(article, ".//PubDate/Year")
        doi = ""
        for eid in article.findall(".//ArticleId"):
            if eid.get("IdType") == "doi":
                doi = eid.text or ""

        articles.append({
            "pmid": pmid,
            "title": title,
            "authors": authors[:5],  # first 5
            "n_authors": len(authors),
            "journal": journal,
            "year": year,
            "doi": doi,
            "abstract": abstract[:1500] if abstract else "",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return articles


def search_and_fetch(query: str, max_results: int = 5) -> dict:
    """Search PubMed and return articles with abstracts."""
    pmids = search_pubmed(query, max_results=max_results)
    if not pmids:
        return {"query": query, "n_results": 0, "articles": []}

    articles = fetch_abstracts(pmids)
    return {
        "query": query,
        "n_results": len(articles),
        "articles": articles,
    }


def _text(el: ET.Element, path: str) -> str:
    node = el.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return ""
