"""Literature Retriever: search the arXiv API and return titles and abstracts."""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _search_sync(query: str, top_k: int, timeout: int, joiner: str) -> List[Dict[str, str]]:
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", query) if len(w) > 1][:6]
    if not words:
        return []
    search = f" {joiner} ".join(f"all:{w}" for w in words)
    resp = requests.get(ARXIV_API, params={"search_query": search, "start": 0,
                                           "max_results": top_k, "sortBy": "relevance"},
                        timeout=timeout)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    papers = []
    for entry in root.findall("a:entry", _NS):
        papers.append({
            "id": _clean(entry.findtext("a:id", "", _NS)).rsplit("/abs/", 1)[-1],
            "title": _clean(entry.findtext("a:title", "", _NS)),
            "abstract": _clean(entry.findtext("a:summary", "", _NS)),
            "published": _clean(entry.findtext("a:published", "", _NS))[:10],
        })
    return papers


async def search_arxiv(query: str, top_k: int = 10, timeout: int = 30) -> List[Dict[str, str]]:
    """Return up to ``top_k`` papers; an empty list if arXiv cannot be reached."""
    for joiner in ("AND", "OR"):
        for attempt in range(3):
            try:
                papers = await asyncio.to_thread(_search_sync, query, top_k, timeout, joiner)
                break
            except Exception as exc:
                logger.warning("arXiv search failed (%s), attempt %d/3", exc, attempt + 1)
                await asyncio.sleep(3 * (attempt + 1))
        else:
            return []
        if papers:
            return papers
    return []
