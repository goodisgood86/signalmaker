from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from typing import Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET

import httpx

RSS_URLS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]

SYMBOL_ALIASES: Dict[str, List[str]] = {
    "BTCUSDT": ["bitcoin", "btc", "비트코인"],
    "ETHUSDT": ["ethereum", "eth", "이더리움"],
    "XRPUSDT": ["xrp", "ripple", "리플"],
    "DOGEUSDT": ["doge", "dogecoin", "도지"],
    "SUIUSDT": ["sui", "수이"],
    "SOLUSDT": ["solana", "sol", "솔라나"],
    "CROSSUSDT": ["cross", "크로쓰"],
}

POSITIVE_KEYWORDS: Tuple[str, ...] = (
    "surge",
    "rally",
    "bull",
    "breakout",
    "adoption",
    "approval",
    "partnership",
    "record high",
    "inflow",
    "gain",
    "up",
    "상승",
    "호재",
    "급등",
    "돌파",
    "채택",
)

NEGATIVE_KEYWORDS: Tuple[str, ...] = (
    "drop",
    "selloff",
    "bear",
    "hack",
    "ban",
    "lawsuit",
    "liquidation",
    "exploit",
    "outflow",
    "fraud",
    "down",
    "하락",
    "악재",
    "급락",
    "해킹",
    "규제",
    "소송",
)


@dataclass(frozen=True)
class NewsItem:
    title: str
    link: str
    summary: str
    source: str
    published_ms: int | None
    sentiment: str
    score: int


def _parse_dt_to_ms(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _score_text(text: str) -> Tuple[str, int]:
    s = text.lower()
    pos = sum(1 for k in POSITIVE_KEYWORDS if k in s)
    neg = sum(1 for k in NEGATIVE_KEYWORDS if k in s)
    score = pos - neg
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", 0


def _contains_alias(text: str, alias: str) -> bool:
    alias_norm = alias.strip().lower()
    if not alias_norm:
        return False
    # ASCII 키워드는 단어 경계를 강제해 "sol" -> "solution" 같은 오탐을 막는다.
    if re.fullmatch(r"[a-z0-9]+", alias_norm):
        pat = rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])"
        return re.search(pat, text) is not None
    return alias_norm in text


def _match_symbols(text: str, aliases: Dict[str, List[str]]) -> List[str]:
    s = text.lower()
    out: List[str] = []
    for symbol, keys in aliases.items():
        if any(_contains_alias(s, k) for k in keys):
            out.append(symbol)
    return out


def _parse_rss(xml_text: str, source: str) -> List[Dict[str, str | int | None]]:
    items: List[Dict[str, str | int | None]] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items

    # RSS
    for node in root.findall(".//item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        summary = (node.findtext("description") or "").strip()
        pub = _parse_dt_to_ms((node.findtext("pubDate") or "").strip())
        if title:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "source": source,
                    "published_ms": pub,
                }
            )

    # Atom fallback
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        pub_raw = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
        pub = _parse_dt_to_ms(pub_raw)
        link = ""
        link_node = entry.find("atom:link", ns)
        if link_node is not None:
            link = (link_node.attrib.get("href") or "").strip()
        if title:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "source": source,
                    "published_ms": pub,
                }
            )
    return items


async def fetch_news_sentiment(
    *, symbols: Iterable[str], limit_items: int = 120, timeout_s: float = 6.0, lookback_hours: int = 24
) -> Dict[str, Dict[str, object]]:
    target = {s.upper(): {"positive": 0, "negative": 0, "neutral": 0, "items": []} for s in symbols}
    seen: set[str] = set()
    raw_items: List[Dict[str, str | int | None]] = []
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cutoff_ms = now_ms - max(1, int(lookback_hours)) * 60 * 60 * 1000

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for url in RSS_URLS:
            try:
                r = await client.get(url)
                r.raise_for_status()
                source = "coindesk" if "coindesk" in url else "cointelegraph"
                raw_items.extend(_parse_rss(r.text, source))
            except Exception:
                continue

    raw_items.sort(key=lambda x: int(x.get("published_ms") or 0), reverse=True)

    for item in raw_items:
        title = str(item.get("title") or "").strip()
        link = str(item.get("link") or "").strip()
        summary = str(item.get("summary") or "").strip()
        pub_ms = item.get("published_ms")
        if not isinstance(pub_ms, int):
            continue
        if pub_ms < cutoff_ms:
            continue
        key = f"{title}|{link}"
        if not title or key in seen:
            continue
        seen.add(key)
        text = f"{title} {summary}".lower()
        matched_symbols = _match_symbols(text, SYMBOL_ALIASES)
        if not matched_symbols:
            continue
        senti, score = _score_text(text)
        payload = {
            "title": title,
            "link": link,
            "source": str(item.get("source") or ""),
            "published_ms": item.get("published_ms"),
            "sentiment": senti,
            "score": score,
        }
        for sym in matched_symbols:
            if sym not in target:
                continue
            target[sym][senti] = int(target[sym][senti]) + 1
            items = target[sym]["items"]
            if isinstance(items, list) and len(items) < 6:
                items.append(payload)
        if sum(int(target[s]["positive"]) + int(target[s]["negative"]) + int(target[s]["neutral"]) for s in target) >= limit_items:
            break

    return target
