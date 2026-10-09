"""Reliable host-side source tools used by the research agents.

Every public tool returns a string and keeps credentials on the host. Empty
results are reported as ``NO RESULTS`` and exhausted failures as ``ERROR: ...``.
"""

import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import httpx
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

ARXIV_URL = "https://export.arxiv.org/api/query"
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_ARXIV_INTERVAL = 3.0
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_CALL = 0.0
_DEFAULT_TIMEOUT = 30.0


class RetryableError(Exception):
    """Ask :func:`with_retry` to retry, optionally after a server delay."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_seconds(response):
    value = response.headers.get("Retry-After") if response is not None else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call ``fn`` with bounded exponential backoff for transient failures."""
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if base < 0 or cap < 0:
        raise ValueError("base and cap must be non-negative")

    for attempt in range(attempts):
        retry_after = None
        try:
            return fn()
        except RetryableError as exc:
            retry_after = exc.retry_after
            error = exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRYABLE_STATUS:
                raise
            retry_after = _retry_after_seconds(exc.response)
            error = exc
        except httpx.TransportError as exc:
            error = exc

        if attempt == attempts - 1:
            raise error
        if retry_after is not None:
            delay = min(cap, max(0.0, float(retry_after)))
        else:
            exponential = base * (2**attempt)
            delay = min(cap, exponential + random.uniform(0.0, max(base, 0.001)))
        time.sleep(delay)


def _clean_space(value, limit=None):
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip() if limit is not None else text


def _clamp_int(value, low, high, default):
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    return max(low, min(high, number))


def _error(exc, secret=None):
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        message = message.replace(secret, "[REDACTED]")
        message = message.replace(quote(secret, safe=""), "[REDACTED]")
    message = re.sub(r"([?&]exaApiKey=)[^&\s]+", r"\1[REDACTED]", message, flags=re.I)
    return f"ERROR: {message}"


def _http_get(url, *, params):
    response = httpx.get(url, params=params, timeout=_DEFAULT_TIMEOUT, follow_redirects=True)
    response.raise_for_status()
    return response


def _arxiv_request(params):
    global _ARXIV_LAST_CALL
    with _ARXIV_LOCK:
        wait = _ARXIV_INTERVAL - (time.monotonic() - _ARXIV_LAST_CALL)
        if wait > 0:
            time.sleep(wait)
        _ARXIV_LAST_CALL = time.monotonic()
        return _http_get(ARXIV_URL, params=params)


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv by safe keywords, newest first; return JSON records with id, URL, date, title, and abstract."""
    try:
        terms = re.findall(r"[^\W_]+(?:-[^\W_]+)*", str(query or ""), flags=re.UNICODE)
        terms = [term for term in terms if term.upper() not in {"AND", "OR", "NOT"}]
        if not terms:
            return "NO RESULTS"
        params = {
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": _clamp_int(max_results, 1, 30, 10),
            "start": 0,
        }
        response = with_retry(lambda: _arxiv_request(params), attempts=7, base=2.0, cap=60.0)
        root = ET.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", namespace):
            raw_id = _clean_space(entry.findtext("atom:id", default="", namespaces=namespace))
            paper_id = raw_id.split("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id.rsplit("/", 1)[-1]
            paper_id = re.sub(r"v\d+$", "", paper_id)
            if not paper_id:
                continue
            records.append(
                {
                    "id": paper_id,
                    "url": f"https://arxiv.org/abs/{paper_id}",
                    "published": _clean_space(
                        entry.findtext("atom:published", default="", namespaces=namespace)
                    )[:10],
                    "title": _clean_space(entry.findtext("atom:title", default="", namespaces=namespace)),
                    "summary": _clean_space(
                        entry.findtext("atom:summary", default="", namespaces=namespace), 600
                    ),
                }
            )
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # tools must never leak exceptions to the agent
        return _error(exc)


def _payload_items(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("papers", "items", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def _pick(paper, item, *names, default=None):
    for name in names:
        value = paper.get(name) if isinstance(paper, dict) else None
        if value not in (None, ""):
            return value
        value = item.get(name) if isinstance(item, dict) else None
        if value not in (None, ""):
            return value
    return default


def _number(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _github_value(value):
    if isinstance(value, dict):
        return value.get("url") or value.get("repo") or value.get("name") or ""
    return str(value or "")


def _hf_records(payload, *, prefer_ai_summary=False):
    records = []
    for item in _payload_items(payload):
        if not isinstance(item, dict):
            continue
        paper = item.get("paper")
        if not isinstance(paper, dict):
            continue
        paper_id = _clean_space(paper.get("id"))
        if not paper_id:
            continue
        summary_names = ("ai_summary", "summary") if prefer_ai_summary else ("summary",)
        records.append(
            {
                "id": paper_id,
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": _clean_space(
                    _pick(paper, item, "publishedAt", "published", "date", default="")
                )[:10],
                "title": _clean_space(_pick(paper, item, "title", default="")),
                "summary": _clean_space(_pick(paper, item, *summary_names, default=""), 600),
                "upvotes": _number(_pick(paper, item, "upvotes", default=0)),
                "github": _github_value(_pick(paper, item, "githubRepo", "github", default="")),
                "stars": _number(_pick(paper, item, "githubStars", "stars", default=0)),
            }
        )
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Get Hugging Face Daily Papers, optionally filter title/summary by keyword, and sort by upvotes."""
    try:
        params = {"limit": _clamp_int(limit, 1, 100, 30)}
        if str(date or "").strip():
            params["date"] = str(date).strip()
        response = with_retry(lambda: _http_get(HF_DAILY_URL, params=params))
        records = _hf_records(response.json())
        needle = _clean_space(keyword).casefold()
        if needle:
            records = [
                record
                for record in records
                if needle in f"{record['title']} {record['summary']}".casefold()
            ]
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic; return normalized JSON records, preferring the AI summary when available."""
    try:
        query = _clean_space(query)
        if not query:
            return "NO RESULTS"
        params = {"q": query, "limit": _clamp_int(limit, 1, 50, 10)}
        response = with_retry(lambda: _http_get(HF_SEARCH_URL, params=params))
        records = _hf_records(response.json(), prefer_ai_summary=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


def _decode_mcp_response(response):
    text = response.text
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" not in content_type:
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError):
            pass
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            continue
    raise ValueError("Exa returned neither JSON nor a valid SSE data event")


def _contains_rate_limit(value, parent_key=""):
    key = parent_key.casefold().replace("_", "").replace("-", "")
    if isinstance(value, dict):
        return any(
            _contains_rate_limit(child, f"{parent_key}.{name}") for name, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_rate_limit(child, parent_key) for child in value)
    if "ratelimit" in key:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value == 0 and "remaining" in key
        if isinstance(value, str):
            return value.casefold().strip() not in {"", "false", "ok", "allowed", "none"}
    if isinstance(value, str):
        lowered = value.casefold()
        return any(
            marker in lowered
            for marker in ("rate limit", "rate-limit", "too many requests", "quota exceeded")
        )
    return False


def _content_is_rate_limit_error(content):
    parts = [
        part.get("text", "")
        for part in content or []
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    text = " ".join(parts).strip()
    if not text or len(text) > 1000:
        return False
    lowered = text.casefold()
    has_limit = any(marker in lowered for marker in ("rate limit", "rate-limit", "too many requests"))
    has_failure = any(marker in lowered for marker in ("exceeded", "reached", "try again", "retry", "quota"))
    return has_limit and has_failure


def _mcp_call(tool_name, arguments):
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    endpoint = EXA_URL
    if api_key:
        endpoint = f"{EXA_URL}?exaApiKey={quote(api_key, safe='')}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    def request():
        response = httpx.post(
            endpoint,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        message = _decode_mcp_response(response)
        if not isinstance(message, dict):
            raise ValueError("invalid JSON-RPC response")
        if "error" in message:
            error = message["error"]
            if _contains_rate_limit(error):
                raise RetryableError(f"Exa rate limited the request: {error}")
            raise RuntimeError(f"Exa JSON-RPC error: {error}")
        result = message.get("result")
        if not isinstance(result, dict):
            raise ValueError("Exa JSON-RPC response has no result object")
        if _contains_rate_limit(result.get("_meta", {})) or _content_is_rate_limit_error(
            result.get("content", [])
        ):
            raise RetryableError("Exa rate limited the request")
        parts = [
            part.get("text", "")
            for part in result.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        ]
        return "\n\n".join(parts).strip()

    return with_retry(request, attempts=6, base=2.0, cap=60.0), api_key


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web with Exa for an objective; return MCP text containing source URLs."""
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        query = _clean_space(query)
        if not query:
            return "NO RESULTS"
        objective = _clean_space(objective) or f"Find reliable sources that answer: {query}"
        text, _ = _mcp_call(
            "web_search_exa",
            {
                "query": query,
                "objective": objective,
                "numResults": _clamp_int(num_results, 1, 10, 5),
            },
        )
        return text or "NO RESULTS"
    except Exception as exc:
        return _error(exc, api_key)


@tool
def web_fetch(url: str) -> str:
    """Fetch one web page through Exa as markdown; return at most 12,000 characters."""
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        url = str(url or "").strip()
        if not url:
            return "NO RESULTS"
        text, _ = _mcp_call("web_fetch_exa", {"urls": [url]})
        return text[:12000].rstrip() if text else "NO RESULTS"
    except Exception as exc:
        return _error(exc, api_key)


SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
