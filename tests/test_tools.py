import json
from types import SimpleNamespace

import httpx
import pytest

import tools


class JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_with_retry_exponential_backoff(monkeypatch):
    attempts = {"count": 0}
    sleeps = []
    monkeypatch.setattr(tools.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise tools.RetryableError("later")
        return "ok"

    assert tools.with_retry(flaky, attempts=4, base=1, cap=10) == "ok"
    assert sleeps == [1, 2]


def test_with_retry_honors_retry_after_and_no_final_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(429, headers={"Retry-After": "7"}, request=request)

    def fail():
        response.raise_for_status()

    with pytest.raises(httpx.HTTPStatusError):
        tools.with_retry(fail, attempts=2, cap=5)
    assert sleeps == [5]


def test_with_retry_does_not_retry_programming_error(monkeypatch):
    monkeypatch.setattr(tools.time, "sleep", lambda _delay: pytest.fail("unexpected sleep"))
    with pytest.raises(ValueError):
        tools.with_retry(lambda: (_ for _ in ()).throw(ValueError("bad")))


def test_arxiv_parsing_cleaning_and_empty_query(monkeypatch):
    xml = b"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/2501.00001v3</id><published>2025-01-02T00:00:00Z</published>
      <title>  A\n paper </title><summary> Some\n result </summary>
    </entry></feed>"""
    captured = {}

    def fake_request(params):
        captured.update(params)
        return SimpleNamespace(content=xml)

    monkeypatch.setattr(tools, "_arxiv_request", fake_request)
    records = json.loads(tools.arxiv_search.invoke({"query": 'world:"model" AND test', "max_results": 99}))
    assert captured["search_query"] == "all:world AND all:model AND all:test"
    assert captured["max_results"] == 30
    assert records == [
        {
            "id": "2501.00001",
            "url": "https://arxiv.org/abs/2501.00001",
            "published": "2025-01-02",
            "title": "A paper",
            "summary": "Some result",
        }
    ]
    assert tools.arxiv_search.invoke({"query": ':: ""'}) == "NO RESULTS"


def test_hugging_face_daily_and_search_parsing(monkeypatch):
    payload = [
        {
            "paper": {
                "id": "2501.1",
                "title": "World Model",
                "summary": "base summary",
                "ai_summary": "AI summary",
                "upvotes": 4,
                "githubRepo": "org/repo",
                "githubStars": 12,
                "publishedAt": "2025-01-03T10:00:00Z",
            }
        },
        {"paper": {"title": "missing id"}},
    ]
    monkeypatch.setattr(tools, "_http_get", lambda *_args, **_kwargs: JsonResponse(payload))
    daily = json.loads(tools.hf_daily_papers.invoke({"keyword": "world"}))
    searched = json.loads(tools.hf_search_papers.invoke({"query": "world"}))
    assert len(daily) == len(searched) == 1
    assert daily[0]["summary"] == "base summary"
    assert searched[0]["summary"] == "AI summary"
    assert searched[0]["url"] == "https://huggingface.co/papers/2501.1"


def test_network_failure_becomes_error(monkeypatch):
    monkeypatch.setattr(tools.time, "sleep", lambda _delay: None)

    def fail(*_args, **_kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(tools, "_http_get", fail)
    result = tools.hf_search_papers.invoke({"query": "world"})
    assert result.startswith("ERROR: ConnectError: offline")


def _response(payload, content_type="application/json"):
    request = httpx.Request("POST", tools.EXA_URL)
    if content_type == "text/event-stream":
        content = f"event: message\ndata: {json.dumps(payload)}\n\n"
        return httpx.Response(200, text=content, headers={"content-type": content_type}, request=request)
    return httpx.Response(200, json=payload, headers={"content-type": content_type}, request=request)


def test_exa_sse_parsing_and_arguments(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return _response(
            {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "found"}]}},
            "text/event-stream",
        )

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(tools.httpx, "post", post)
    assert tools.web_search.invoke({"query": "topic", "num_results": 2}) == "found"
    assert captured["json"]["params"]["name"] == "web_search_exa"
    assert captured["json"]["params"]["arguments"]["numResults"] == 2
    assert captured["json"]["params"]["arguments"]["objective"]


def test_exa_rate_limit_in_http_200_retries(monkeypatch):
    responses = iter(
        [
            _response(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "_meta": {"rateLimited": True},
                        "content": [{"type": "text", "text": "Rate limit reached"}],
                    },
                }
            ),
            _response(
                {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "ok"}]}}
            ),
        ]
    )
    monkeypatch.setattr(tools.httpx, "post", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(tools.time, "sleep", lambda _delay: None)
    assert tools.web_fetch.invoke({"url": "https://example.test"}) == "ok"


def test_exa_jsonrpc_error_and_secret_redaction(monkeypatch):
    secret = "secret-key-value-123"
    monkeypatch.setenv("EXA_API_KEY", secret)
    monkeypatch.setattr(
        tools.httpx,
        "post",
        lambda *_args, **_kwargs: _response(
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": f"bad {secret}"}}
        ),
    )
    result = tools.web_search.invoke({"query": "topic"})
    assert result.startswith("ERROR: RuntimeError: Exa JSON-RPC error")
    assert secret not in result
    assert "[REDACTED]" in result


def test_web_fetch_truncates(monkeypatch):
    monkeypatch.setattr(tools, "_mcp_call", lambda *_args, **_kwargs: ("x" * 13000, ""))
    assert len(tools.web_fetch.invoke({"url": "https://example.test"})) == 12000

