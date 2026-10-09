import json

import check_citations


def sources(count=3):
    values = [
        {"n": 1, "url": "https://arxiv.org/abs/2501.00001", "title": "S1", "source": "arxiv"},
        {
            "n": 2,
            "url": "https://huggingface.co/papers/2501.00002",
            "title": "S2",
            "source": "hf-search",
        },
        {"n": 3, "url": "https://example.test/3", "title": "S3", "source": "web"},
        {
            "n": 4,
            "url": "https://huggingface.co/papers/2501.00004",
            "title": "S4",
            "source": "hf-daily",
        },
    ]
    return values[:count]


def valid_report():
    return """# Survey

Body claim [1, 2] and another claim [2-3].
Inline code `[99]`, a [7](https://ignored.test), and fenced code do not count:
```python
x = "[88]"
```

## References
[1] One. arxiv. https://arxiv.org/abs/2501.00001 (2026-01-01)
[2] Two. hf-search. https://huggingface.co/papers/2501.00002 (2026-01-02)
[3] Three. web. https://example.test/3 (2026-01-03)
"""


def test_valid_grouped_citations_ignore_code_and_links():
    assert check_citations.check(valid_report(), sources()) == []


def test_sources_must_be_nonempty():
    assert check_citations.check("", []) == ["no sources in sources.json"]
    assert check_citations.check("", {}) == ["no sources in sources.json"]


def test_source_number_url_and_duplicate_validation():
    bad = [
        {"n": "1", "url": "ftp://bad", "source": "web"},
        {"n": 2, "url": "https://same.test/x", "source": "web"},
        {"n": 2, "url": "https://same.test/x", "source": "web"},
    ]
    problems = check_citations.check("Body [2]\n\n## References\n", bad)
    assert any("invalid n" in problem for problem in problems)
    assert any("invalid url" in problem for problem in problems)
    assert any("appears more than once in sources.json" in problem for problem in problems)
    assert any("duplicates url" in problem for problem in problems)


def test_source_family_must_be_exact_enum():
    entries = sources()
    entries[0]["source"] = "arXiv primary paper"
    problems = check_citations.check(valid_report(), entries)
    assert any(
        "source [1] has invalid source family: 'arXiv primary paper'" in problem
        for problem in problems
    )


def test_source_family_url_must_match_structured_family():
    entries = sources()
    entries[0]["url"] = "https://example.test/not-arxiv"
    report = valid_report().replace(
        "https://arxiv.org/abs/2501.00001", "https://example.test/not-arxiv"
    )
    problems = check_citations.check(report, entries)
    assert any("is labeled arxiv but URL must start" in problem for problem in problems)

    entries = sources()
    entries[1]["url"] = "https://example.test/not-hf"
    report = valid_report().replace(
        "https://huggingface.co/papers/2501.00002", "https://example.test/not-hf"
    )
    problems = check_citations.check(report, entries)
    assert any("is labeled hf-search but URL must start" in problem for problem in problems)


def test_arxiv_url_found_through_web_remains_web():
    entries = [
        {"n": 1, "url": "https://arxiv.org/abs/2501.00001", "title": "A", "source": "web"},
        {
            "n": 2,
            "url": "https://huggingface.co/papers/2501.00002",
            "title": "B",
            "source": "hf-search",
        },
        {
            "n": 3,
            "url": "https://huggingface.co/papers/2501.00003",
            "title": "C",
            "source": "hf-daily",
        },
    ]
    report = """Claim [1][2][3].

## References
[1] A. web. https://arxiv.org/abs/2501.00001 (2026-01-01)
[2] B. hf-search. https://huggingface.co/papers/2501.00002 (2026-01-02)
[3] C. hf-daily. https://huggingface.co/papers/2501.00003 (2026-01-03)
"""
    assert check_citations.check(report, entries) == []


def test_validator_requires_three_valid_families():
    problems = check_citations.check(
        "Claim [1][2].\n\n## References\n"
        "[1] A. arxiv. https://arxiv.org/abs/2501.00001 (2026-01-01)\n"
        "[2] B. hf-search. https://huggingface.co/papers/2501.00002 (2026-01-02)\n",
        sources(2),
    )
    assert any("uses only 2 valid source families; at least 3" in problem for problem in problems)


def test_corrected_source_requires_re_finalized_reference_line():
    report = valid_report().replace("One. arxiv.", "One. arXiv primary paper.")
    problems = check_citations.check(report, sources())
    assert any(
        "reference [1] source family does not match sources.json" in problem
        and "rerun finalize_citations.py" in problem
        for problem in problems
    )


def test_missing_heading_unknown_and_uncited_sources():
    problems = check_citations.check("Claim [9].", sources(1))
    assert "report is missing ## References" in problems
    assert "[9] cited but missing from sources.json" in problems
    assert "source [1] never cited" in problems


def test_reference_line_must_be_unique_exact_and_single_url():
    report = """Claim [1][2].

## References
[1] Wrong https://wrong.test and https://second.test
[1] Duplicate https://example.test/1
[4] Unknown https://example.test/4
"""
    problems = check_citations.check(report, sources(2))
    assert "reference [1] appears 2 times" in problems
    assert "source [2] has no reference line" in problems
    assert "reference [4] is missing from sources.json" in problems
    assert any("must contain exactly one URL" in problem for problem in problems)


def test_cli_exit_codes(tmp_path, capsys):
    report_path = tmp_path / "report.md"
    sources_path = tmp_path / "sources.json"
    report_path.write_text(valid_report(), encoding="utf-8")
    sources_path.write_text(json.dumps(sources()), encoding="utf-8")
    assert check_citations.main(["check", str(report_path), str(sources_path)]) == 0
    assert "OK: 3 sources, all citations resolve" in capsys.readouterr().out

    report_path.write_text("No references", encoding="utf-8")
    assert check_citations.main(["check", str(report_path), str(sources_path)]) == 1
