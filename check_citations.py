"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys
from collections import Counter
from urllib.parse import urlsplit

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"
SOURCE_FAMILIES = frozenset({"arxiv", "hf-daily", "hf-search", "web"})


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
          source must be exactly arxiv/hf-daily/hf-search/web
          arxiv and hf-* URLs must match their structured source family
      at least three valid source families must remain
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          its displayed source family must equal sources.json (so corrections require re-finalizing)
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    valid_sources = {}
    source_numbers = []
    seen_urls = {}
    valid_families = set()
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} is not an object")
            continue

        number = source.get("n")
        if isinstance(number, bool) or not isinstance(number, int):
            problems.append(f"source entry {index} has invalid n: {number!r}")
        else:
            source_numbers.append(number)
            if number in valid_sources:
                problems.append(f"source number [{number}] appears more than once in sources.json")
            else:
                valid_sources[number] = source

        url = source.get("url")
        if not _valid_url(url):
            problems.append(f"source [{number}] has invalid url: {url!r}")
        elif url in seen_urls:
            problems.append(
                f"source [{number}] duplicates url from source [{seen_urls[url]}]: {url}"
            )
        else:
            seen_urls[url] = number

        family = source.get("source")
        if family not in SOURCE_FAMILIES:
            problems.append(
                f"source [{number}] has invalid source family: {family!r}; "
                f"expected one of {sorted(SOURCE_FAMILIES)}"
            )
        else:
            valid_families.add(family)
            if (
                family == "arxiv"
                and isinstance(url, str)
                and not url.startswith("https://arxiv.org/abs/")
            ):
                problems.append(
                    f"source [{number}] is labeled arxiv but URL must start with "
                    "https://arxiv.org/abs/"
                )
            if (
                family in {"hf-daily", "hf-search"}
                and isinstance(url, str)
                and not url.startswith("https://huggingface.co/papers/")
            ):
                problems.append(
                    f"source [{number}] is labeled {family} but URL must start with "
                    "https://huggingface.co/papers/"
                )

    if len(valid_families) < 3:
        problems.append(
            f"sources.json uses only {len(valid_families)} valid source families; "
            f"at least 3 of {sorted(SOURCE_FAMILIES)} are required"
        )

    headings = list(re.finditer(r"(?m)^## References\s*$", report_text or ""))
    if not headings:
        problems.append("report is missing ## References")
        body = report_text or ""
        references = ""
    else:
        if len(headings) > 1:
            problems.append("report contains more than one ## References heading")
        heading = headings[0]
        body = (report_text or "")[: heading.start()]
        references = (report_text or "")[heading.end() :]

    cited = set()
    for match in _citation_groups(_remove_code_and_links(body)):
        cited.update(_expand_group(match.group(1)))

    source_number_set = set(source_numbers)
    for number in sorted(cited - source_number_set):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(source_number_set - cited):
        problems.append(f"source [{number}] never cited")

    reference_lines = []
    for line_no, line in enumerate(references.splitlines(), 1):
        match = re.match(r"^\s*\[(\d+)\]\s+(.+?)\s*$", line)
        if match:
            reference_lines.append((int(match.group(1)), match.group(2), line_no))

    counts = Counter(number for number, _, _ in reference_lines)
    for number in sorted(source_number_set):
        count = counts[number]
        if count == 0:
            problems.append(f"source [{number}] has no reference line")
        elif count > 1:
            problems.append(f"reference [{number}] appears {count} times")
    for number in sorted(set(counts) - source_number_set):
        problems.append(f"reference [{number}] is missing from sources.json")

    for number, text, line_no in reference_lines:
        urls = re.findall(r"https?://[^\s<>]+", text)
        urls = [url.rstrip(".,;:!?") for url in urls]
        if len(urls) != 1:
            problems.append(
                f"reference [{number}] on line {line_no} must contain exactly one URL (found {len(urls)})"
            )
            continue
        source = valid_sources.get(number)
        if source is not None and urls[0] != source.get("url"):
            problems.append(
                f"reference [{number}] URL does not match sources.json: {urls[0]}"
            )
        if source is not None and source.get("source") in SOURCE_FAMILIES:
            family = source["source"]
            prefix = text.split(urls[0], 1)[0]
            if not re.search(rf"\.\s+{re.escape(family)}\.\s*$", prefix):
                problems.append(
                    f"reference [{number}] source family does not match sources.json: "
                    f"expected {family!r}; rerun finalize_citations.py"
                )

    return problems


def _valid_url(value):
    if not isinstance(value, str) or any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _remove_code_and_links(text):
    """Remove fenced/inline code and Markdown links before scanning citations."""
    text = re.sub(r"(?ms)^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", " ", text)
    text = re.sub(r"(`+)[^\n]*?\1", " ", text)
    text = re.sub(r"!?\[[^\]\n]*\]\([^\)\n]*\)", " ", text)

    def remove_reference_link(match):
        # ``[1][2]`` is the finalizer's normal adjacent-citation form, while
        # ``[label][target]`` is a Markdown reference link.
        if not match.group(1) and match.group(2).isdigit() and match.group(3).isdigit():
            return match.group(0)
        return " "

    text = re.sub(r"(!?)\[([^\]\n]*)\]\[([^\]\n]*)\]", remove_reference_link, text)
    return text


def _citation_groups(text):
    return re.finditer(r"\[((?:\d+\s*(?:[-–]\s*\d+)?)(?:\s*,\s*\d+\s*(?:[-–]\s*\d+)?)*)\]", text)


def _expand_group(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
        if not span:
            numbers.append(int(part))
            continue
        start, end = int(span.group(1)), int(span.group(2))
        if start <= end and end - start <= 200:
            numbers.extend(range(start, end + 1))
        else:
            numbers.extend((start, end))
    return numbers


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
