"""Run one bounded deep-research job and persist its validated artifacts.

Usage: ``python research.py "survey about world model"``
"""

import json
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from agents import (
    FINALIZER_PATH,
    LEAD_MODEL_CALL_LIMIT,
    LEAD_TOOL_CALL_LIMIT,
    REPORT_PATH,
    SOURCES_PATH,
    SUBAGENT_MODEL_CALL_LIMIT,
    SUBAGENT_TOOL_CALL_LIMIT,
    VALIDATOR_PATH,
    WORKDIR,
    build_lead_agent,
)
from check_citations import check as check_citations
from langchain_core.callbacks import BaseCallbackHandler
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
RECURSION_LIMIT = 1000
SANDBOX_REPAIR_LIMIT = 2
SOURCE_FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}
DEBUG_ENV = "DAY23_DEBUG"
_DEBUG_EVENT_LIMIT = 400
_SECRET_ENV_NAMES = (
    "LAB_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "DAYTONA_API_KEY",
    "EXA_API_KEY",
)
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def slugify(topic):
    """Convert user text to a safe, lowercase file stem of at most 60 chars."""
    slug = re.sub(r"[^\w]+", "-", str(topic or "").strip().lower(), flags=re.UNICODE)
    slug = slug.strip("-_")[:60].rstrip("-_")
    if not slug:
        return "topic"
    if slug.casefold() in _WINDOWS_RESERVED:
        slug = f"topic-{slug}"
    return slug


def build_prompt(topic):
    """Build the sole user message sent to the lead agent."""
    clean_topic = " ".join(str(topic or "").split())
    return (
        f'Research the topic "{clean_topic}" and produce a rigorous English survey. '
        "Follow the complete delegated workflow in your system instructions: use at least three independent "
        "researcher tasks, ground every claim in retrieved notes, preserve at least three valid source families after "
        "citation finalization, spot-check claims, and finish only after the citation validator prints OK. Your chat "
        f"answer is not the deliverable: you MUST use sandbox file tools to create non-empty `{REPORT_PATH}` and "
        f"`{SOURCES_PATH}`. Before your final response, read both files back and confirm they exist."
    )


def _message_value(message, name, default=None):
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _debug_enabled():
    return (os.getenv(DEBUG_ENV) or "").strip().casefold() in {"1", "true", "yes", "on"}


def _redact(value):
    text = str(value)
    for name in _SECRET_ENV_NAMES:
        secret = os.getenv(name) or ""
        if len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"([?&]exaApiKey=)[^&\s]+", r"\1[REDACTED]", text, flags=re.I)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._~+\-/=]+", r"\1[REDACTED]", text, flags=re.I)
    return text


def _preview(value, limit=600):
    text = _redact(value).replace("\x00", "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _tool_input_preview(input_str, inputs=None):
    value = inputs if isinstance(inputs, dict) and inputs else input_str
    if not isinstance(value, dict):
        return _preview(value, 400)
    safe = {}
    for key, item in value.items():
        lowered = str(key).casefold()
        if any(marker in lowered for marker in ("key", "secret", "token", "password")):
            safe[key] = "[REDACTED]"
        elif lowered in {"content", "prompt", "description"}:
            safe[key] = _preview(item, 240)
        else:
            safe[key] = item
    try:
        return _preview(json.dumps(safe, ensure_ascii=False, default=str), 500)
    except (TypeError, ValueError):
        return _preview(safe, 500)


def _debug_print(message):
    print(f"[day23-debug] {_redact(message)}", file=sys.stderr, flush=True)


class Day23DebugHandler(BaseCallbackHandler):
    """Bounded, redacted callback trace for lead and nested subagent activity."""

    run_inline = True

    def __init__(self, event_limit=_DEBUG_EVENT_LIMIT):
        self.event_limit = event_limit
        self.events = 0
        self.model_calls = 0
        self.tool_calls = Counter()
        self.tool_errors = Counter()
        self._tool_runs = {}
        self._lock = threading.Lock()

    def _emit(self, text):
        with self._lock:
            self.events += 1
            if self.events <= self.event_limit:
                _debug_print(text)
            elif self.events == self.event_limit + 1:
                _debug_print(f"event log capped at {self.event_limit}; further event details suppressed")

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs
    ):
        with self._lock:
            self.model_calls += 1
            number = self.model_calls
        self._emit(f"model call #{number} started (run={run_id}, parent={parent_run_id})")

    def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        inputs=None,
        **kwargs,
    ):
        serialized = serialized if isinstance(serialized, dict) else {}
        name = str(serialized.get("name") or (metadata or {}).get("tool_name") or "unknown-tool")
        with self._lock:
            self.tool_calls[name] += 1
            self._tool_runs[run_id] = name
        self._emit(
            f"tool start name={name} run={run_id} parent={parent_run_id} "
            f"input={_tool_input_preview(input_str, inputs)}"
        )

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        with self._lock:
            name = self._tool_runs.pop(run_id, "unknown-tool")
        self._emit(
            f"tool end name={name} run={run_id} parent={parent_run_id} output={_preview(output)}"
        )

    def on_tool_error(self, error, *, run_id, parent_run_id=None, tags=None, **kwargs):
        with self._lock:
            name = self._tool_runs.pop(run_id, "unknown-tool")
            self.tool_errors[name] += 1
        self._emit(
            f"tool ERROR name={name} run={run_id} parent={parent_run_id}: "
            f"{type(error).__name__}: {_preview(error)}"
        )

    def on_llm_error(self, error, *, run_id, parent_run_id=None, tags=None, **kwargs):
        self._emit(
            f"model ERROR run={run_id} parent={parent_run_id}: {type(error).__name__}: {_preview(error)}"
        )

    def report(self):
        _debug_print(
            "callback totals: "
            + json.dumps(
                {
                    "model_calls": self.model_calls,
                    "tool_calls": dict(sorted(self.tool_calls.items())),
                    "tool_errors": dict(sorted(self.tool_errors.items())),
                },
                ensure_ascii=False,
            )
        )


def _debug_messages(messages, elapsed, model_name):
    run_summary = summarize(messages, elapsed, model_name)
    _debug_print(f"lead returned {len(messages)} messages; lead summary={json.dumps(run_summary)}")
    for index, message in enumerate((messages or [])[-16:], max(0, len(messages or []) - 16)):
        kind = _message_value(message, "type", type(message).__name__)
        content = _message_value(message, "content", "")
        status = _message_value(message, "status", None)
        name = _message_value(message, "name", None)
        metadata = _message_value(message, "response_metadata", {}) or {}
        finish_reason = metadata.get("finish_reason") if isinstance(metadata, dict) else None
        calls = []
        for call in _message_value(message, "tool_calls", []) or []:
            if isinstance(call, dict):
                calls.append(
                    {
                        "name": call.get("name"),
                        "args": _tool_input_preview("", call.get("args") or {}),
                    }
                )
            else:
                calls.append({"name": getattr(call, "name", None)})
        _debug_print(
            f"message[{index}] type={kind} name={name} status={status} finish_reason={finish_reason} "
            f"tool_calls={json.dumps(calls, ensure_ascii=False)} content={_preview(content)}"
        )


def _debug_sandbox(backend):
    commands = [
        f"ls -la {WORKDIR}",
        f"ls -la {WORKDIR}/research",
        f"ls -la {WORKDIR}/research/notes",
        f"ls -la {WORKDIR}/report",
        (
            f"for p in {REPORT_PATH} {SOURCES_PATH}; do "
            'if [ -f "$p" ]; then printf "FILE %s bytes=" "$p"; wc -c < "$p"; '
            'else printf "MISSING %s\\n" "$p"; fi; done'
        ),
    ]
    for command in commands:
        try:
            response = backend.execute(command)
            _debug_print(
                f"sandbox command={command!r} exit_code={getattr(response, 'exit_code', None)} "
                f"output={_preview(getattr(response, 'output', ''), 4000)}"
            )
        except Exception as exc:
            _debug_print(
                f"sandbox diagnostic failed command={command!r}: {type(exc).__name__}: {_preview(exc)}"
            )


def summarize(messages, elapsed, model_name):
    """Summarize lead-level tool calls and token usage from returned messages."""
    calls = Counter()
    input_tokens = 0
    output_tokens = 0
    for message in messages or []:
        for call in _message_value(message, "tool_calls", []) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                calls[str(name)] += 1
        usage = _message_value(message, "usage_metadata", {}) or {}
        if isinstance(usage, dict):
            input_tokens += _integer(usage.get("input_tokens"))
            output_tokens += _integer(usage.get("output_tokens"))
        else:
            input_tokens += _integer(getattr(usage, "input_tokens", 0))
            output_tokens += _integer(getattr(usage, "output_tokens", 0))
    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": calls.get("task", 0),
        "tool_calls": dict(sorted(calls.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def _downloaded_bytes(files, path, label):
    content = files.get(path)
    if content is None:
        raise RuntimeError(
            f"sandbox did not produce {label}: {path}; rerun with {DEBUG_ENV}=1 for agent and sandbox diagnostics"
        )
    if isinstance(content, str):
        content = content.encode("utf-8")
    if not isinstance(content, (bytes, bytearray)):
        raise RuntimeError(f"sandbox returned invalid {label} content")
    return bytes(content)


def _atomic_write_set(payloads):
    """Write all temp files first, then atomically replace their destinations."""
    temporary = []
    try:
        for destination, content in payloads:
            temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            temp.write_bytes(content)
            temporary.append((temp, destination))
        for temp, destination in temporary:
            os.replace(temp, destination)
    finally:
        for temp, _ in temporary:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download, validate, and save the report, sources, and measured metadata."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = _downloaded_bytes(files, REPORT_PATH, "report")
    sources_bytes = _downloaded_bytes(files, SOURCES_PATH, "sources.json")
    try:
        report_text = report_bytes.decode("utf-8")
        sources_text = sources_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"sandbox output is not valid UTF-8: {exc}") from exc
    if not report_text.strip():
        raise RuntimeError("sandbox report is empty")
    try:
        sources = json.loads(sources_text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources or not all(isinstance(item, dict) for item in sources):
        raise RuntimeError("sources.json must be a non-empty JSON list of objects")
    citation_problems = check_citations(report_text, sources)
    if citation_problems:
        raise RuntimeError("downloaded citation artifacts are invalid: " + "; ".join(citation_problems[:5]))

    families = sorted({item.get("source") for item in sources if item.get("source") in SOURCE_FAMILIES})
    for item in sources:
        family = item.get("source")
        url = item.get("url", "")
        if family not in SOURCE_FAMILIES:
            raise RuntimeError(f"source [{item.get('n')}] has invalid source family: {family!r}")
        if family == "arxiv" and not url.startswith("https://arxiv.org/abs/"):
            raise RuntimeError(f"source [{item.get('n')}] is labeled arxiv but has a non-arXiv URL")
        if family in {"hf-daily", "hf-search"} and not url.startswith("https://huggingface.co/papers/"):
            raise RuntimeError(f"source [{item.get('n')}] is labeled {family} but has a non-Hugging-Face URL")
    if len(families) < 3:
        raise RuntimeError(f"finalized sources use only {len(families)} source families; at least 3 are required")

    run_summary = summarize(messages, elapsed, model_name)
    if run_summary["subagent_calls"] < 3:
        raise RuntimeError(
            f"lead made only {run_summary['subagent_calls']} researcher/task calls; at least 3 are required"
        )
    metadata = {
        "topic": topic,
        **run_summary,
        "n_sources": len(sources),
        "source_families": families,
    }
    slug = slugify(topic)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"
    meta_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write_set(
        [(report_path, report_bytes), (sources_path, sources_bytes), (meta_path, meta_bytes)]
    )
    return report_path


def _model_name(model):
    configured = (os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or "").strip()
    if configured:
        return configured
    for name in ("model_name", "model"):
        value = getattr(model, name, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def _sandbox_validation(backend):
    command = f"python3 {VALIDATOR_PATH} {REPORT_PATH} {SOURCES_PATH}"
    response = backend.execute(command)
    output = str(getattr(response, "output", "") or "")
    exit_code = getattr(response, "exit_code", None)
    return exit_code == 0 and "OK:" in output, exit_code, output


def _repair_prompt(attempt, exit_code, validator_output):
    return f"""The mandatory sandbox acceptance validator failed after your previous response.
Repair attempt {attempt} of {SANDBOX_REPAIR_LIMIT}; validator exit code: {exit_code}.

Validator output:
{_preview(validator_output, 4000)}

Continue working in the existing sandbox. Inspect `{SOURCES_PATH}`, `{REPORT_PATH}`, and the researcher notes under
`{WORKDIR}/research/notes`. For every invalid source family, recover the exact tool provenance from the notes: only
`arxiv`, `hf-daily`, `hf-search`, or `web` are allowed, and an arXiv-domain URL found through a web tool remains `web`.
Never infer or invent provenance from the URL. If provenance is unavailable, remove the unsupported record and perform
targeted supplementary research. After any source correction, run `{FINALIZER_PATH}` to regenerate References, then
run `{VALIDATOR_PATH}` again. Do not finish until it prints OK and at least three valid source families remain. Do not
return report prose in chat; repair the files in the sandbox."""


def main(topic):
    """Run one topic; return 0 on success, 1 on failed execution, or 2 for no topic."""
    topic = " ".join(str(topic or "").split())
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2

    started = time.monotonic()
    debug = _debug_enabled()
    debug_handler = Day23DebugHandler() if debug else None
    if debug:
        _debug_print(
            f"run started topic={topic!r} model_env={(os.getenv('LAB_MODEL') or '').strip()!r} "
            f"sandbox={(os.getenv('SANDBOX') or 'daytona').strip()!r}; limits="
            f"lead-model:{LEAD_MODEL_CALL_LIMIT}, lead-tool:{LEAD_TOOL_CALL_LIMIT}, "
            f"subagent-model:{SUBAGENT_MODEL_CALL_LIMIT}, subagent-tool:{SUBAGENT_TOOL_CALL_LIMIT}"
        )
    try:
        model = make_model()
        model_name = _model_name(model)
        if debug:
            _debug_print(f"model constructed class={type(model).__name__} name={model_name!r}")
        with open_sandbox() as backend:
            if debug:
                _debug_print(f"sandbox opened backend={type(backend).__name__}")
            setup = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if debug:
                _debug_print(
                    f"sandbox setup exit_code={getattr(setup, 'exit_code', None)} "
                    f"output={_preview(getattr(setup, 'output', ''), 1000)}"
                )
            if getattr(setup, "exit_code", 0) != 0:
                raise RuntimeError(f"cannot prepare sandbox directories: {getattr(setup, 'output', '')}")
            upload(
                backend,
                {
                    VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                    FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                },
            )
            if debug:
                _debug_print("validator and finalizer uploaded; constructing lead agent")
            agent = build_lead_agent(backend, model)
            config = {"recursion_limit": RECURSION_LIMIT}
            if debug_handler is not None:
                config["callbacks"] = [debug_handler]
            if debug:
                _debug_print(f"agent.invoke starting recursion_limit={RECURSION_LIMIT}")
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                    config=config,
                )
            except Exception:
                if debug:
                    _debug_print("agent.invoke raised; capturing sandbox state before cleanup")
                    debug_handler.report()
                    _debug_sandbox(backend)
                raise
            messages = result.get("messages", []) if isinstance(result, dict) else []
            if debug:
                _debug_print(
                    f"agent.invoke finished initial pass elapsed_s={time.monotonic() - started:.1f} "
                    f"result_type={type(result).__name__} messages={len(messages)}"
                )

            valid, validator_exit, validator_output = _sandbox_validation(backend)
            for repair_attempt in range(1, SANDBOX_REPAIR_LIMIT + 1):
                if valid:
                    break
                print(
                    f"Sandbox validation failed; asking lead to repair in place "
                    f"({repair_attempt}/{SANDBOX_REPAIR_LIMIT}).",
                    file=sys.stderr,
                )
                if debug:
                    _debug_print(
                        f"acceptance validator exit_code={validator_exit} "
                        f"output={_preview(validator_output, 4000)}"
                    )
                repair_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": _repair_prompt(
                            repair_attempt, validator_exit, validator_output
                        ),
                    },
                ]
                try:
                    repaired = agent.invoke({"messages": repair_messages}, config=config)
                except Exception:
                    if debug:
                        _debug_print(
                            f"repair agent.invoke #{repair_attempt} raised; capturing sandbox state"
                        )
                        debug_handler.report()
                        _debug_sandbox(backend)
                    raise
                messages = repaired.get("messages", repair_messages) if isinstance(repaired, dict) else repair_messages
                valid, validator_exit, validator_output = _sandbox_validation(backend)

            if not valid:
                if debug:
                    _debug_print(
                        f"sandbox validation still failing after {SANDBOX_REPAIR_LIMIT} repairs: "
                        f"exit_code={validator_exit} output={_preview(validator_output, 4000)}"
                    )
                    debug_handler.report()
                    _debug_messages(messages, time.monotonic() - started, model_name)
                    _debug_sandbox(backend)
                raise RuntimeError(
                    f"sandbox validation failed after {SANDBOX_REPAIR_LIMIT} bounded repair attempts "
                    f"(exit {validator_exit}): {_preview(validator_output, 2000)}"
                )

            elapsed = time.monotonic() - started
            if debug:
                _debug_print(
                    f"agent.invoke finished elapsed_s={elapsed:.1f} result_type={type(result).__name__}"
                )
                debug_handler.report()
                _debug_messages(messages, elapsed, model_name)
                _debug_print("capturing sandbox state before save_outputs and before cleanup")
                _debug_sandbox(backend)
            report_path = save_outputs(backend, topic, messages, elapsed, model_name)
    except Exception as exc:  # one failed run must return nonzero without partial artifacts
        if debug:
            _debug_print(
                f"run failed elapsed_s={time.monotonic() - started:.1f}: "
                f"{type(exc).__name__}: {_preview(exc, 2000)}"
            )
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if debug:
        _debug_print(f"run completed successfully elapsed_s={time.monotonic() - started:.1f}")
    print(f"Saved validated report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
