import json
from contextlib import contextmanager
from types import SimpleNamespace

import agents
import pytest
import research


def test_build_subagents_tools_and_limits():
    specs = agents.build_subagents()
    assert [spec["name"] for spec in specs] == ["researcher", "citation-checker"]
    assert [tool.name for tool in specs[0]["tools"]] == [tool.name for tool in agents.SOURCE_TOOLS]
    assert [tool.name for tool in specs[1]["tools"]] == ["web_fetch"]
    assert all(len(spec["middleware"]) == 2 for spec in specs)


def test_prompts_enforce_source_enum_and_provenance_repair():
    lead = " ".join(agents.LEAD_PROMPT.split())
    researcher = " ".join(agents.RESEARCHER_PROMPT.split())
    for value in ("arxiv", "hf-daily", "hf-search", "web"):
        assert value in lead
        assert value in researcher
    assert "strict, case-sensitive enum" in lead
    assert "NOT the URL domain" in lead
    assert "arXiv URL discovered by web_search/web_fetch remains" in lead
    assert "inspect the corresponding researcher note" in lead
    assert "rerun the finalizer" in lead
    assert "If you cannot identify the producing tool, omit the source" in researcher


def test_build_lead_agent_passes_required_configuration(monkeypatch):
    captured = {}
    monkeypatch.setattr(agents, "create_deep_agent", lambda **kwargs: captured.update(kwargs) or "agent")
    assert agents.build_lead_agent("backend", "model") == "agent"
    assert captured["backend"] == "backend"
    assert captured["model"] == "model"
    assert len(captured["subagents"]) == 2
    assert len(captured["middleware"]) == 3


def test_slugify_is_safe_bounded_and_nonempty():
    assert research.slugify("../../Hello, World!") == "hello-world"
    assert research.slugify("   ") == "topic"
    assert research.slugify("CON") == "topic-con"
    assert len(research.slugify("x" * 100)) == 60
    assert "/" not in research.slugify("../../x")


def test_build_prompt_contains_topic_and_acceptance_requirements():
    prompt = research.build_prompt("  survey   about world model ")
    assert '"survey about world model"' in prompt
    assert "three independent researcher tasks" in prompt
    assert "validator prints OK" in prompt
    assert research.REPORT_PATH in prompt
    assert research.SOURCES_PATH in prompt


def test_summarize_counts_real_calls_and_usage():
    messages = [
        {
            "tool_calls": [{"name": "task"}, {"name": "read_file"}],
            "usage_metadata": {"input_tokens": 10, "output_tokens": 4},
        },
        SimpleNamespace(
            tool_calls=[{"name": "task"}, {"name": "task"}, {"name": "execute"}],
            usage_metadata={"input_tokens": 7, "output_tokens": 3},
        ),
    ]
    result = research.summarize(messages, 1.26, "model-x")
    assert result == {
        "model": "model-x",
        "elapsed_s": 1.3,
        "subagent_calls": 3,
        "tool_calls": {"execute": 1, "read_file": 1, "task": 3},
        "tokens": {"input": 17, "output": 7},
    }


def _valid_artifacts():
    report = (
        b"# T\n\nClaim [1][2][3].\n\n## References\n"
        b"[1] A. arxiv. https://arxiv.org/abs/2501.1 (2026-01-01)\n"
        b"[2] B. hf-search. https://huggingface.co/papers/2501.2 (2026-01-02)\n"
        b"[3] C. web. https://example.test/3 (2026-01-03)\n"
    )
    sources = json.dumps(
        [
            {"n": 1, "id": "2501.1", "url": "https://arxiv.org/abs/2501.1", "title": "A", "date": "2026-01-01", "source": "arxiv"},
            {"n": 2, "id": "2501.2", "url": "https://huggingface.co/papers/2501.2", "title": "B", "date": "2026-01-02", "source": "hf-search"},
            {"n": 3, "id": "c", "url": "https://example.test/3", "title": "C", "date": "2026-01-03", "source": "web"},
        ]
    ).encode()
    return {research.REPORT_PATH: report, research.SOURCES_PATH: sources}


def test_save_outputs_writes_three_valid_files(monkeypatch, tmp_path):
    artifacts = _valid_artifacts()
    monkeypatch.setattr(research, "download", lambda _backend, _paths: artifacts)
    messages = [
        {
            "tool_calls": [{"name": "task"}, {"name": "task"}, {"name": "task"}],
            "usage_metadata": {"input_tokens": 2, "output_tokens": 1},
        }
    ]
    report_path = research.save_outputs(object(), "My Topic", messages, 2.0, "m", tmp_path)
    assert report_path == tmp_path / "my-topic.md"
    assert report_path.read_bytes() == artifacts[research.REPORT_PATH]
    assert (tmp_path / "my-topic.sources.json").read_bytes() == artifacts[research.SOURCES_PATH]
    meta = json.loads((tmp_path / "my-topic.meta.json").read_text(encoding="utf-8"))
    assert meta["n_sources"] == 3
    assert meta["source_families"] == ["arxiv", "hf-search", "web"]
    assert meta["subagent_calls"] == 3


def test_save_outputs_invalid_download_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        research,
        "download",
        lambda _backend, _paths: {research.REPORT_PATH: b"", research.SOURCES_PATH: b"not-json"},
    )
    try:
        research.save_outputs(object(), "bad", [], 0, "m", tmp_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
    assert list(tmp_path.iterdir()) == []


def test_save_outputs_rejects_invalid_family_without_host_rewrite(monkeypatch, tmp_path):
    artifacts = _valid_artifacts()
    original_sources = artifacts[research.SOURCES_PATH]
    invalid = json.loads(original_sources)
    invalid[0]["source"] = "arXiv primary paper"
    artifacts[research.SOURCES_PATH] = json.dumps(invalid).encode()
    monkeypatch.setattr(research, "download", lambda _backend, _paths: artifacts)
    messages = [{"tool_calls": [{"name": "task"}] * 3, "usage_metadata": {}}]

    with pytest.raises(RuntimeError, match="invalid source family"):
        research.save_outputs(object(), "bad family", messages, 1.0, "m", tmp_path)

    assert list(tmp_path.iterdir()) == []
    assert json.loads(artifacts[research.SOURCES_PATH])[0]["source"] == "arXiv primary paper"
    assert b"arXiv primary paper" not in original_sources


def test_main_cleans_up_and_returns_one_on_agent_failure(monkeypatch):
    state = {"exited": False}

    class Backend:
        def execute(self, _command):
            return SimpleNamespace(exit_code=0, output="")

    @contextmanager
    def sandbox():
        try:
            yield Backend()
        finally:
            state["exited"] = True

    class FailingAgent:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("agent failed")

    monkeypatch.setattr(research, "make_model", lambda: object())
    monkeypatch.setattr(research, "open_sandbox", sandbox)
    monkeypatch.setattr(research, "upload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(research, "build_lead_agent", lambda *_args: FailingAgent())
    assert research.main("topic") == 1
    assert state["exited"] is True


def test_main_returns_two_for_empty_topic(monkeypatch):
    monkeypatch.setattr(research, "make_model", lambda: (_ for _ in ()).throw(AssertionError("must not run")))
    assert research.main("  ") == 2


def test_main_repairs_failed_validator_inside_same_sandbox(monkeypatch, tmp_path):
    state = {"active": False, "validator_calls": 0, "agent_inputs": [], "uploaded": {}}

    class Backend:
        def execute(self, command):
            assert state["active"] is True
            if command.startswith("python3 ") and research.VALIDATOR_PATH in command:
                state["validator_calls"] += 1
                if state["validator_calls"] == 1:
                    return SimpleNamespace(
                        exit_code=1,
                        output="source [1] has invalid source family: 'arXiv primary paper'",
                    )
                return SimpleNamespace(exit_code=0, output="OK: 3 sources, all citations resolve")
            return SimpleNamespace(exit_code=0, output="")

    @contextmanager
    def sandbox():
        state["active"] = True
        try:
            yield Backend()
        finally:
            state["active"] = False

    class RepairingAgent:
        def invoke(self, payload, config):
            state["agent_inputs"].append(payload)
            if len(state["agent_inputs"]) == 1:
                return {
                    "messages": [
                        {
                            "type": "ai",
                            "tool_calls": [{"name": "task"}] * 3,
                            "usage_metadata": {},
                        }
                    ]
                }
            return {"messages": payload["messages"] + [{"type": "ai", "content": "repaired"}]}

    monkeypatch.delenv(research.DEBUG_ENV, raising=False)
    monkeypatch.setattr(research, "make_model", lambda: object())
    monkeypatch.setattr(research, "open_sandbox", sandbox)
    monkeypatch.setattr(
        research,
        "upload",
        lambda _backend, files: state["uploaded"].update(files),
    )
    monkeypatch.setattr(research, "build_lead_agent", lambda *_args: RepairingAgent())
    monkeypatch.setattr(research, "save_outputs", lambda *_args, **_kwargs: tmp_path / "report.md")

    assert research.main("topic") == 0
    assert state["validator_calls"] == 2
    assert len(state["agent_inputs"]) == 2
    repair_message = state["agent_inputs"][1]["messages"][-1]["content"]
    assert "recover the exact tool provenance" in repair_message
    assert "Never infer or invent provenance from the URL" in repair_message
    assert research.FINALIZER_PATH in repair_message
    assert research.VALIDATOR_PATH in state["uploaded"]
    assert b"invalid source family" in state["uploaded"][research.VALIDATOR_PATH]
    assert state["active"] is False


def test_debug_missing_report_captures_state_before_cleanup(monkeypatch, capsys):
    secret = "private-debug-key-value"
    state = {"active": False, "exited": False, "commands": []}

    class Backend:
        def execute(self, command):
            assert state["active"] is True
            state["commands"].append(command)
            output = f"MISSING {research.REPORT_PATH} {secret}" if command.startswith("for p in") else "ok"
            return SimpleNamespace(exit_code=0, output=output)

    @contextmanager
    def sandbox():
        state["active"] = True
        try:
            yield Backend()
        finally:
            state["active"] = False
            state["exited"] = True

    class EarlyAgent:
        def invoke(self, _input, config):
            assert config["callbacks"]
            return {
                "messages": [
                    {
                        "type": "ai",
                        "content": "Returned directly without writing files",
                        "response_metadata": {"finish_reason": "stop"},
                    }
                ]
            }

    monkeypatch.setenv(research.DEBUG_ENV, "1")
    monkeypatch.setenv("LAB_API_KEY", secret)
    monkeypatch.setattr(research, "make_model", lambda: object())
    monkeypatch.setattr(research, "open_sandbox", sandbox)
    monkeypatch.setattr(research, "upload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(research, "build_lead_agent", lambda *_args: EarlyAgent())
    monkeypatch.setattr(
        research,
        "save_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError(f"sandbox did not produce report: {research.REPORT_PATH}")
        ),
    )

    assert research.main("topic") == 1
    captured = capsys.readouterr().err
    assert "agent.invoke finished" in captured
    assert "lead returned 1 messages" in captured
    assert f"MISSING {research.REPORT_PATH}" in captured
    assert "finish_reason=stop" in captured
    assert secret not in captured
    assert "[REDACTED]" in captured
    assert any(command == f"ls -la {research.WORKDIR}/report" for command in state["commands"])
    assert state["exited"] is True


def test_debug_callback_counts_and_redacts_tool_error(monkeypatch, capsys):
    secret = "callback-private-key"
    monkeypatch.setenv("EXA_API_KEY", secret)
    handler = research.Day23DebugHandler(event_limit=10)
    run_id = "run-1"
    handler.on_tool_start(
        {"name": "task"},
        "ignored",
        run_id=run_id,
        inputs={"description": "research", "api_key": secret},
    )
    handler.on_tool_error(RuntimeError(f"failed exaApiKey={secret}"), run_id=run_id)
    handler.report()
    captured = capsys.readouterr().err
    assert handler.tool_calls["task"] == 1
    assert handler.tool_errors["task"] == 1
    assert secret not in captured
    assert "[REDACTED]" in captured
