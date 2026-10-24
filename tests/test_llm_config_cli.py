from types import SimpleNamespace

import pytest

from rag_citation_agent.cli import main
from rag_citation_agent.config import DEFAULT_MODEL, Settings, load_dotenv
from rag_citation_agent.llm import AnthropicClient, ScriptedLLMClient


class FakeChatModel:
    def __init__(self, content):
        self.content, self.args = content, None

    def invoke(self, messages, **kwargs):
        self.args = (messages, kwargs)
        return SimpleNamespace(content=self.content)


def test_anthropic_client_sends_system_and_user_and_joins_text_blocks():
    fake = FakeChatModel(["hel", {"type": "text", "text": "lo"}, {"type": "tool_use"}])
    client = AnthropicClient(None, "claude-test", client=fake)
    assert client.complete("sys", "usr", max_tokens=50) == "hello"
    assert fake.args == ([("system", "sys"), ("human", "usr")], {"max_tokens": 50})
    assert AnthropicClient(None, "m", client=FakeChatModel("plain")).complete("s", "u") == "plain"


def test_anthropic_client_requires_a_key_when_no_client_injected():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(None, "m")


def test_scripted_client_repeats_last_reply_and_records_calls():
    llm = ScriptedLLMClient(["a", "b"])
    assert [llm.complete("s", "u") for _ in range(3)] == ["a", "b", "b"]
    assert len(llm.calls) == 3


def test_settings_defaults_and_env_overrides(tmp_path):
    assert Settings().model == DEFAULT_MODEL == "claude-sonnet-5-5"
    s = Settings.from_env({"RAG_MODEL": "other", "RAG_TOP_K": "7", "ANTHROPIC_API_KEY": "k"}, dotenv=None)
    assert (s.model, s.top_k, s.anthropic_api_key) == ("other", 7, "k")


def test_dotenv_is_read_and_environment_wins(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# c\nANTHROPIC_API_KEY='from-file'\nRAG_MODEL=file-model\n", encoding="utf-8")
    assert load_dotenv(env)["ANTHROPIC_API_KEY"] == "from-file"
    s = Settings.from_env({"RAG_MODEL": "env-model"}, dotenv=env)
    assert (s.anthropic_api_key, s.model) == ("from-file", "env-model")


def test_env_example_ships_without_secrets():
    from pathlib import Path

    text = (Path(__file__).parent.parent / ".env.example").read_text()
    assert "ANTHROPIC_API_KEY=\n" in text and "RAG_MODEL=claude-sonnet-5-5" in text


def test_cli_ingest_then_ask_offline(tmp_path, monkeypatch, capsys):
    root = __import__("pathlib").Path(__file__).parent.parent
    monkeypatch.chdir(root)
    monkeypatch.setenv("RAG_INDEX_PATH", str(tmp_path / "idx.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["ingest", "data/knowledge_base"]) == 0
    assert main(["ask", "--offline", "How long are backups retained after deletion?"]) == 0
    out = capsys.readouterr().out
    assert "35 days" in out and "[1]" in out


def test_cli_ask_without_index_or_key_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["ask", "q"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert main(["ask", "--offline", "q"]) == 2
    assert "ingest" in capsys.readouterr().err


def test_demo_runs_all_four_behaviours_offline(monkeypatch, capsys):
    monkeypatch.chdir(__import__("pathlib").Path(__file__).parent.parent)
    assert main(["demo", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "[SEARCH FALLBACK]" in out and "REMOVED unverifiable claim" in out
    assert "status=abstained" in out and out.count("=== ") == 4
