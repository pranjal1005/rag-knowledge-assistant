"""
Tests for build_prompt() in llm_client.py — the pure, network-free half of
that module. call_llm() itself needs a real API key and isn't covered here;
that's a manual/integration check, not a unit test.
Run with: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import Chunk
from src.llm_client import build_prompt


def _chunk(text, source_id="doc1", title="Doc One", idx=0):
    return Chunk(text=text, chunk_index=idx, source_id=source_id, source_title=title)


def test_prompt_includes_the_question():
    prompt = build_prompt("What is a data pipeline?", [])
    assert "What is a data pipeline?" in prompt


def test_prompt_with_no_chunks_says_so_explicitly():
    prompt = build_prompt("anything", [])
    assert "no relevant passages" in prompt.lower()


def test_prompt_numbers_chunks_starting_at_one():
    chunks = [_chunk("first passage"), _chunk("second passage", idx=1)]
    prompt = build_prompt("q", chunks)
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "first passage" in prompt
    assert "second passage" in prompt


def test_prompt_includes_source_titles_for_traceability():
    chunks = [_chunk("some text", source_id="paper42", title="Attention Is All You Need")]
    prompt = build_prompt("q", chunks)
    assert "Attention Is All You Need" in prompt


def test_prompt_asks_for_citations():
    prompt = build_prompt("q", [_chunk("x")])
    # loose check: the instruction to cite should be present somewhere,
    # without pinning the test to exact wording of the prompt copy
    assert "cit" in prompt.lower()  # matches "cite"/"citing"/"citation"


def test_unknown_provider_raises_value_error(monkeypatch):
    import src.llm_client as llm_client_module
    from types import SimpleNamespace

    stub_settings = SimpleNamespace(llm_provider="not-a-real-provider")
    monkeypatch.setattr(llm_client_module, "settings", stub_settings)
    with pytest.raises(ValueError):
        llm_client_module.call_llm("does not matter")


def test_gemini_without_key_raises_runtime_error(monkeypatch):
    import src.llm_client as llm_client_module
    from types import SimpleNamespace

    # settings is a frozen dataclass, so we swap the module-level `settings`
    # binding for a lightweight stub rather than trying to mutate it
    stub_settings = SimpleNamespace(llm_provider="gemini", gemini_api_key="")
    monkeypatch.setattr(llm_client_module, "settings", stub_settings)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm_client_module.call_llm("does not matter")


def test_anthropic_without_key_raises_runtime_error(monkeypatch):
    import src.llm_client as llm_client_module
    from types import SimpleNamespace

    stub_settings = SimpleNamespace(llm_provider="anthropic", anthropic_api_key="")
    monkeypatch.setattr(llm_client_module, "settings", stub_settings)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm_client_module.call_llm("does not matter")
