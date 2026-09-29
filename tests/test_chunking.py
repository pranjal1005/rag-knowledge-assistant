"""
Unit tests for chunking.py. No network, no external deps — this is exactly
the kind of test that should run in CI on every commit in under a second.
Run with: pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.chunking import Chunk, chunk_documents, chunk_text


def test_short_text_returns_single_chunk():
    text = "This is a short sentence."
    result = chunk_text(text, chunk_size=800, overlap=150)
    assert result == [text]


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=800, overlap=150) == []
    assert chunk_text("   ", chunk_size=800, overlap=150) == []


def test_long_text_splits_into_multiple_chunks():
    text = "word " * 500  # ~2500 chars
    result = chunk_text(text, chunk_size=800, overlap=150)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= 800


def test_chunks_do_not_split_words():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 20
    result = chunk_text(text, chunk_size=100, overlap=20)
    for chunk in result:
        # no chunk should start or end with a partial word fragment that
        # isn't present as a whole word in the original text
        assert chunk == chunk.strip()
        words = chunk.split()
        for w in words:
            assert w in text.split()


def test_overlap_creates_shared_context_between_consecutive_chunks():
    text = "one two three four five six seven eight nine ten " * 10
    chunks = chunk_text(text, chunk_size=60, overlap=20)
    assert len(chunks) > 1
    # consecutive chunks should share at least one word given real overlap
    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    assert first_words & second_words


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, overlap=0)


def test_overlap_must_be_less_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_chunk_documents_preserves_source_metadata():
    docs = [
        {"id": "doc1", "title": "First Doc", "text": "short text"},
        {"id": "doc2", "title": "Second Doc", "text": "word " * 500},
    ]
    chunks = chunk_documents(docs, chunk_size=800, overlap=150)

    assert all(isinstance(c, Chunk) for c in chunks)

    doc1_chunks = [c for c in chunks if c.source_id == "doc1"]
    doc2_chunks = [c for c in chunks if c.source_id == "doc2"]

    assert len(doc1_chunks) == 1
    assert doc1_chunks[0].source_title == "First Doc"
    assert doc1_chunks[0].chunk_index == 0

    assert len(doc2_chunks) > 1
    # chunk_index should be sequential per-document, starting at 0
    assert [c.chunk_index for c in doc2_chunks] == list(range(len(doc2_chunks)))


def test_chunk_documents_handles_empty_list():
    assert chunk_documents([], chunk_size=800, overlap=150) == []


# --- Regression tests -------------------------------------------------
# Each of these pins down a real bug found while building this chunker:
# an infinite loop and two flavors of mid-word splitting. All three came
# from the same class of mistake (advancing `start` without guaranteeing
# it lands on a clean boundary AND makes forward progress), so these are
# kept as permanent guards rather than deleted once the fix landed.

def test_no_infinite_loop_on_a_token_longer_than_chunk_size():
    """A single word longer than chunk_size has nowhere clean to break --
    this used to hang forever instead of falling back to a hard cut."""
    text = "supercalifragilisticexpialidocious the quick brown fox " * 15
    for chunk_size, overlap in [(20, 5), (50, 20), (100, 50)]:
        result = chunk_text(text, chunk_size, overlap)
        assert len(result) > 0  # if this hangs, pytest's default timeout (if configured) will catch it


def test_no_mid_word_chunk_start_from_overlap_alignment():
    """The overlap offset (end - overlap) can land mid-word; this used to
    ship a chunk starting with a word fragment like 'arlie' instead of
    'charlie'."""
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 20
    words = set(text.split())
    for chunk_size, overlap in [(20, 5), (50, 15), (100, 40)]:
        for chunk in chunk_text(text, chunk_size, overlap):
            for w in chunk.split():
                assert w in words, f"split word {w!r} at chunk_size={chunk_size} overlap={overlap}"


def test_no_mid_word_start_after_hard_cut_lands_on_whitespace():
    """When a hard cut's boundary happens to be a space character, the next
    window used to start ON that space instead of after it, corrupting the
    following chunk's first word."""
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 20
    words = set(text.split())
    for chunk in chunk_text(text, chunk_size=5, overlap=0):
        for w in chunk.split():
            # chunk_size=5 is smaller than some words (e.g. "charlie"),
            # so a split there is an inherent, unavoidable limitation --
            # but words that DO fit (like "alpha", "bravo") must never
            # come out corrupted (e.g. "brav" instead of "bravo").
            if len(w) <= 5:
                assert w in words, f"short word {w!r} got corrupted"


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [(cs, ov) for cs in range(15, 100, 10) for ov in range(0, cs, 10)],
)
def test_no_word_splitting_sweep_realistic_sizes(chunk_size, overlap):
    """Sweep chunk_size/overlap for a fixed corpus, restricted to sizes at
    least as large as the corpus's longest word — the only regime where
    'never split a word' is achievable at all."""
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 20
    words = set(text.split())
    for chunk in chunk_text(text, chunk_size, overlap):
        for w in chunk.split():
            assert w in words
