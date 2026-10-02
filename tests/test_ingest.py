import pytest

from app.ingest import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share some words because of the overlap
    assert set(chunks[0].split()[-3:]) & set(chunks[1].split())


def test_words_are_not_split():
    text = " ".join(f"token{i}" for i in range(300))
    for c in chunk_text(text, size=120, overlap=30):
        for w in c.split():
            assert w.startswith("token")


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)
