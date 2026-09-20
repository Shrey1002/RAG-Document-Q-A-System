import pytest
from src.ingestion.chunker import chunk_fixed, chunk_sentence, chunk_recursive, CHUNKING_STRATEGIES
from src.ingestion.models import Document

def test_chunk_fixed():
    doc = Document(text=" ".join([f"word{i}" for i in range(100)]), source="test.txt")
    chunks = chunk_fixed(doc, chunk_size=50, overlap=10)
    assert len(chunks) > 0
    assert all(c.chunk_index == i for i, c in enumerate(chunks))

def test_chunk_fixed_overlap():
    doc = Document(text="a " * 200, source="test.txt")
    chunks = chunk_fixed(doc, chunk_size=50, overlap=10)
    assert len(chunks) > 1

def test_chunk_sentence():
    doc = Document(text="This is sentence one. This is sentence two. This is sentence three.", source="test.txt")
    chunks = chunk_sentence(doc, sentences_per_chunk=2, overlap=1)
    assert len(chunks) > 0

def test_chunk_recursive():
    text = "\n\n".join([f"Paragraph {i}. " * 10 for i in range(5)])
    doc = Document(text=text, source="test.txt")
    chunks = chunk_recursive(doc, chunk_size=100, overlap=10)
    assert len(chunks) > 0

def test_chunk_recursive_small_doc():
    doc = Document(text="Short text.", source="test.txt")
    chunks = chunk_recursive(doc, chunk_size=512, overlap=64)
    assert len(chunks) == 1

def test_empty_document():
    doc = Document(text="", source="test.txt")
    chunks = chunk_recursive(doc)
    assert len(chunks) == 1

def test_single_sentence():
    doc = Document(text="Only one sentence here.", source="test.txt")
    chunks = chunk_sentence(doc)
    assert len(chunks) > 0

def test_chunking_strategies_dict():
    assert "fixed" in CHUNKING_STRATEGIES
    assert "sentence" in CHUNKING_STRATEGIES
    assert "recursive" in CHUNKING_STRATEGIES

def test_metadata_preserved():
    doc = Document(text="Test text for metadata.", source="report.pdf", page=3)
    chunks = chunk_recursive(doc, chunk_size=50, overlap=5)
    assert all(c.source == "report.pdf" for c in chunks)

def test_overlap_correctness():
    text = "word" * 100
    doc = Document(text=text, source="test.txt")
    chunks = chunk_fixed(doc, chunk_size=20, overlap=5)
    assert len(chunks) > 1