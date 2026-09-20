import pytest
from src.retrieval.retriever import Retriever
from src.ingestion.models import Document, Chunk

@pytest.fixture
def retriever(tmp_path):
    persist_dir = str(tmp_path / "test_chroma")
    r = Retriever(persist_dir=persist_dir)
    yield r
    r.clear()

def test_add_and_search(retriever):
    chunks = [
        Chunk(text="The quick brown fox jumps over the lazy dog.", source="test1.txt", chunk_index=0, page=1),
        Chunk(text="Climate change is affecting global weather patterns.", source="test2.txt", chunk_index=0, page=1),
    ]
    retriever.add_chunks(chunks)
    results = retriever.search("fox", top_k=1)
    assert len(results) >= 1
    assert "fox" in results[0]["text"].lower() or "dog" in results[0]["text"].lower()

def test_search_top_k(retriever):
    chunks = [
        Chunk(text=f"Document {i} content about topic A.", source=f"doc{i}.txt", chunk_index=0, page=1)
        for i in range(10)
    ]
    retriever.add_chunks(chunks)
    results = retriever.search("topic A", top_k=3)
    assert len(results) <= 3

def test_source_filtering(retriever):
    chunks = [
        Chunk(text="Apple stock price rose today.", source="finance.txt", chunk_index=0, page=1),
        Chunk(text="Apple released a new iPhone model.", source="tech.txt", chunk_index=0, page=1),
    ]
    retriever.add_chunks(chunks)
    results = retriever.search("Apple", top_k=5, filter_source="finance.txt")
    assert all(r["source"] == "finance.txt" for r in results)

def test_list_sources(retriever):
    chunks = [
        Chunk(text="Content A.", source="doc1.pdf", chunk_index=0, page=1),
        Chunk(text="Content B.", source="doc2.pdf", chunk_index=0, page=1),
        Chunk(text="Content C.", source="doc1.pdf", chunk_index=1, page=1),
    ]
    retriever.add_chunks(chunks)
    sources = retriever.list_sources()
    assert "doc1.pdf" in sources
    assert "doc2.pdf" in sources
    assert len(sources) == 2

def test_count(retriever):
    chunks = [
        Chunk(text=f"Text {i}.", source=f"doc{i}.txt", chunk_index=0, page=1)
        for i in range(5)
    ]
    retriever.add_chunks(chunks)
    assert retriever.count() == 5

def test_dedup_same_source(retriever):
    chunks = [
        Chunk(text=f"Chunk {i} from doc1.", source="doc1.txt", chunk_index=i, page=1)
        for i in range(3)
    ]
    retriever.add_chunks(chunks)
    results = retriever.search("doc1", top_k=5)
    assert len(results) == 3

def test_clear(retriever):
    chunks = [Chunk(text="Test.", source="test.txt", chunk_index=0, page=1)]
    retriever.add_chunks(chunks)
    assert retriever.count() > 0
    retriever.clear()
    assert retriever.count() == 0