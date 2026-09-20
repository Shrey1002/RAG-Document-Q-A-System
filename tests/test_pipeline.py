import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import RAGPipeline
from src.ingestion.models import Document, Chunk

@pytest.fixture
def pipeline(tmp_path):
    config = {
        "CHROMA_PERSIST_DIR": str(tmp_path / "test_chroma"),
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "OPENCODE_BASE_URL": "http://localhost:11434/v1",
        "OPENCODE_API_KEY": "test_key",
        "LLM_MODEL": "test-model",
    }
    yield RAGPipeline(config=config)

def test_ingest_and_count(pipeline, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document about artificial intelligence and machine learning.")
    result = pipeline.ingest(str(test_file))
    assert result["chunks_created"] > 0
    assert pipeline.count() > 0

def test_list_sources(pipeline, tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Sample content for testing.")
    pipeline.ingest(str(test_file))
    sources = pipeline.list_sources()
    assert "sample.txt" in sources

def test_query_returns_answer(pipeline, tmp_path):
    test_file = tmp_path / "data.txt"
    test_file.write_text("The company reported $10 million in revenue for Q3 2024. Growth was driven by cloud services.")
    pipeline.ingest(str(test_file))
    mock_answer = MagicMock()
    mock_answer.answer = "The revenue was $10 million."
    mock_answer.citations = [
        MagicMock(source="data.txt", page=0, chunk_index=0, relevance_score=0.95)
    ]
    mock_answer.chunks_used = 1
    mock_answer.retrieval_time_ms = 10
    mock_answer.generation_time_ms = 100
    with patch.object(pipeline.generator, 'generate', return_value=mock_answer):
        answer = pipeline.query("What was the revenue?", top_k=3)
    assert answer is not None
    assert answer.answer is not None

def test_query_citations_structure(pipeline, tmp_path):
    test_file = tmp_path / "info.txt"
    test_file.write_text("Python is a programming language. It is widely used in data science.")
    pipeline.ingest(str(test_file))
    mock_answer = MagicMock()
    mock_answer.answer = "Python is a programming language."
    mock_answer.citations = [
        MagicMock(source="info.txt", page=0, chunk_index=0, relevance_score=0.95)
    ]
    mock_answer.chunks_used = 1
    mock_answer.retrieval_time_ms = 5
    mock_answer.generation_time_ms = 50
    with patch.object(pipeline.generator, 'generate', return_value=mock_answer):
        answer = pipeline.query("What is Python?", top_k=2)
    assert len(answer.citations) > 0
    for c in answer.citations:
        assert hasattr(c, "source")
        assert hasattr(c, "page")
        assert hasattr(c, "chunk_index")
        assert hasattr(c, "relevance_score")

def test_clear(pipeline, tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content.")
    pipeline.ingest(str(test_file))
    assert pipeline.count() > 0
    pipeline.clear()
    assert pipeline.count() == 0