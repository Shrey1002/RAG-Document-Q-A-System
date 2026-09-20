import time
from typing import List, Optional
from .ingestion.loader import load_file
from .ingestion.chunker import CHUNKING_STRATEGIES
from .retrieval.retriever import Retriever
from .generation.generator import Generator, Answer

class RAGPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.retriever = Retriever(
            persist_dir=config.get("CHROMA_PERSIST_DIR", ".chroma_db"),
            embedding_model=config.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        self.generator = Generator(
            base_url=config.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=config.get("GROQ_API_KEY", ""),
            model=config.get("LLM_MODEL", "openai/gpt-oss-20b")
        )

    def ingest(self, file_path: str, chunking_strategy: str = "recursive") -> dict:
        from .ingestion.loader import load_file
        start = time.time()
        doc = load_file(file_path)
        chunks = CHUNKING_STRATEGIES[chunking_strategy](doc)
        self.retriever.add_chunks(chunks)
        elapsed = int((time.time() - start) * 1000)
        return {
            "chunks_created": len(chunks),
            "sources_indexed": len(set(c.source for c in chunks)),
            "time_taken_ms": elapsed
        }

    def query(self, question: str, top_k: int = 5, filter_source: Optional[str] = None, stream: bool = False) -> Answer:
        start_ret = time.time()
        chunks = self.retriever.search(question, top_k=top_k, filter_source=filter_source)
        retrieval_time = int((time.time() - start_ret) * 1000)

        answer = self.generator.generate(question, chunks, top_k)
        answer.retrieval_time_ms = retrieval_time

        return answer

    def list_sources(self) -> List[str]:
        return self.retriever.list_sources()

    def count(self) -> int:
        return self.retriever.count()

    def clear(self):
        self.retriever.clear()