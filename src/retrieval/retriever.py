import os
from typing import List, Optional
from chromadb import PersistentClient
from ..ingestion.embedder import Embedder
from ..ingestion.models import Chunk

class Retriever:
    def __init__(self, persist_dir: str = ".chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("rag_documents")
        self.embedder = Embedder(model_name=embedding_model, cache_dir=os.path.join(persist_dir, ".embedding_cache"))

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        ids = [f"{c.source}_{c.chunk_index}" for c in chunks]
        metadatas = [
            {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
            for c in chunks
        ]
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 5, filter_source: Optional[str] = None) -> List[dict]:
        query_embedding = self.embedder.embed_query(query)
        where_filter = {"source": filter_source} if filter_source else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i].get("page", 0),
                "chunk_index": results["metadatas"][0][i]["chunk_index"],
                "relevance_score": results["distances"][0][i],
            })
        return chunks

    def list_sources(self) -> List[str]:
        all_docs = self.collection.get()
        sources = set()
        for meta in all_docs.get("metadatas", []):
            sources.add(meta["source"])
        return sorted(sources)

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.client.delete_collection("rag_documents")
        self.collection = self.client.get_or_create_collection("rag_documents")