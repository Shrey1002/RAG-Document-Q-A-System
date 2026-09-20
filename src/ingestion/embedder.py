import hashlib
from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = ".embedding_cache"):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self._cache = {}
        self._load_cache()

    def _load_cache(self):
        try:
            with open(self.cache_dir, "r") as f:
                self._cache = {}
        except FileNotFoundError:
            self._cache = {}

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            key = self._get_cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True).tolist()
            for idx, embedding in zip(uncached_indices, new_embeddings):
                key = self._get_cache_key(uncached_texts[idx])
                self._cache[key] = embedding
                results.append(embedding)
            self._save_cache()

        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def _save_cache(self):
        import json
        with open(self.cache_dir, "w") as f:
            json.dump({k: v for k, v in self._cache.items()}, f)

    def get_cache_stats(self) -> dict:
        return {"cache_size": len(self._cache)}