import os
from typing import List, Optional
from openai import OpenAI
from dataclasses import dataclass

@dataclass
class Citation:
    source: str
    page: int
    chunk_index: int
    relevance_score: float

@dataclass
class Answer:
    answer: str
    citations: List[Citation]
    model_used: str
    chunks_used: int
    retrieval_time_ms: int
    generation_time_ms: int

class Generator:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, question: str, chunks: List[dict], top_k: int) -> Answer:
        import time

        retrieved = chunks[:top_k]
        context_parts = []
        citations = []

        for i, chunk in enumerate(retrieved):
            context_parts.append(f"[Source: {chunk['source']}, Page {chunk['page']}, Chunk {chunk['chunk_index']}]")
            context_parts.append(chunk['text'])
            citations.append(Citation(
                source=chunk['source'],
                page=chunk['page'],
                chunk_index=chunk['chunk_index'],
                relevance_score=chunk['relevance_score']
            ))

        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context. 
If the answer is not in the context, say "I could not find this in the provided documents."

Context:
{context}

Question: {question}

Answer:"""

        start_gen = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            generation_time = int((time.time() - start_gen) * 1000)
            answer_text = response.choices[0].message.content.strip()
        except Exception as e:
            generation_time = int((time.time() - start_gen) * 1000)
            answer_text = f"I could not generate an answer because the LLM service is unavailable: {str(e)} Please check your GROQ_API_KEY and ensure the LLM endpoint is reachable."

        return Answer(
            answer=answer_text,
            citations=citations,
            model_used=self.model,
            chunks_used=len(retrieved),
            retrieval_time_ms=0,
            generation_time_ms=generation_time
        )