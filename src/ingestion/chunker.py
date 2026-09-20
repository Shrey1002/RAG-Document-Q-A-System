from typing import List
from .models import Document, Chunk

def chunk_fixed(doc: Document, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    text = doc.text
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(
            text=chunk_text,
            source=doc.source,
            chunk_index=idx,
            page=doc.page,
            start_char=start,
            end_char=end
        ))
        idx += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def chunk_sentence(doc: Document, sentences_per_chunk: int = 5, overlap: int = 1) -> List[Chunk]:
    import re
    text = doc.text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    idx = 0
    for i in range(0, len(sentences), sentences_per_chunk - overlap):
        group = sentences[i:i + sentences_per_chunk]
        if not group:
            break
        chunk_text = " ".join(group)
        chunks.append(Chunk(
            text=chunk_text,
            source=doc.source,
            chunk_index=idx,
            page=doc.page
        ))
        idx += 1
    return chunks

def chunk_recursive(doc: Document, chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    text = doc.text
    if len(text) <= chunk_size:
        return [Chunk(
            text=text,
            source=doc.source,
            chunk_index=0,
            page=doc.page,
            start_char=0,
            end_char=len(text)
        )]

    chunks = []
    idx = 0
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            last_space = text[start:end].rfind('\n\n')
            if last_space > overlap:
                end = start + last_space + 2
            else:
                last_space = text[start:end].rfind('\n')
                if last_space > overlap:
                    end = start + last_space + 1
                else:
                    last_space = text[start:end].rfind(' ')
                    if last_space > overlap:
                        end = start + last_space + 1

        end = max(end, start + 1)
        chunk_text = text[start:end]

        chunks.append(Chunk(
            text=chunk_text,
            source=doc.source,
            chunk_index=idx,
            page=doc.page,
            start_char=start,
            end_char=end
        ))
        idx += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

CHUNKING_STRATEGIES = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "recursive": chunk_recursive,
}