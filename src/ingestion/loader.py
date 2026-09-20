import os
from typing import List
from .models import Document, Chunk
from .chunker import CHUNKING_STRATEGIES

def load_file(file_path: str) -> Document:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    elif ext == ".md":
        return load_md(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_pdf(file_path: str) -> Document:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("PyPDF2 is required. Install with: pip install PyPDF2")

    reader = PdfReader(file_path)
    text_parts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(f"--- Page {i+1} ---\n{page_text}")

    text = "\n\n".join(text_parts)
    return Document(text=text, source=os.path.basename(file_path), page=0)

def load_txt(file_path: str) -> Document:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return Document(text=text, source=os.path.basename(file_path), page=0)

def load_md(file_path: str) -> Document:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return Document(text=text, source=os.path.basename(file_path), page=0)

LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_md,
}