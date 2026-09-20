from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Document:
    text: str
    source: str
    page: int = 0
    metadata: dict = field(default_factory=dict)

@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    page: int = 0
    metadata: dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0