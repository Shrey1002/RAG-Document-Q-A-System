# 📄 RAG Document Q&A System

A production-quality Retrieval-Augmented Generation (RAG) system that lets you upload documents (PDF, TXT, Markdown) and ask natural-language questions about them — with source citations for every answer. Built entirely with free tools and the Groq LLM API.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Architecture Overview](#architecture-overview)
- [Architecture Decision Records](#architecture-decision-records)
  - [Chunking Strategy Comparison](#chunking-strategy-comparison)
  - [Why ChromaDB](#why-chromadb-over-pinecone-weaviate-or-pgvector)
  - [Embedding Cache](#embedding-cache-and-why-it-matters)
  - [Free LLM via Groq](#free-llm-via-groq)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [CLI Reference](#cli-reference)
- [Running Tests](#running-tests)
- [How to Extend](#how-to-extend)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What This Project Does

You point this system at a folder of PDFs, text files, or Markdown documents. It:

1. **Ingests** each document — loading, cleaning, and splitting it into overlapping chunks
2. **Embeds** each chunk into a vector using a local sentence-transformer model (no API calls, no cost)
3. **Stores** the vectors in a persistent local ChromaDB database
4. **Answers** your questions by retrieving the most relevant chunks, building a grounded context window, and calling the Groq LLM API
5. **Cites** its sources — every answer includes the filename, page number, and relevance score of the chunks it used

The result is a system that answers questions about your documents without hallucinating, because the LLM is explicitly instructed to use only what was retrieved — and it tells you exactly where each piece of information came from.

---

## Architecture Overview

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                      RAG Pipeline                        │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  Document    │    │   Vector     │    │   LLM     │  │
│  │  Ingestion   │───▶│   Store      │───▶│ Generator │  │
│  │              │    │  (ChromaDB)  │    │           │  │
│  │  • Loader    │    │              │    │   Groq    │  │
│  │  • Chunker   │    │  • Embed     │    │ Free API  │  │
│  │  • Embedder  │    │  • Search    │    │           │  │
│  └──────────────┘    │  • Dedup     │    └───────────┘  │
│                      └──────────────┘          │        │
└─────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                     Answer + Citations
```

**Data flow for ingestion:**
`File` → `Loader` (extract text + metadata) → `Chunker` (split with overlap) → `Embedder` (local model, cached) → `ChromaDB` (persist to disk)

**Data flow for querying:**
`Question` → `Embedder` (embed question) → `ChromaDB` (cosine similarity search, top-k) → `Deduplicator` → `Generator` (build prompt + call LLM) → `Answer + Citations`

---

## Architecture Decision Records

### Chunking Strategy Comparison

The way you split documents has a larger impact on retrieval quality than almost any other design decision. This system implements three strategies, selectable at runtime.

#### Strategy 1: Fixed-Size Chunking

Split the document into chunks of exactly N tokens, with M tokens of overlap between consecutive chunks.

```
Default: chunk_size=512, overlap=64
```

**When to use it:**
- Documents with highly uniform structure (e.g., transcripts, logs, form data)
- When you need predictable, consistent chunk sizes for downstream processing
- As a baseline to benchmark other strategies against

**Trade-offs:**
- ✅ Predictable: every chunk is approximately the same size
- ✅ Simple to reason about and debug
- ❌ Breaks mid-sentence constantly — a chunk may end halfway through a key fact
- ❌ Semantically incoherent chunks reduce retrieval precision
- ❌ Overlap is token-based, not meaning-based, so it may repeat noise

#### Strategy 2: Sentence-Boundary Chunking

Group N complete sentences into a chunk, with 1 sentence of overlap between chunks.

```
Default: sentences_per_chunk=5, overlap=1
```

**When to use it:**
- Journalistic, legal, or literary text where sentences are the natural unit of meaning
- Short documents where sentence boundaries align with logical units
- When you want each chunk to contain only complete thoughts

**Trade-offs:**
- ✅ Never cuts mid-sentence — every chunk is semantically complete
- ✅ Natural overlap via shared sentences
- ❌ Sentence length varies wildly, so chunk sizes are unpredictable
- ❌ Sentence boundary detection can fail on abbreviations (e.g., "Dr. Smith")
- ❌ Poor for code, tables, or bullet-point documents where "sentences" don't apply

#### Strategy 3: Recursive Character Splitting (Default)

Attempt to split on paragraph breaks (`\n\n`), then newlines (`\n`), then spaces — recursively, until chunks are within the size limit.

```
Default: chunk_size=512 tokens, overlap=64 tokens
```

**When to use it:**
- Mixed documents (the common case) — PDFs, reports, README files, books
- When document structure is unknown in advance
- As the safe default because it degrades gracefully

**Trade-offs:**
- ✅ Respects natural document structure wherever possible
- ✅ Falls back cleanly when structure is absent
- ✅ Works well on prose, code, markdown, and lists
- ✅ Most widely used in production RAG systems for good reason
- ❌ More complex to implement and test than fixed-size
- ❌ Can still break at paragraph boundaries that don't align with topic shifts

**Why recursive is the default:** In practice, documents arrive in all shapes — a legal PDF, a README, a meeting transcript. Recursive chunking tries to preserve the largest meaningful structural unit that fits within the size constraint. It sacrifices some predictability for much better retrieval relevance.

#### The impact of overlap

All strategies use overlap. Without overlap, a fact that spans two chunks (e.g., "The deadline is..." ending one chunk, "...March 15th" starting the next) would be retrieved only partially. With overlap, the overlapping tokens ensure that cross-boundary facts appear in at least one complete chunk. The cost is storing slightly more data and embedding slightly more text — a worthwhile trade-off.

---

### Why ChromaDB Over Pinecone, Weaviate, or pgvector

This is a deliberate choice, not a limitation. Here is the reasoning:

| Dimension | ChromaDB | Pinecone | Weaviate | pgvector |
|---|---|---|---|---|
| **Cost** | Free, local | Free tier + paid | Free tier + paid | Free (self-hosted) |
| **Setup** | `pip install chromadb` | Account + API key | Docker or cloud | Postgres + extension |
| **Persistence** | Local disk | Cloud | Cloud or local | Postgres DB |
| **Python API** | Native | REST/SDK | REST/SDK | SQLAlchemy / psycopg |
| **Production scale** | Up to ~1M vectors | Billions | Billions | Depends on Postgres |
| **Metadata filtering** | Yes | Yes | Yes | Yes |
| **Hybrid search** | Basic | Yes | Yes | With pg_trgm |
| **Portfolio signal** | Shows pragmatism | Shows cloud fluency | Shows stack breadth | Shows DB depth |

**For this project, ChromaDB wins because:**

1. **Zero dependencies** beyond `pip install` — no accounts, no Docker, no cloud bills. This lets anyone clone the repo and run it immediately.

2. **Persistent by default** — data survives restarts without any configuration. The `.chroma_db/` directory on disk is the entire database.

3. **Native Python** — the API feels like working with a Python list, not sending HTTP requests to a cloud service. This keeps the code readable.

4. **Sufficient for the scale** — a portfolio RAG system processing hundreds of documents with thousands of chunks is well within ChromaDB's range.

**When you'd switch away from ChromaDB:**
- Pinecone: when you need managed cloud hosting, multi-tenant access, or billions of vectors
- Weaviate: when you need hybrid BM25+vector search, GraphQL queries, or a full multimodal pipeline
- pgvector: when your existing stack is already Postgres and you want vectors co-located with relational data

For a recruiter demo or interview conversation, knowing *why* you made this choice — and what the migration path would be — matters more than which one you picked.

---

### Embedding Cache and Why It Matters

Every call to the embedding model takes time (typically 10–100ms per batch on CPU). If you re-ingest the same document because you restarted the server, re-ran the pipeline, or cleared and rebuilt your vector store, you'd re-embed every chunk from scratch — wasting compute time that scales linearly with document volume.

This system caches embeddings to disk using a content hash as the key:

```python
cache_key = hashlib.sha256(chunk_text.encode()).hexdigest()
```

If a chunk with that exact content has been embedded before, the embedding is read from cache instead of recomputed. This means:

- **First ingest of a 100-page PDF**: full embedding time (~30 seconds on CPU)
- **Second ingest of the same PDF**: near-instant (cache hit for every chunk)
- **Re-ingest after adding 5 new pages**: only the 5 new pages' chunks are embedded

In production, this pattern matters for three reasons:

1. **Cost** — if you switch from a local embedding model to a paid API (OpenAI's `text-embedding-3-small` costs money per token), re-embedding the same content repeatedly burns budget
2. **Latency** — users don't wait for re-embedding when the content hasn't changed
3. **Reproducibility** — the same text always produces the same embedding, so the cache never serves stale or incorrect data (unlike a cache for LLM responses, which are non-deterministic)

---

### Free LLM via Groq

This project uses the [Groq](https://console.groq.com) API, which provides ultra-fast inference on open-weight models (currently `openai/gpt-oss-20b` and `openai/gpt-oss-120b`) via an OpenAI-compatible API, with a free tier for development.

```
Base URL: https://api.groq.com/openai/v1
Endpoint: POST /chat/completions
Auth:     Bearer <GROQ_API_KEY>
```

Because it uses the OpenAI SDK interface, swapping in any other provider (OpenAI, Anthropic, local Ollama) requires changing only the `base_url` and `model` in your `.env` file — no code changes.

---

## Project Structure

```
rag-qa/
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── models.py        # Document and Chunk dataclasses
│   │   ├── loader.py        # PDF, TXT, MD file loaders
│   │   ├── chunker.py       # Three chunking strategies
│   │   └── embedder.py      # Local sentence-transformer embeddings + cache
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py     # ChromaDB vector store + search + dedup
│   ├── generation/
│   │   ├── __init__.py
│   │   └── generator.py     # LLM call with context building + citations
│   └── pipeline.py          # Orchestrates ingest + query end-to-end
├── app/
│   └── main.py              # FastAPI REST API
├── cli.py                   # Command-line interface
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A Groq account with a free API key — sign up at [console.groq.com](https://console.groq.com)

### Install dependencies

```bash
git clone https://github.com/yourusername/rag-qa.git
cd rag-qa
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first time you run the system, `sentence-transformers` will download the embedding model (~90 MB). This happens once and is cached locally by the library.

### Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=.chroma_db
CHUNK_SIZE=512
CHUNK_OVERLAP=64
DEFAULT_TOP_K=5
```

---

## Running the App

### FastAPI server

```bash
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

### CLI

```bash
# Ingest a single file
python cli.py ingest ./documents/report.pdf

# Ingest all PDFs and markdown files in a directory
python cli.py ingest ./documents/

# Ask a question
python cli.py query "What were the main findings of the study?"

# Ask a question and stream the response live
python cli.py query "Summarize the methodology section" --stream

# List all indexed sources
python cli.py sources

# Show system statistics
python cli.py stats

# Clear the vector store (irreversible)
python cli.py clear
```

---

## API Reference

All requests and responses use JSON. The server runs at `http://localhost:8000` by default.

---

### `POST /ingest`

Ingest a document into the vector store.

**Request body:**

```json
{
  "file_path": "/absolute/path/to/document.pdf",
  "chunking_strategy": "recursive"
}
```

Or, to upload file content directly:

```json
{
  "filename": "report.pdf",
  "content_base64": "<base64-encoded file bytes>",
  "chunking_strategy": "recursive"
}
```

`chunking_strategy` is optional. Accepts `"recursive"` (default), `"fixed"`, or `"sentence"`.

**Response:**

```json
{
  "chunks_created": 142,
  "sources_indexed": 1,
  "time_taken_ms": 3241
}
```

**curl example:**

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/home/user/docs/annual_report.pdf", "chunking_strategy": "recursive"}'
```

---

### `POST /query`

Ask a question and receive an answer with citations.

**Request body:**

```json
{
  "question": "What were the key revenue drivers in Q3?",
  "top_k": 5,
  "filter_source": null,
  "stream": false
}
```

`top_k` is optional (default 5). `filter_source` accepts a filename string to restrict retrieval to a single document. `stream` returns a streaming response when `true`.

**Response:**

```json
{
  "answer": "According to the Q3 earnings report, the key revenue drivers were cloud services (up 34% YoY) and enterprise software subscriptions (up 18% YoY).",
  "citations": [
    {
      "source": "q3_earnings_report.pdf",
      "page": 4,
      "chunk_index": 17,
      "relevance_score": 0.91
    },
    {
      "source": "q3_earnings_report.pdf",
      "page": 5,
      "chunk_index": 21,
      "relevance_score": 0.87
    }
  ],
  "model_used": "openai/gpt-oss-20b",
  "chunks_used": 5,
  "retrieval_time_ms": 48,
  "generation_time_ms": 1203
}
```

**curl example:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the company revenue?", "top_k": 5}'
```

**Search within a specific file:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the risks?", "filter_source": "annual_report.pdf"}'
```

---

### `GET /sources`

List all documents currently indexed in the vector store.

**Response:**

```json
{
  "sources": [
    "annual_report.pdf",
    "q3_earnings.pdf",
    "product_roadmap.md"
  ],
  "total": 3
}
```

**curl example:**

```bash
curl http://localhost:8000/sources
```

---

### `GET /stats`

Return system statistics.

**Response:**

```json
{
  "total_chunks": 847,
  "total_sources": 3,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_model": "openai/gpt-oss-20b",
  "chroma_persist_dir": ".chroma_db"
}
```

**curl example:**

```bash
curl http://localhost:8000/stats
```

---

### `DELETE /clear`

Remove all documents and vectors from the store. **This is irreversible.**

**Response:**

```json
{
  "message": "Vector store cleared successfully.",
  "chunks_removed": 847
}
```

**curl example:**

```bash
curl -X DELETE http://localhost:8000/clear
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite uses an in-memory ChromaDB instance and mocks all LLM API calls — no real API calls are made and no network connection is required. Tests are safe to run in CI.

```bash
# Run a specific test file
pytest tests/test_chunker.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

**What the tests cover:**

- `test_chunker.py` — all three chunking strategies, edge cases (empty doc, single sentence, doc shorter than chunk size), metadata preservation, overlap correctness
- `test_retriever.py` — add documents, cosine similarity search, deduplication of same-source chunks, source filtering, `count()` and `list_sources()`
- `test_pipeline.py` — end-to-end integration test using a sample text string, verifying that a question returns an answer with correctly structured citations

---

## How to Extend

### Swap in a different embedding model

In `.env`, change:

```env
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

Any model from the [sentence-transformers model hub](https://www.sbert.net/docs/pretrained_models.html) works. Larger models are more accurate but slower. After changing the model, clear your vector store and re-ingest (embeddings from different models are not comparable):

```bash
python cli.py clear
python cli.py ingest ./documents/
```

### Switch to a paid LLM

In `.env`, replace the Groq endpoint with any OpenAI-compatible provider:

```env
# OpenAI
GROQ_BASE_URL=https://api.openai.com/v1
GROQ_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

# Anthropic (via their OpenAI-compatible endpoint)
GROQ_BASE_URL=https://api.anthropic.com/v1
GROQ_API_KEY=sk-ant-...
LLM_MODEL=claude-sonnet-4-5

# Local Ollama
GROQ_BASE_URL=http://localhost:11434/v1
GROQ_API_KEY=unused
LLM_MODEL=llama3.2
```

No code changes required.

### Add a new file type

Add a loader function in `src/ingestion/loader.py`:

```python
def load_docx(file_path: str) -> Document:
    # Use python-docx to extract text
    ...
```

Register it in the `LOADERS` dispatch dict:

```python
LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md":  load_md,
    ".docx": load_docx,   # add this line
}
```

### Add a new chunking strategy

Add a function in `src/ingestion/chunker.py` with this signature:

```python
def chunk_by_topic(doc: Document, **kwargs) -> List[Chunk]:
    ...
```

Register it in `CHUNKING_STRATEGIES`:

```python
CHUNKING_STRATEGIES = {
    "fixed":     chunk_fixed,
    "sentence":  chunk_sentence,
    "recursive": chunk_recursive,
    "topic":     chunk_by_topic,   # add this line
}
```

### Deploy to production

1. **Containerize** — add a `Dockerfile` that installs dependencies and runs `uvicorn`
2. **Swap ChromaDB for a cloud vector store** — update `src/retrieval/retriever.py` to use Pinecone or Weaviate; the `RAGPipeline` interface stays the same
3. **Persist the embedding cache** — move the cache directory to a mounted volume
4. **Add authentication** — wrap the FastAPI app with an API key middleware or OAuth

---

## Troubleshooting

**`sentence-transformers` model download is slow**
This happens only on first run. The model (~90 MB) is cached locally after that.

**`chromadb` raises a dimension mismatch error**
This happens if you changed `EMBEDDING_MODEL` without clearing the vector store. Run `python cli.py clear` and re-ingest.

**LLM API returns a 401 Unauthorized**
Check that `GROQ_API_KEY` is set correctly in `.env` and that the file is being loaded (the app uses `python-dotenv`).

**LLM says "I could not find this in the provided documents"**
Either the information genuinely isn't in your documents, or the relevant chunk scored below the retrieval threshold. Try increasing `top_k` in your query, or check `python cli.py sources` to confirm the correct document was ingested.

**Retrieval returns irrelevant chunks**
Experiment with chunking strategies. If your documents have dense paragraphs, try `--strategy sentence`. If they are well-structured with section headers, `recursive` is likely already optimal. You can also increase `CHUNK_OVERLAP` to reduce the chance of splitting a key fact across chunk boundaries.

---

## License

MIT