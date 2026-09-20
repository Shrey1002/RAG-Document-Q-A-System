from jinja2 import Environment, FileSystemLoader
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from src.pipeline import RAGPipeline
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Document Q&A System")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

_jinja_env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")))

_pipeline = RAGPipeline(config={
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
    "GROQ_BASE_URL": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    "LLM_MODEL": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
    "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", ".chroma_db"),
})

class IngestRequest(BaseModel):
    file_path: str
    chunking_strategy: str = "recursive"

class IngestResponse(BaseModel):
    chunks_created: int
    sources_indexed: int
    time_taken_ms: int

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filter_source: Optional[str] = None
    stream: bool = False

class QueryResponse(BaseModel):
    answer: str
    citations: List[dict]
    model_used: str
    chunks_used: int
    retrieval_time_ms: int
    generation_time_ms: int

class SourcesResponse(BaseModel):
    sources: List[str]
    total: int

class StatsResponse(BaseModel):
    total_chunks: int
    total_sources: int
    embedding_model: str
    llm_model: str
    chroma_persist_dir: str

@app.get("/")
def home(request: Request):
    template = _jinja_env.get_template("index.html")
    return HTMLResponse(content=template.render(request=request), status_code=200)

@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    result = _pipeline.ingest(request.file_path, request.chunking_strategy)
    return result

@app.post("/ingest-multiple")
async def ingest_multiple(request: Request, chunking_strategy: str = "recursive"):
    import tempfile
    form = await request.form()
    files = form.getlist("files")
    total_chunks = 0
    sources = set()
    total_time = 0
    for upload_file in files:
        content = upload_file.file.read()
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = _pipeline.ingest(tmp_path, chunking_strategy)
            total_chunks += result["chunks_created"]
            sources.add(upload_file.filename)
            total_time += result["time_taken_ms"]
        finally:
            os.unlink(tmp_path)
    return {
        "chunks_created": total_chunks,
        "sources_indexed": len(sources),
        "time_taken_ms": total_time
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    answer = _pipeline.query(request.question, top_k=request.top_k, filter_source=request.filter_source)
    return {
        "answer": answer.answer,
        "citations": [c.__dict__ for c in answer.citations],
        "model_used": answer.model_used,
        "chunks_used": answer.chunks_used,
        "retrieval_time_ms": answer.retrieval_time_ms,
        "generation_time_ms": answer.generation_time_ms,
    }

@app.get("/sources", response_model=SourcesResponse)
def sources():
    srcs = _pipeline.list_sources()
    return {"sources": srcs, "total": len(srcs)}

@app.get("/stats", response_model=StatsResponse)
def stats():
    return {
        "total_chunks": _pipeline.count(),
        "total_sources": len(_pipeline.list_sources()),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "llm_model": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        "chroma_persist_dir": os.getenv("CHROMA_PERSIST_DIR", ".chroma_db"),
    }

@app.delete("/clear")
def clear():
    _pipeline.clear()
    return {"message": "Vector store cleared successfully."}