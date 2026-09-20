import argparse
import asyncio
from src.pipeline import RAGPipeline
from dotenv import load_dotenv
import os

load_dotenv()

def get_pipeline() -> RAGPipeline:
    return RAGPipeline(config={
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        "GROQ_BASE_URL": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "LLM_MODEL": os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", ".chroma_db"),
    })

def cmd_ingest(args):
    pipeline = get_pipeline()
    strategy = args.strategy or "recursive"
    result = pipeline.ingest(args.path, chunking_strategy=strategy)
    print(f"✅ Ingested: {result['chunks_created']} chunks from {result['sources_indexed']} source(s) in {result['time_taken_ms']}ms")

def cmd_query(args):
    pipeline = get_pipeline()
    answer = pipeline.query(args.question, top_k=args.top_k)
    print(f"\n🤖 Answer: {answer.answer}\n")
    print("📚 Citations:")
    for c in answer.citations:
        print(f"   - {c.source} (Page {c.page}, Chunk {c.chunk_index}, Score: {c.relevance_score:.4f})")
    print(f"\n⏱️  Retrieval: {answer.retrieval_time_ms}ms | Generation: {answer.generation_time_ms}ms")

def cmd_sources(args):
    pipeline = get_pipeline()
    srcs = pipeline.list_sources()
    print(f"📄 Indexed sources ({len(srcs)}):")
    for s in srcs:
        print(f"   - {s}")

def cmd_stats(args):
    pipeline = get_pipeline()
    print(f"📊 Stats:")
    print(f"   Total chunks: {pipeline.count()}")
    print(f"   Total sources: {len(pipeline.list_sources())}")
    print(f"   Embedding model: {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}")
    print(f"   LLM model: {os.getenv('LLM_MODEL', 'openai/gpt-oss-20b')}")

def cmd_clear(args):
    pipeline = get_pipeline()
    pipeline.clear()
    print("🗑️  Vector store cleared.")

def main():
    parser = argparse.ArgumentParser(description="RAG Document Q&A System")
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("path", help="Path to file or directory")
    p_ingest.add_argument("--strategy", choices=["fixed", "sentence", "recursive"], default="recursive")

    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--stream", action="store_true")

    p_sources = sub.add_parser("sources", help="List indexed sources")

    p_stats = sub.add_parser("stats", help="Show system stats")

    p_clear = sub.add_parser("clear", help="Clear the vector store")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "sources": cmd_sources,
        "stats": cmd_stats,
        "clear": cmd_clear,
    }
    commands[args.command](args)

if __name__ == "__main__":
    main()