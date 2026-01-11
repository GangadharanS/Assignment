"""
Local Blob Storage - A Python library for uploading documents to local blob storage.

Supports both Azurite (Azure Storage Emulator) and local filesystem storage.
Includes semantic search capabilities with vector embeddings.
"""

from blob_storage.storage import (
    BlobStorageBase,
    AzuriteBlobStorage,
    LocalFilesystemStorage,
    get_blob_storage,
)
from blob_storage.config import config, Config

__version__ = "1.0.0"
__all__ = [
    # Storage
    "BlobStorageBase",
    "AzuriteBlobStorage",
    "LocalFilesystemStorage",
    "get_blob_storage",
    # Config
    "config",
    "Config",
    # API
    "create_app",
    "run_server",
    # Document Processing
    "DocumentProcessor",
    "TextChunker",
    "SemanticChunker",
    "EmbeddingGenerator",
    "VectorStore",
    # LLM & RAG
    "OllamaLLM",
    "RAGPipeline",
]


def create_app():
    """Create and return the FastAPI application."""
    from blob_storage.api import create_app as _create_app
    return _create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    from blob_storage.api import run_server as _run_server
    _run_server(host=host, port=port, reload=reload)


# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "DocumentProcessor":
        from blob_storage.document_processor import DocumentProcessor
        return DocumentProcessor
    elif name == "TextChunker":
        from blob_storage.chunker import TextChunker
        return TextChunker
    elif name == "SemanticChunker":
        from blob_storage.chunker import SemanticChunker
        return SemanticChunker
    elif name == "EmbeddingGenerator":
        from blob_storage.embeddings import EmbeddingGenerator
        return EmbeddingGenerator
    elif name == "VectorStore":
        from blob_storage.vector_store import VectorStore
        return VectorStore
    elif name == "OllamaLLM":
        from blob_storage.llm import OllamaLLM
        return OllamaLLM
    elif name == "RAGPipeline":
        from blob_storage.rag import RAGPipeline
        return RAGPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

