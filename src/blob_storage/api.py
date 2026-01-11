"""
FastAPI REST API for blob storage operations.

Provides endpoints for uploading, downloading, and managing documents.
Includes semantic search capabilities with vector embeddings.
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from blob_storage.config import config
from blob_storage.storage import BlobStorageBase, get_blob_storage


# Pydantic models for request/response
class BlobInfo(BaseModel):
    """Information about a blob."""

    name: str
    size: int
    last_modified: Optional[str] = None
    content_type: Optional[str] = None


class UploadResponse(BaseModel):
    """Response after successful upload."""

    message: str = "Upload successful"
    blob_name: str
    container: str
    size: int
    etag: str
    last_modified: str
    url: Optional[str] = None
    path: Optional[str] = None
    indexed: bool = False
    chunks_count: int = 0
    index_error: Optional[str] = None


class DeleteResponse(BaseModel):
    """Response after successful deletion."""

    message: str
    blob_name: str
    deleted: bool


class IndexResponse(BaseModel):
    """Response after indexing a document."""
    
    message: str
    document_name: str
    chunks_count: int
    indexed: bool


class SearchResultItem(BaseModel):
    """A single search result."""
    
    text: str
    score: float
    document_name: str
    chunk_index: int
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    """Response for search queries."""
    
    query: str
    results: List[SearchResultItem]
    total_results: int


class VectorStoreStats(BaseModel):
    """Vector store statistics."""
    
    total_chunks: int
    total_documents: int
    collection_name: str
    persist_directory: str


class IndexedDocument(BaseModel):
    """Information about an indexed document."""
    
    name: str
    chunk_count: int
    indexed_at: Optional[str] = None


class ChatMessage(BaseModel):
    """A chat message."""
    
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    
    query: str = Field(..., description="User's question")
    n_results: int = Field(3, description="Number of context chunks to retrieve")
    filter_document: Optional[str] = Field(None, description="Filter to specific document")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(512, ge=1, le=4096, description="Max response tokens")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    include_sources: bool = Field(True, description="Include source chunks in response")
    use_reranking: bool = Field(True, description="Enable re-ranking for better results")
    rerank_top_k: Optional[int] = Field(None, description="Number of top results after reranking")


class ChatSource(BaseModel):
    """Source information for a chat response."""
    
    document: str
    chunk_index: int
    text: str
    relevance: float
    original_score: Optional[float] = None
    rerank_score: Optional[float] = None


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    
    answer: str
    query: str
    model: str
    sources: List[ChatSource] = []
    context_chunks: int
    usage: Optional[Dict[str, int]] = None
    reranked: bool = False
    # Guardrails info
    guardrails: Optional["GuardrailInfo"] = None
    citations: Optional[str] = None


class CustomerResponse(BaseModel):
    """
    Optimized response format for customer-facing applications.
    Clean, simple structure without internal details.
    """
    
    answer: str = Field(..., description="The AI-generated answer")
    confidence: str = Field(..., description="Confidence level: high, medium, low")
    sources: List[str] = Field(default=[], description="List of source document names")
    follow_up_questions: List[str] = Field(default=[], description="Suggested follow-up questions")


class GuardrailInfo(BaseModel):
    """Information about guardrail checks."""
    
    query_valid: bool = Field(..., description="Whether the query passed validation")
    query_type: str = Field(..., description="Classified query type")
    confidence: str = Field(..., description="Confidence level in the response")
    has_relevant_context: bool = Field(..., description="Whether relevant context was found")
    warnings: List[str] = Field(default=[], description="Any warnings from guardrails")
    used_fallback: bool = Field(False, description="Whether a fallback response was used")


class MultiTurnChatRequest(BaseModel):
    """Request body for multi-turn chat."""
    
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    n_results: int = Field(5, description="Number of context chunks to retrieve")
    filter_document: Optional[str] = Field(None, description="Filter to specific document")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(1024, ge=1, le=4096, description="Max response tokens")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")


class LLMStatusResponse(BaseModel):
    """Response for LLM status check."""
    
    available: bool
    provider: str
    model: str
    models_list: List[str] = []


class StatusResponse(BaseModel):
    """Storage status response."""

    status: str
    storage_mode: str
    default_container: str
    storage_path: Optional[str] = None
    endpoint: Optional[str] = None
    blob_count: int


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str


class ContainerResponse(BaseModel):
    """Response after container creation."""

    message: str
    container_name: str
    created: bool


# Global storage instance
_storage: Optional[BlobStorageBase] = None


def get_storage() -> BlobStorageBase:
    """Get or create the blob storage instance."""
    global _storage
    if _storage is None:
        storage_mode = os.getenv("STORAGE_MODE", config.STORAGE_MODE)
        _storage = get_blob_storage(storage_mode)
    return _storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: Initialize storage
    get_storage()
    yield
    # Shutdown: Cleanup if needed
    pass


# Create FastAPI app
app = FastAPI(
    title="Blob Storage API",
    description="""
    REST API for uploading and managing documents in local blob storage.
    
    Supports both Azurite (Azure Storage Emulator) and local filesystem storage.
    
    ## Features
    
    * **Upload** - Upload single or multiple files
    * **Download** - Download files by name
    * **List** - List all files in a container
    * **Delete** - Delete files from storage
    * **Containers** - Create and manage containers
    """,
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return {
        "service": "Blob Storage API",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        storage = get_storage()
        blobs = storage.list_blobs()
        return {
            "status": "healthy",
            "storage_connected": True,
            "blob_count": len(blobs),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "storage_connected": False,
            "error": str(e),
        }


@app.get(
    "/status",
    response_model=StatusResponse,
    tags=["Health"],
    summary="Get storage status",
)
async def get_status():
    """Get current storage status and configuration."""
    try:
        storage = get_storage()
        blobs = storage.list_blobs()
        storage_mode = os.getenv("STORAGE_MODE", config.STORAGE_MODE)

        response = StatusResponse(
            status="connected",
            storage_mode=storage_mode,
            default_container=config.BLOB_CONTAINER_NAME,
            blob_count=len(blobs),
        )

        if storage_mode == "local":
            response.storage_path = config.LOCAL_STORAGE_PATH
        else:
            response.endpoint = "http://127.0.0.1:10000"

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Files"],
    summary="Upload a file",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
    },
)
async def upload_file(
    file: UploadFile = File(..., description="File to upload"),
    blob_name: Optional[str] = Query(
        None, description="Custom name for the blob (defaults to filename)"
    ),
    container: Optional[str] = Query(
        None, description="Target container (defaults to configured container)"
    ),
    index: bool = Query(
        True, description="Whether to index the document for semantic search"
    ),
    max_chunk_size: int = Query(
        5000, description="Maximum characters per chunk"
    ),
    min_chunk_size: int = Query(
        1000, description="Minimum characters per chunk"
    ),
    similarity_threshold: float = Query(
        0.8, description="Similarity threshold for semantic chunking (0-1, lower = more splits)", ge=0.0, le=1.0
    ),
    window_size: int = Query(
        3, description="Number of sentences to compare in sliding window", ge=1, le=10
    ),
):
    """
    Upload a file to blob storage with automatic semantic chunking and indexing.
    
    - **file**: The file to upload (required)
    - **blob_name**: Custom name for the stored blob (optional)
    - **container**: Target container name (optional)
    - **index**: Whether to chunk and index for semantic search (default: True)
    - **max_chunk_size**: Maximum characters per chunk (default: 1000)
    - **min_chunk_size**: Minimum characters per chunk (default: 100)
    - **similarity_threshold**: Below this value, text is split (default: 0.5)
    - **window_size**: Sentences to consider together (default: 3)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        storage = get_storage()
        name = blob_name or file.filename

        # Read file content
        content = await file.read()

        # Upload using upload_data method
        result = storage.upload_data(content, name, container_name=container)

        # Index the document if requested
        indexed = False
        chunks_count = 0
        index_error = None
        
        if index:
            try:
                indexed, chunks_count = await _index_document(
                    content=content,
                    document_name=name,
                    max_chunk_size=max_chunk_size,
                    min_chunk_size=min_chunk_size,
                    similarity_threshold=similarity_threshold,
                    window_size=window_size,
                )
            except ImportError as e:
                index_error = f"Missing dependency: {e}. Install with: pip install sentence-transformers chromadb pypdf python-docx"
                print(f"Warning: {index_error}")
            except Exception as e:
                index_error = f"Indexing failed: {str(e)}"
                print(f"Warning: {index_error}")

        return UploadResponse(
            blob_name=result["blob_name"],
            container=result["container"],
            size=result["size"],
            etag=result["etag"],
            last_modified=result["last_modified"],
            url=result.get("url"),
            path=result.get("path"),
            indexed=indexed,
            chunks_count=chunks_count,
            index_error=index_error,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def _index_document(
    content: bytes,
    document_name: str,
    max_chunk_size: int = 5000,
    min_chunk_size: int = 1000,
    similarity_threshold: float = 0.8,
    window_size: int = 3,
) -> tuple:
    """
    Index a document: extract text, chunk using semantic embeddings, and store in vector DB.
    
    Uses SemanticChunker which:
    1. Splits text into sentences
    2. Generates embeddings for each sentence  
    3. Uses sliding window to compare semantic similarity
    4. Splits at points where topic/meaning changes significantly
    
    Args:
        content: Raw file bytes
        document_name: Name of the document
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        similarity_threshold: Threshold below which to split (0-1, lower = more splits)
        window_size: Number of sentences to consider in each window
    
    Returns:
        Tuple of (indexed: bool, chunks_count: int)
    """
    try:
        print(f"[Indexing] Starting to index: {document_name}")
        print(f"[Indexing] Content size: {len(content)} bytes")
        
        from blob_storage.document_processor import document_processor
        from blob_storage.chunker import SemanticChunker
        from blob_storage.embeddings import get_embedding_generator
        from blob_storage.vector_store import get_vector_store
        
        # Extract text from document
        print(f"[Indexing] Extracting text from document...")
        text = document_processor.extract_text(
            file_content=content,
            filename=document_name,
        )
        
        if not text or not text.strip():
            print(f"[Indexing] ERROR: No text extracted from {document_name}")
            return False, 0
        
        print(f"[Indexing] Extracted {len(text)} characters of text")
        
        # Chunk the text using semantic similarity with embeddings
        print(f"[Indexing] Creating SemanticChunker (max_size={max_chunk_size}, threshold={similarity_threshold})...")
        chunker = SemanticChunker(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            similarity_threshold=similarity_threshold,
            window_size=window_size,
        )
        print(f"[Indexing] Running semantic chunking...")
        chunks = chunker.chunk(text, metadata={"source": document_name})
        
        if not chunks:
            print(f"[Indexing] ERROR: No chunks created from {document_name}")
            return False, 0
        
        print(f"[Indexing] Created {len(chunks)} chunks")
        
        # Generate embeddings for chunks
        print(f"[Indexing] Generating embeddings for {len(chunks)} chunks...")
        embedding_gen = get_embedding_generator()
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedding_gen.embed(chunk_texts)
        print(f"[Indexing] Generated embeddings with shape: {embeddings.shape}")
        
        # Store in vector database
        print(f"[Indexing] Storing in vector database...")
        vector_store = get_vector_store()
        
        # Delete existing chunks for this document (update case)
        vector_store.delete_document(document_name)
        
        # Add new chunks
        vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings.tolist(),
            document_name=document_name,
            document_metadata={"size": len(content)},
        )
        
        print(f"[Indexing] SUCCESS: Indexed {len(chunks)} chunks for {document_name}")
        return True, len(chunks)
        
    except ImportError as e:
        print(f"[Indexing] ERROR - Missing dependency: {e}")
        raise
    except Exception as e:
        print(f"[Indexing] ERROR - Exception: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.post(
    "/upload/multiple",
    response_model=List[UploadResponse],
    tags=["Files"],
    summary="Upload multiple files",
)
async def upload_multiple_files(
    files: List[UploadFile] = File(..., description="Files to upload"),
    container: Optional[str] = Query(
        None, description="Target container (defaults to configured container)"
    ),
):
    """
    Upload multiple files to blob storage.
    
    - **files**: List of files to upload (required)
    - **container**: Target container name (optional)
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    storage = get_storage()

    for file in files:
        if not file.filename:
            continue

        try:
            content = await file.read()
            result = storage.upload_data(
                content, file.filename, container_name=container
            )
            results.append(
                UploadResponse(
                    blob_name=result["blob_name"],
                    container=result["container"],
                    size=result["size"],
                    etag=result["etag"],
                    last_modified=result["last_modified"],
                    url=result.get("url"),
                    path=result.get("path"),
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Upload failed for {file.filename}: {str(e)}"
            )

    return results


@app.get(
    "/files",
    tags=["Files"],
    summary="List files or download a specific file",
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
    },
)
async def get_files(
    name: Optional[str] = Query(
        None, description="Name of the blob to download (if provided, downloads the file; otherwise lists all files)"
    ),
    container: Optional[str] = Query(
        None, description="Container name (defaults to configured container)"
    ),
):
    """
    List all files or download a specific file.
    
    - If **name** is provided: Downloads the specified file
    - If **name** is not provided: Lists all files in the container
    
    Examples:
    - List files: GET /files
    - Download file: GET /files?name=document.pdf
    """
    storage = get_storage()
    
    # If name is provided, download the file
    if name:
        try:
            # Check if blob exists
            if not storage.blob_exists(name, container_name=container):
                raise HTTPException(status_code=404, detail=f"File '{name}' not found")

            # For local filesystem storage, we can return the file directly
            from blob_storage.storage import LocalFilesystemStorage

            if isinstance(storage, LocalFilesystemStorage):
                file_path = storage._get_blob_path(name, container)
                if file_path.exists():
                    return FileResponse(
                        path=str(file_path),
                        filename=Path(name).name,
                        media_type=storage._guess_content_type(file_path),
                    )

            # For Azurite storage, download to temp file and stream
            with NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
                success = storage.download_file(
                    name, tmp_path, container_name=container
                )

                if not success:
                    raise HTTPException(
                        status_code=404, detail=f"File '{name}' not found"
                    )

                return FileResponse(
                    path=tmp_path,
                    filename=Path(name).name,
                    media_type="application/octet-stream",
                )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # If name is not provided, list all files
    try:
        blobs = storage.list_blobs(container_name=container)
        return [BlobInfo(**blob) for blob in blobs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/files/info",
    response_model=BlobInfo,
    tags=["Files"],
    summary="Get file information",
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
    },
)
async def get_file_info(
    name: str = Query(..., description="Name of the blob to get info for"),
    container: Optional[str] = Query(
        None, description="Container name (defaults to configured container)"
    ),
):
    """
    Get information about a specific file.
    
    - **name**: Name of the blob (required)
    - **container**: Container name (optional)
    
    Example: GET /files/info?name=document.pdf
    """
    try:
        storage = get_storage()

        if not storage.blob_exists(name, container_name=container):
            raise HTTPException(status_code=404, detail=f"File '{name}' not found")

        # Find the blob in the list
        blobs = storage.list_blobs(container_name=container)
        for blob in blobs:
            if blob["name"] == name:
                return BlobInfo(**blob)

        raise HTTPException(status_code=404, detail=f"File '{name}' not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/files",
    response_model=DeleteResponse,
    tags=["Files"],
    summary="Delete a file",
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
    },
)
async def delete_file(
    name: str = Query(..., description="Name of the blob to delete"),
    container: Optional[str] = Query(
        None, description="Container name (defaults to configured container)"
    ),
):
    """
    Delete a file from blob storage.
    
    - **name**: Name of the blob to delete (required)
    - **container**: Container name (optional)
    
    Example: DELETE /files?name=document.pdf
    """
    try:
        storage = get_storage()

        if not storage.blob_exists(name, container_name=container):
            raise HTTPException(status_code=404, detail=f"File '{name}' not found")

        success = storage.delete_blob(name, container_name=container)

        return DeleteResponse(
            message="File deleted successfully" if success else "Delete failed",
            blob_name=name,
            deleted=success,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/containers/{container_name}",
    response_model=ContainerResponse,
    tags=["Containers"],
    summary="Create a container",
)
async def create_container(container_name: str):
    """
    Create a new container.
    
    - **container_name**: Name of the container to create (required)
    """
    try:
        storage = get_storage()
        success = storage.create_container(container_name)

        return ContainerResponse(
            message=(
                "Container created successfully"
                if success
                else "Container already exists"
            ),
            container_name=container_name,
            created=success,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/containers/{container_name}/files",
    response_model=List[BlobInfo],
    tags=["Containers"],
    summary="List files in container",
)
async def list_container_files(container_name: str):
    """
    List all files in a specific container.
    
    - **container_name**: Name of the container (required)
    """
    try:
        storage = get_storage()
        blobs = storage.list_blobs(container_name=container_name)
        return [BlobInfo(**blob) for blob in blobs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Search & Indexing Endpoints
# ============================================================================

@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Semantic search across documents",
)
async def search_documents(
    query: str = Query(..., description="Search query text"),
    n_results: int = Query(5, description="Number of results to return", ge=1, le=50),
    document: Optional[str] = Query(None, description="Filter by document name"),
):
    """
    Perform semantic search across indexed documents.
    
    - **query**: The search query text (required)
    - **n_results**: Number of results to return (default: 5, max: 50)
    - **document**: Optional filter to search within a specific document
    
    Example: POST /search?query=What is machine learning?&n_results=10
    """
    try:
        from blob_storage.embeddings import get_embedding_generator
        from blob_storage.vector_store import get_vector_store
        
        embedding_gen = get_embedding_generator()
        vector_store = get_vector_store()
        
        results = vector_store.search_by_text(
            query_text=query,
            embedding_generator=embedding_gen,
            n_results=n_results,
            filter_document=document,
        )
        
        return SearchResponse(
            query=query,
            results=[
                SearchResultItem(
                    text=r.text,
                    score=r.score,
                    document_name=r.document_name,
                    chunk_index=r.chunk_index,
                    metadata=r.metadata,
                )
                for r in results
            ],
            total_results=len(results),
        )
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search dependencies not installed: {e}. "
                   "Install with: pip install sentence-transformers chromadb"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/index",
    response_model=IndexResponse,
    tags=["Search"],
    summary="Index an existing document",
)
async def index_document(
    name: str = Query(..., description="Name of the document to index"),
    container: Optional[str] = Query(None, description="Container name"),
    max_chunk_size: int = Query(1000, description="Maximum characters per chunk"),
    min_chunk_size: int = Query(100, description="Minimum characters per chunk"),
    similarity_threshold: float = Query(
        0.5, description="Similarity threshold (0-1, lower = more splits)", ge=0.0, le=1.0
    ),
    window_size: int = Query(3, description="Sentences to compare in sliding window", ge=1, le=10),
):
    """
    Index an existing document in blob storage for semantic search.
    
    Uses SemanticChunker with embedding-based similarity detection.
    
    - **name**: Name of the document to index (required)
    - **container**: Container name (optional)
    - **max_chunk_size**: Maximum characters per chunk (default: 1000)
    - **min_chunk_size**: Minimum characters per chunk (default: 100)
    - **similarity_threshold**: Below this value, text is split (default: 0.5)
    - **window_size**: Sentences to consider together (default: 3)
    """
    try:
        storage = get_storage()
        
        if not storage.blob_exists(name, container_name=container):
            raise HTTPException(status_code=404, detail=f"File '{name}' not found")
        
        # Download the file content
        from blob_storage.storage import LocalFilesystemStorage
        
        if isinstance(storage, LocalFilesystemStorage):
            file_path = storage._get_blob_path(name, container)
            with open(file_path, "rb") as f:
                content = f.read()
        else:
            # For Azurite, download to memory
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(delete=False) as tmp:
                storage.download_file(name, tmp.name, container_name=container)
                with open(tmp.name, "rb") as f:
                    content = f.read()
        
        # Index the document using semantic chunking
        indexed, chunks_count = await _index_document(
            content=content,
            document_name=name,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            similarity_threshold=similarity_threshold,
            window_size=window_size,
        )
        
        return IndexResponse(
            message="Document indexed successfully" if indexed else "Indexing failed",
            document_name=name,
            chunks_count=chunks_count,
            indexed=indexed,
        )
        
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing dependencies not installed: {e}. "
                   "Install with: pip install sentence-transformers chromadb pypdf python-docx"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/index",
    response_model=List[IndexedDocument],
    tags=["Search"],
    summary="List indexed documents",
)
async def list_indexed_documents():
    """List all documents that have been indexed for semantic search."""
    try:
        from blob_storage.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        documents = vector_store.list_documents()
        
        return [
            IndexedDocument(
                name=doc["name"],
                chunk_count=doc["chunk_count"],
                indexed_at=doc.get("indexed_at"),
            )
            for doc in documents
        ]
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vector store not available: {e}. "
                   "Install with: pip install chromadb"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/index",
    response_model=DeleteResponse,
    tags=["Search"],
    summary="Remove document from index",
)
async def remove_from_index(
    name: str = Query(..., description="Name of the document to remove from index"),
):
    """
    Remove a document from the search index.
    
    - **name**: Name of the document to remove (required)
    
    Note: This only removes from the search index, not from blob storage.
    """
    try:
        from blob_storage.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        count = vector_store.delete_document(name)
        
        return DeleteResponse(
            message=f"Removed {count} chunks from index" if count > 0 else "Document not found in index",
            blob_name=name,
            deleted=count > 0,
        )
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vector store not available: {e}. "
                   "Install with: pip install chromadb"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/index/stats",
    response_model=VectorStoreStats,
    tags=["Search"],
    summary="Get vector store statistics",
)
async def get_index_stats():
    """Get statistics about the vector store and indexed documents."""
    try:
        from blob_storage.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        stats = vector_store.get_stats()
        
        return VectorStoreStats(**stats)
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vector store not available: {e}. "
                   "Install with: pip install chromadb"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/index/clear",
    tags=["Search"],
    summary="Clear all documents from the index",
)
async def clear_index():
    """
    Clear all documents and chunks from the vector store.
    
    **Warning:** This will delete all indexed data. Use with caution.
    """
    try:
        from blob_storage.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        result = vector_store.clear()
        
        return result
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vector store not available: {e}. "
                   "Install with: pip install chromadb"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Chat / RAG Endpoints
# =============================================================================

@app.get(
    "/llm/status",
    response_model=LLMStatusResponse,
    tags=["Chat"],
    summary="Check LLM availability",
)
async def check_llm_status():
    """Check if the LLM (Ollama) is available and list models."""
    try:
        from blob_storage.llm import OllamaLLM
        
        llm = OllamaLLM()
        available = llm.is_available()
        models = llm.list_models() if available else []
        
        return LLMStatusResponse(
            available=available,
            provider="ollama",
            model=llm.model,
            models_list=models,
        )
    except Exception as e:
        return LLMStatusResponse(
            available=False,
            provider="ollama",
            model="unknown",
            models_list=[],
        )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Chat with documents using RAG",
)
async def chat_with_documents(request: ChatRequest):
    """
    Ask questions about your documents using RAG (Retrieval-Augmented Generation).
    
    This endpoint includes **guardrails** for safety and quality:
    - Query validation (rejects invalid/too-short queries)
    - Relevance checking (verifies context quality)
    - Fallback responses (handles no-context scenarios gracefully)
    - Source attribution (adds citations)
    - Confidence scoring (high/medium/low/none)
    
    **Requirements:**
    - Documents must be indexed first (use /upload with index=true or POST /index)
    - Ollama must be running locally (https://ollama.ai)
    - A model must be available (e.g., `ollama pull llama3.2`)
    """
    try:
        from blob_storage.guardrails import (
            FallbackResponses,
            RAGGuardrails,
            SourceAttributionGuardrails,
        )
        from blob_storage.llm import DEFAULT_RAG_SYSTEM_PROMPT, OllamaLLM
        from blob_storage.rag import RAGPipeline
        
        # Initialize guardrails
        guardrails = RAGGuardrails()
        
        # Step 1: Validate query with guardrails
        query_check = guardrails.check_query(request.query)
        
        # Handle invalid queries
        if not query_check.is_valid:
            return ChatResponse(
                answer=query_check.issues[0] if query_check.issues else "Please provide a valid question.",
                query=request.query,
                model="guardrails",
                sources=[],
                context_chunks=0,
                guardrails=GuardrailInfo(
                    query_valid=False,
                    query_type=query_check.query_type.value,
                    confidence="none",
                    has_relevant_context=False,
                    warnings=query_check.issues,
                    used_fallback=True,
                ),
            )
        
        # Handle greetings
        if query_check.query_type.value == "greeting":
            return ChatResponse(
                answer=FallbackResponses.GREETING_RESPONSE,
                query=request.query,
                model="guardrails",
                sources=[],
                context_chunks=0,
                guardrails=GuardrailInfo(
                    query_valid=True,
                    query_type="greeting",
                    confidence="high",
                    has_relevant_context=False,
                    warnings=[],
                    used_fallback=True,
                ),
            )
        
        # Check if LLM is available
        llm = OllamaLLM()
        if not llm.is_available():
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Please start Ollama: https://ollama.ai"
            )
        
        # Create RAG pipeline with custom system prompt if provided
        system_prompt = request.system_prompt or DEFAULT_RAG_SYSTEM_PROMPT
        rag = RAGPipeline(
            llm=llm, 
            system_prompt=system_prompt,
            enable_reranking=request.use_reranking,
        )
        
        # Query the RAG pipeline with reranking
        response = await rag.aquery(
            query=query_check.cleaned_query,
            n_results=request.n_results,
            filter_document=request.filter_document,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_reranking=request.use_reranking,
            rerank_top_k=request.rerank_top_k,
        )
        
        # Step 2: Check relevance with guardrails
        relevance_check = guardrails.check_relevance(response.sources, request.query)
        
        warnings = list(relevance_check.issues)
        used_fallback = False
        final_answer = response.answer
        
        # Handle low/no relevance scenarios
        if not relevance_check.has_relevant_context:
            final_answer = FallbackResponses.NO_CONTEXT
            used_fallback = True
        elif relevance_check.confidence.value == "low":
            final_answer = FallbackResponses.LOW_CONFIDENCE + response.answer
            warnings.append("Response may be less reliable due to low context relevance.")
        
        # Add query type warning if ambiguous
        if query_check.query_type.value == "ambiguous":
            warnings.insert(0, "Query may be ambiguous - consider being more specific.")
        
        # Step 3: Build sources
        sources = []
        if request.include_sources and not used_fallback:
            sources = [
                ChatSource(
                    document=s["document"],
                    chunk_index=s["chunk_index"],
                    text=s["text"],
                    relevance=s["relevance"],
                    original_score=s.get("original_score"),
                    rerank_score=s.get("rerank_score"),
                )
                for s in response.sources
            ]
        
        # Step 4: Add citations
        citations = None
        if request.include_sources and response.sources and not used_fallback:
            citations = SourceAttributionGuardrails.format_citations(response.sources)
        
        return ChatResponse(
            answer=final_answer,
            query=query_check.cleaned_query,
            model=response.model if not used_fallback else "guardrails",
            sources=sources,
            context_chunks=len(response.sources) if not used_fallback else 0,
            usage=response.usage if not used_fallback else None,
            reranked=response.reranked if not used_fallback else False,
            guardrails=GuardrailInfo(
                query_valid=query_check.is_valid,
                query_type=query_check.query_type.value,
                confidence=relevance_check.confidence.value,
                has_relevant_context=relevance_check.has_relevant_context,
                warnings=warnings,
                used_fallback=used_fallback,
            ),
            citations=citations,
        )
        
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Missing dependencies: {e}. "
                   "Install with: pip install chromadb sentence-transformers"
        )
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[Chat Error] {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)


@app.post(
    "/chat/conversation",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Multi-turn conversation with documents",
)
async def multi_turn_chat(request: MultiTurnChatRequest):
    """
    Have a multi-turn conversation about your documents.
    
    Send conversation history and get contextual responses.
    The last user message is used to retrieve relevant chunks.
    """
    try:
        from blob_storage.llm import DEFAULT_RAG_SYSTEM_PROMPT, OllamaLLM
        from blob_storage.rag import RAGPipeline
        
        # Check if LLM is available
        llm = OllamaLLM()
        if not llm.is_available():
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Please start Ollama: https://ollama.ai"
            )
        
        # Create RAG pipeline
        system_prompt = request.system_prompt or DEFAULT_RAG_SYSTEM_PROMPT
        rag = RAGPipeline(llm=llm, system_prompt=system_prompt)
        
        # Convert messages
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
        ]
        
        # Query with conversation history
        response = rag.chat(
            messages=messages,
            n_results=request.n_results,
            filter_document=request.filter_document,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        # Build response
        sources = [
            ChatSource(
                document=s["document"],
                chunk_index=s["chunk_index"],
                text=s["text"],
                relevance=s["relevance"],
            )
            for s in response.sources
        ]
        
        return ChatResponse(
            answer=response.answer,
            query=response.query,
            model=response.model,
            sources=sources,
            context_chunks=response.context_chunks,
            usage=response.usage,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# System prompt for generating follow-up questions
FOLLOW_UP_SYSTEM_PROMPT = """Based on the conversation, suggest 2-3 brief follow-up questions the user might ask. 
Return ONLY the questions, one per line, no numbering or bullets."""


@app.post(
    "/chat/customer",
    response_model=CustomerResponse,
    tags=["Chat"],
    summary="Customer-optimized chat response",
)
async def customer_chat(request: ChatRequest):
    """
    Get a customer-friendly response optimized for end-user display.
    
    This endpoint provides:
    - Clean, simple response format
    - Confidence level (high/medium/low based on relevance scores)
    - Source document names only (no technical details)
    - Suggested follow-up questions
    
    Use this for customer-facing chatbot interfaces.
    """
    try:
        from blob_storage.llm import DEFAULT_RAG_SYSTEM_PROMPT, OllamaLLM
        from blob_storage.rag import RAGPipeline
        
        # Check if LLM is available
        llm = OllamaLLM()
        if not llm.is_available():
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Please start Ollama: https://ollama.ai"
            )
        
        # Create RAG pipeline with reranking enabled for best results
        system_prompt = request.system_prompt or DEFAULT_RAG_SYSTEM_PROMPT
        rag = RAGPipeline(
            llm=llm, 
            system_prompt=system_prompt,
            enable_reranking=True,
        )
        
        # Query with reranking
        response = await rag.aquery(
            query=request.query,
            n_results=request.n_results,
            filter_document=request.filter_document,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_reranking=True,
        )
        
        # Calculate confidence based on source relevance scores
        if response.sources:
            avg_relevance = sum(s["relevance"] for s in response.sources) / len(response.sources)
            top_relevance = response.sources[0]["relevance"] if response.sources else 0
            
            if top_relevance >= 0.7 and avg_relevance >= 0.5:
                confidence = "high"
            elif top_relevance >= 0.4 or avg_relevance >= 0.3:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "low"
        
        # Extract unique source document names
        source_docs = list(set(s["document"] for s in response.sources))
        
        # Generate follow-up questions
        follow_up_questions = []
        try:
            follow_up_response = llm.generate(
                prompt=f"User asked: {request.query}\n\nAssistant answered: {response.answer[:500]}",
                system_prompt=FOLLOW_UP_SYSTEM_PROMPT,
                temperature=0.8,
                max_tokens=150,
            )
            # Parse questions from response
            lines = follow_up_response.content.strip().split('\n')
            follow_up_questions = [
                line.strip().lstrip('0123456789.-) ')
                for line in lines 
                if line.strip() and '?' in line
            ][:3]
        except Exception:
            # If follow-up generation fails, continue without them
            pass
        
        return CustomerResponse(
            answer=response.answer,
            confidence=confidence,
            sources=source_docs,
            follow_up_questions=follow_up_questions,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server using uvicorn."""
    import uvicorn

    uvicorn.run(
        "blob_storage.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()

