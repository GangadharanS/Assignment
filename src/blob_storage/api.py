"""
FastAPI REST API for blob storage operations.

Provides endpoints for uploading, downloading, and managing documents.
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

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


class DeleteResponse(BaseModel):
    """Response after successful deletion."""

    message: str
    blob_name: str
    deleted: bool


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
):
    """
    Upload a file to blob storage.
    
    - **file**: The file to upload (required)
    - **blob_name**: Custom name for the stored blob (optional)
    - **container**: Target container name (optional)
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

        return UploadResponse(
            blob_name=result["blob_name"],
            container=result["container"],
            size=result["size"],
            etag=result["etag"],
            last_modified=result["last_modified"],
            url=result.get("url"),
            path=result.get("path"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


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

