"""
Local Blob Storage - A Python library for uploading documents to local blob storage.

Supports both Azurite (Azure Storage Emulator) and local filesystem storage.
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
    "BlobStorageBase",
    "AzuriteBlobStorage",
    "LocalFilesystemStorage",
    "get_blob_storage",
    "config",
    "Config",
    "create_app",
    "run_server",
]


def create_app():
    """Create and return the FastAPI application."""
    from blob_storage.api import create_app as _create_app
    return _create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    from blob_storage.api import run_server as _run_server
    _run_server(host=host, port=port, reload=reload)

