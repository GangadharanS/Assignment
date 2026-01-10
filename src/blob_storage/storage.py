"""
Blob Storage module for handling document uploads to local blob storage.
Supports both Azurite (Azure Storage Emulator) and local filesystem storage.
"""
import hashlib
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContainerClient
from tqdm import tqdm

from blob_storage.config import config


class BlobStorageBase(ABC):
    """Abstract base class for blob storage operations."""
    
    @abstractmethod
    def create_container(self, container_name: str) -> bool:
        """Create a container/bucket for storing blobs."""
        pass
    
    @abstractmethod
    def upload_file(
        self,
        file_path: str,
        blob_name: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> Dict:
        """Upload a file to blob storage."""
        pass
    
    @abstractmethod
    def upload_data(
        self,
        data: bytes,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> Dict:
        """Upload raw bytes to blob storage."""
        pass
    
    @abstractmethod
    def download_file(
        self,
        blob_name: str,
        destination_path: str,
        container_name: Optional[str] = None,
    ) -> bool:
        """Download a blob to a local file."""
        pass
    
    @abstractmethod
    def list_blobs(self, container_name: Optional[str] = None) -> List[Dict]:
        """List all blobs in a container."""
        pass
    
    @abstractmethod
    def delete_blob(
        self,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> bool:
        """Delete a blob from storage."""
        pass
    
    @abstractmethod
    def blob_exists(
        self,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> bool:
        """Check if a blob exists."""
        pass


class AzuriteBlobStorage(BlobStorageBase):
    """
    Blob storage implementation using Azurite (Azure Storage Emulator).
    
    Azurite provides a local Azure Blob Storage emulator for development.
    Install via: npm install -g azurite
    Run via: azurite --silent --location ./azurite-data --debug ./azurite-debug.log
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        default_container: Optional[str] = None,
    ):
        """Initialize the Azure Blob Storage client."""
        self.connection_string = connection_string or config.AZURE_STORAGE_CONNECTION_STRING
        self.default_container = default_container or config.BLOB_CONTAINER_NAME
        
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        
        # Ensure default container exists
        self.create_container(self.default_container)
    
    def create_container(self, container_name: str) -> bool:
        """Create a container if it doesn't exist."""
        try:
            self.blob_service_client.create_container(container_name)
            print(f"✓ Container '{container_name}' created successfully")
            return True
        except ResourceExistsError:
            # Container already exists
            return True
        except Exception as e:
            print(f"✗ Error creating container: {e}")
            return False
    
    def _get_container_client(
        self,
        container_name: Optional[str] = None,
    ) -> ContainerClient:
        """Get a container client."""
        container = container_name or self.default_container
        return self.blob_service_client.get_container_client(container)
    
    def upload_file(
        self,
        file_path: str,
        blob_name: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> Dict:
        """
        Upload a file to blob storage.
        
        Args:
            file_path: Path to the local file to upload
            blob_name: Name for the blob (defaults to filename)
            container_name: Target container (defaults to default container)
            
        Returns:
            Dict with upload details
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        blob_name = blob_name or file_path.name
        container = container_name or self.default_container
        
        blob_client = self.blob_service_client.get_blob_client(
            container=container,
            blob=blob_name,
        )
        
        file_size = file_path.stat().st_size
        
        with open(file_path, "rb") as data:
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Uploading {blob_name}",
            ) as pbar:
                blob_client.upload_blob(data, overwrite=True)
                pbar.update(file_size)
        
        # Get blob properties
        properties = blob_client.get_blob_properties()
        
        return {
            "blob_name": blob_name,
            "container": container,
            "size": file_size,
            "etag": properties.etag,
            "last_modified": properties.last_modified.isoformat(),
            "url": blob_client.url,
        }
    
    def upload_data(
        self,
        data: bytes,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> Dict:
        """Upload raw bytes to blob storage."""
        container = container_name or self.default_container
        
        blob_client = self.blob_service_client.get_blob_client(
            container=container,
            blob=blob_name,
        )
        
        blob_client.upload_blob(data, overwrite=True)
        properties = blob_client.get_blob_properties()
        
        return {
            "blob_name": blob_name,
            "container": container,
            "size": len(data),
            "etag": properties.etag,
            "last_modified": properties.last_modified.isoformat(),
            "url": blob_client.url,
        }
    
    def download_file(
        self,
        blob_name: str,
        destination_path: str,
        container_name: Optional[str] = None,
    ) -> bool:
        """Download a blob to a local file."""
        container = container_name or self.default_container
        
        blob_client = self.blob_service_client.get_blob_client(
            container=container,
            blob=blob_name,
        )
        
        try:
            with open(destination_path, "wb") as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())
            print(f"✓ Downloaded '{blob_name}' to '{destination_path}'")
            return True
        except ResourceNotFoundError:
            print(f"✗ Blob '{blob_name}' not found")
            return False
    
    def list_blobs(self, container_name: Optional[str] = None) -> List[Dict]:
        """List all blobs in a container."""
        container_client = self._get_container_client(container_name)
        
        blobs = []
        for blob in container_client.list_blobs():
            blobs.append({
                "name": blob.name,
                "size": blob.size,
                "last_modified": (
                    blob.last_modified.isoformat() if blob.last_modified else None
                ),
                "content_type": (
                    blob.content_settings.content_type
                    if blob.content_settings
                    else None
                ),
            })
        return blobs
    
    def delete_blob(
        self,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> bool:
        """Delete a blob from storage."""
        container = container_name or self.default_container
        
        blob_client = self.blob_service_client.get_blob_client(
            container=container,
            blob=blob_name,
        )
        
        try:
            blob_client.delete_blob()
            print(f"✓ Deleted blob '{blob_name}'")
            return True
        except ResourceNotFoundError:
            print(f"✗ Blob '{blob_name}' not found")
            return False
    
    def blob_exists(
        self,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> bool:
        """Check if a blob exists."""
        container = container_name or self.default_container
        
        blob_client = self.blob_service_client.get_blob_client(
            container=container,
            blob=blob_name,
        )
        
        return blob_client.exists()


class LocalFilesystemStorage(BlobStorageBase):
    """
    Blob storage implementation using local filesystem.
    
    This is a fallback option when Azurite is not available.
    Mimics blob storage structure using directories and files.
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        default_container: Optional[str] = None,
    ):
        """Initialize local filesystem storage."""
        self.base_path = Path(base_path or config.LOCAL_STORAGE_PATH)
        self.default_container = default_container or config.BLOB_CONTAINER_NAME
        
        # Create base storage directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure default container exists
        self.create_container(self.default_container)
    
    def _get_container_path(self, container_name: Optional[str] = None) -> Path:
        """Get the filesystem path for a container."""
        container = container_name or self.default_container
        return self.base_path / container
    
    def _get_blob_path(
        self,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> Path:
        """Get the filesystem path for a blob."""
        return self._get_container_path(container_name) / blob_name
    
    def create_container(self, container_name: str) -> bool:
        """Create a container (directory)."""
        container_path = self._get_container_path(container_name)
        try:
            container_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Container '{container_name}' ready at {container_path}")
            return True
        except Exception as e:
            print(f"✗ Error creating container: {e}")
            return False
    
    def upload_file(
        self,
        file_path: str,
        blob_name: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> Dict:
        """Upload a file by copying it to local storage."""
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        blob_name = blob_name or source_path.name
        dest_path = self._get_blob_path(blob_name, container_name)
        
        # Create parent directories if blob_name contains path separators
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_size = source_path.stat().st_size
        
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Uploading {blob_name}",
        ) as pbar:
            shutil.copy2(source_path, dest_path)
            pbar.update(file_size)
        
        # Calculate MD5 hash
        md5_hash = hashlib.md5(dest_path.read_bytes()).hexdigest()
        
        return {
            "blob_name": blob_name,
            "container": container_name or self.default_container,
            "size": file_size,
            "etag": f'"{md5_hash}"',
            "last_modified": datetime.fromtimestamp(
                dest_path.stat().st_mtime
            ).isoformat(),
            "path": str(dest_path),
        }
    
    def upload_data(
        self,
        data: bytes,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> Dict:
        """Upload raw bytes to local storage."""
        dest_path = self._get_blob_path(blob_name, container_name)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        dest_path.write_bytes(data)
        
        md5_hash = hashlib.md5(data).hexdigest()
        
        return {
            "blob_name": blob_name,
            "container": container_name or self.default_container,
            "size": len(data),
            "etag": f'"{md5_hash}"',
            "last_modified": datetime.now().isoformat(),
            "path": str(dest_path),
        }
    
    def download_file(
        self,
        blob_name: str,
        destination_path: str,
        container_name: Optional[str] = None,
    ) -> bool:
        """Download (copy) a blob to destination."""
        source_path = self._get_blob_path(blob_name, container_name)
        
        if not source_path.exists():
            print(f"✗ Blob '{blob_name}' not found")
            return False
        
        shutil.copy2(source_path, destination_path)
        print(f"✓ Downloaded '{blob_name}' to '{destination_path}'")
        return True
    
    def list_blobs(self, container_name: Optional[str] = None) -> List[Dict]:
        """List all blobs in a container."""
        container_path = self._get_container_path(container_name)
        
        if not container_path.exists():
            return []
        
        blobs = []
        for file_path in container_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(container_path)
                stat = file_path.stat()
                blobs.append({
                    "name": str(relative_path),
                    "size": stat.st_size,
                    "last_modified": datetime.fromtimestamp(
                        stat.st_mtime
                    ).isoformat(),
                    "content_type": self._guess_content_type(file_path),
                })
        return blobs
    
    def delete_blob(
        self,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> bool:
        """Delete a blob from storage."""
        blob_path = self._get_blob_path(blob_name, container_name)
        
        if not blob_path.exists():
            print(f"✗ Blob '{blob_name}' not found")
            return False
        
        blob_path.unlink()
        print(f"✓ Deleted blob '{blob_name}'")
        return True
    
    def blob_exists(
        self,
        blob_name: str,
        container_name: Optional[str] = None,
    ) -> bool:
        """Check if a blob exists."""
        return self._get_blob_path(blob_name, container_name).exists()
    
    @staticmethod
    def _guess_content_type(file_path: Path) -> str:
        """Guess content type based on file extension."""
        content_types = {
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": (
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ),
            ".txt": "text/plain",
            ".json": "application/json",
            ".xml": "application/xml",
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".csv": "text/csv",
            ".xlsx": (
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            ),
        }
        return content_types.get(file_path.suffix.lower(), "application/octet-stream")


def get_blob_storage(storage_mode: Optional[str] = None) -> BlobStorageBase:
    """
    Factory function to get the appropriate blob storage implementation.
    
    Args:
        storage_mode: "azurite" or "local" (defaults to config value)
        
    Returns:
        Appropriate BlobStorageBase implementation
    """
    mode = storage_mode or config.STORAGE_MODE
    
    if mode.lower() == "azurite":
        return AzuriteBlobStorage()
    elif mode.lower() == "local":
        return LocalFilesystemStorage()
    else:
        raise ValueError(f"Unknown storage mode: {mode}. Use 'azurite' or 'local'")

