"""
Tests for the FastAPI REST API.
"""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blob_storage.api import app
from blob_storage.storage import LocalFilesystemStorage


@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage instance."""
    return LocalFilesystemStorage(
        base_path=str(tmp_path / "storage"),
        default_container="test-container",
    )


@pytest.fixture
def client(temp_storage, monkeypatch):
    """Create a test client with temporary storage."""
    # Patch the get_storage function to return our temp storage
    import blob_storage.api as api_module
    
    def mock_get_storage():
        return temp_storage
    
    monkeypatch.setattr(api_module, "get_storage", mock_get_storage)
    monkeypatch.setattr(api_module, "_storage", temp_storage)
    
    # Also patch config for status endpoint
    monkeypatch.setattr("blob_storage.api.config.BLOB_CONTAINER_NAME", "test-container")
    monkeypatch.setattr("blob_storage.api.config.STORAGE_MODE", "local")
    monkeypatch.setattr("blob_storage.api.config.LOCAL_STORAGE_PATH", str(temp_storage.base_path))
    
    with TestClient(app) as client:
        yield client


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Blob Storage API"
        assert data["status"] == "healthy"
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["storage_connected"] is True
    
    def test_status(self, client):
        """Test status endpoint."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "connected"
        assert data["storage_mode"] == "local"


class TestUploadEndpoints:
    """Tests for file upload endpoints."""
    
    def test_upload_file(self, client):
        """Test single file upload."""
        content = b"Hello, World!"
        files = {"file": ("test.txt", io.BytesIO(content), "text/plain")}
        
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert data["blob_name"] == "test.txt"
        assert data["size"] == len(content)
        assert data["message"] == "Upload successful"
    
    def test_upload_file_with_custom_name(self, client):
        """Test upload with custom blob name."""
        content = b"Test content"
        files = {"file": ("original.txt", io.BytesIO(content), "text/plain")}
        
        response = client.post("/upload?blob_name=custom/path/file.txt", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert data["blob_name"] == "custom/path/file.txt"
    
    def test_upload_multiple_files(self, client):
        """Test multiple file upload."""
        files = [
            ("files", ("file1.txt", io.BytesIO(b"Content 1"), "text/plain")),
            ("files", ("file2.txt", io.BytesIO(b"Content 2"), "text/plain")),
        ]
        
        response = client.post("/upload/multiple", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 2
        assert data[0]["blob_name"] == "file1.txt"
        assert data[1]["blob_name"] == "file2.txt"


class TestListEndpoints:
    """Tests for file listing endpoints."""
    
    def test_list_empty(self, client):
        """Test listing empty container."""
        response = client.get("/files")
        assert response.status_code == 200
        assert response.json() == []
    
    def test_list_files(self, client):
        """Test listing files after upload."""
        # Upload a file first
        files = {"file": ("test.txt", io.BytesIO(b"Hello"), "text/plain")}
        client.post("/upload", files=files)
        
        response = client.get("/files")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test.txt"


class TestDownloadEndpoints:
    """Tests for file download endpoints."""
    
    def test_download_file(self, client):
        """Test file download."""
        content = b"Download test content"
        files = {"file": ("download.txt", io.BytesIO(content), "text/plain")}
        client.post("/upload", files=files)
        
        response = client.get("/files?name=download.txt")
        assert response.status_code == 200
        assert response.content == content
    
    def test_download_nonexistent(self, client):
        """Test downloading non-existent file."""
        response = client.get("/files?name=nonexistent.txt")
        assert response.status_code == 404
    
    def test_get_file_info(self, client):
        """Test getting file info."""
        files = {"file": ("info.txt", io.BytesIO(b"Info test"), "text/plain")}
        client.post("/upload", files=files)
        
        response = client.get("/files/info?name=info.txt")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "info.txt"
        assert data["size"] == 9


class TestDeleteEndpoints:
    """Tests for file deletion endpoints."""
    
    def test_delete_file(self, client):
        """Test file deletion."""
        files = {"file": ("delete.txt", io.BytesIO(b"Delete me"), "text/plain")}
        client.post("/upload", files=files)
        
        response = client.delete("/files?name=delete.txt")
        assert response.status_code == 200
        
        data = response.json()
        assert data["deleted"] is True
        
        # Verify file is deleted
        response = client.get("/files?name=delete.txt")
        assert response.status_code == 404
    
    def test_delete_nonexistent(self, client):
        """Test deleting non-existent file."""
        response = client.delete("/files?name=nonexistent.txt")
        assert response.status_code == 404


class TestContainerEndpoints:
    """Tests for container management endpoints."""
    
    def test_create_container(self, client):
        """Test container creation."""
        response = client.post("/containers/new-container")
        assert response.status_code == 200
        
        data = response.json()
        assert data["container_name"] == "new-container"
    
    def test_list_container_files(self, client):
        """Test listing files in specific container."""
        # Create container and upload file
        client.post("/containers/my-container")
        files = {"file": ("test.txt", io.BytesIO(b"Test"), "text/plain")}
        client.post("/upload?container=my-container", files=files)
        
        response = client.get("/containers/my-container/files")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test.txt"
