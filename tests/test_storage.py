"""
Tests for the blob storage module.
"""
import tempfile
from pathlib import Path

import pytest

from blob_storage import LocalFilesystemStorage, get_blob_storage


class TestLocalFilesystemStorage:
    """Tests for LocalFilesystemStorage class."""
    
    @pytest.fixture
    def storage(self, tmp_path):
        """Create a temporary storage instance."""
        return LocalFilesystemStorage(
            base_path=str(tmp_path),
            default_container="test-container",
        )
    
    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample file for testing."""
        file_path = tmp_path / "sample.txt"
        file_path.write_text("Hello, World!")
        return file_path
    
    def test_create_container(self, storage, tmp_path):
        """Test container creation."""
        result = storage.create_container("new-container")
        assert result is True
        assert (tmp_path / "new-container").exists()
    
    def test_upload_file(self, storage, sample_file):
        """Test file upload."""
        result = storage.upload_file(str(sample_file))
        
        assert result["blob_name"] == "sample.txt"
        assert result["container"] == "test-container"
        assert result["size"] == 13
        assert "etag" in result
        assert "last_modified" in result
    
    def test_upload_file_with_custom_name(self, storage, sample_file):
        """Test file upload with custom blob name."""
        result = storage.upload_file(str(sample_file), blob_name="custom/path/file.txt")
        
        assert result["blob_name"] == "custom/path/file.txt"
    
    def test_upload_data(self, storage):
        """Test raw data upload."""
        data = b"Test data content"
        result = storage.upload_data(data, "test-data.bin")
        
        assert result["blob_name"] == "test-data.bin"
        assert result["size"] == len(data)
    
    def test_download_file(self, storage, sample_file, tmp_path):
        """Test file download."""
        # First upload
        storage.upload_file(str(sample_file))
        
        # Then download
        download_path = tmp_path / "downloaded.txt"
        result = storage.download_file("sample.txt", str(download_path))
        
        assert result is True
        assert download_path.exists()
        assert download_path.read_text() == "Hello, World!"
    
    def test_download_nonexistent_file(self, storage, tmp_path):
        """Test downloading a file that doesn't exist."""
        download_path = tmp_path / "downloaded.txt"
        result = storage.download_file("nonexistent.txt", str(download_path))
        
        assert result is False
    
    def test_list_blobs(self, storage, sample_file):
        """Test listing blobs."""
        # Upload a file first
        storage.upload_file(str(sample_file))
        
        blobs = storage.list_blobs()
        
        assert len(blobs) == 1
        assert blobs[0]["name"] == "sample.txt"
        assert blobs[0]["size"] == 13
    
    def test_list_blobs_empty_container(self, storage):
        """Test listing blobs in empty container."""
        blobs = storage.list_blobs()
        assert blobs == []
    
    def test_delete_blob(self, storage, sample_file):
        """Test blob deletion."""
        storage.upload_file(str(sample_file))
        
        result = storage.delete_blob("sample.txt")
        
        assert result is True
        assert not storage.blob_exists("sample.txt")
    
    def test_delete_nonexistent_blob(self, storage):
        """Test deleting a blob that doesn't exist."""
        result = storage.delete_blob("nonexistent.txt")
        assert result is False
    
    def test_blob_exists(self, storage, sample_file):
        """Test blob existence check."""
        assert storage.blob_exists("sample.txt") is False
        
        storage.upload_file(str(sample_file))
        
        assert storage.blob_exists("sample.txt") is True
    
    def test_upload_file_not_found(self, storage):
        """Test uploading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            storage.upload_file("/nonexistent/path/file.txt")


class TestGetBlobStorage:
    """Tests for the get_blob_storage factory function."""
    
    def test_get_local_storage(self, tmp_path, monkeypatch):
        """Test getting local storage instance."""
        monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
        storage = get_blob_storage("local")
        
        assert isinstance(storage, LocalFilesystemStorage)
    
    def test_invalid_storage_mode(self):
        """Test invalid storage mode."""
        with pytest.raises(ValueError, match="Unknown storage mode"):
            get_blob_storage("invalid")


