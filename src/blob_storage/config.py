"""
Configuration module for blob storage settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for blob storage settings."""
    
    # Storage mode: "azurite" or "local"
    # Default to "local" for easier setup without external dependencies
    STORAGE_MODE: str = os.getenv("STORAGE_MODE", "local")
    
    # Azurite (Azure Storage Emulator) default connection string
    # This is the well-known development storage account
    AZURE_STORAGE_CONNECTION_STRING: str = os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
        "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
        "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
    )
    
    # Default container name for documents
    BLOB_CONTAINER_NAME: str = os.getenv("BLOB_CONTAINER_NAME", "documents")
    
    # Local filesystem storage path (fallback option)
    LOCAL_STORAGE_PATH: str = os.getenv("LOCAL_STORAGE_PATH", "./local_blob_storage")
    
    @classmethod
    def get_local_storage_path(cls) -> Path:
        """Get the local storage path as a Path object."""
        return Path(cls.LOCAL_STORAGE_PATH)


# Singleton instance
config = Config()

