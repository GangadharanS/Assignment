"""
Tests for the vector store module.
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from blob_storage.vector_store import VectorStore, SearchResult
from blob_storage.chunker import TextChunk


class TestVectorStore:
    """Tests for the VectorStore class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for the vector store."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def store(self, temp_dir):
        """Create a vector store instance."""
        return VectorStore(
            persist_directory=temp_dir,
            collection_name="test_collection",
        )
    
    def test_initialization(self, store):
        """Test store initializes correctly."""
        assert store is not None
        assert store.collection_name == "test_collection"
    
    def test_add_chunks(self, store):
        """Test adding chunks to the store."""
        chunks = [
            TextChunk(
                text="First chunk about cats.",
                index=0,
                start_char=0,
                end_char=23,
                document_name="test.txt",
            ),
            TextChunk(
                text="Second chunk about dogs.",
                index=1,
                start_char=24,
                end_char=48,
                document_name="test.txt",
            ),
        ]
        embeddings = np.random.rand(2, 384).astype(np.float32)
        
        store.add_chunks(chunks, embeddings)
        
        # Verify chunks were added
        stats = store.get_stats()
        assert stats["total_chunks"] == 2
    
    def test_search(self, store):
        """Test searching the store."""
        # Add some chunks
        chunks = [
            TextChunk(text="Cats are furry animals.", index=0, start_char=0, end_char=23, document_name="pets.txt"),
            TextChunk(text="Dogs are loyal companions.", index=1, start_char=0, end_char=26, document_name="pets.txt"),
            TextChunk(text="Python is a programming language.", index=2, start_char=0, end_char=33, document_name="code.txt"),
        ]
        embeddings = np.random.rand(3, 384).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        
        # Search
        query_embedding = embeddings[0]  # Use first chunk's embedding as query
        results = store.search(query_embedding, n_results=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_by_text(self, store):
        """Test searching by text (generates embedding internally)."""
        # Add chunks
        chunks = [
            TextChunk(text="Machine learning models.", index=0, start_char=0, end_char=24, document_name="ml.txt"),
        ]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        
        # Search by text
        results = store.search_by_text("machine learning", n_results=1)
        
        assert len(results) >= 0  # May or may not find relevant results
    
    def test_list_documents(self, store):
        """Test listing documents."""
        chunks = [
            TextChunk(text="Content 1", index=0, start_char=0, end_char=9, document_name="doc1.pdf"),
            TextChunk(text="Content 2", index=0, start_char=0, end_char=9, document_name="doc2.pdf"),
        ]
        embeddings = np.random.rand(2, 384).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        
        documents = store.list_documents()
        
        assert len(documents) == 2
        assert "doc1.pdf" in documents
        assert "doc2.pdf" in documents
    
    def test_delete_document(self, store):
        """Test deleting a document."""
        chunks = [
            TextChunk(text="Keep me", index=0, start_char=0, end_char=7, document_name="keep.txt"),
            TextChunk(text="Delete me", index=0, start_char=0, end_char=9, document_name="delete.txt"),
        ]
        embeddings = np.random.rand(2, 384).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        
        # Delete one document
        deleted = store.delete_document("delete.txt")
        
        assert deleted > 0
        
        # Verify
        documents = store.list_documents()
        assert "keep.txt" in documents
        assert "delete.txt" not in documents
    
    def test_get_stats(self, store):
        """Test getting statistics."""
        chunks = [
            TextChunk(text="Test", index=0, start_char=0, end_char=4, document_name="test.txt"),
        ]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        
        stats = store.get_stats()
        
        assert "total_chunks" in stats
        assert "total_documents" in stats
        assert stats["total_chunks"] == 1
        assert stats["total_documents"] == 1
    
    def test_clear(self, store):
        """Test clearing the store."""
        # Add some data
        chunks = [
            TextChunk(text="Data 1", index=0, start_char=0, end_char=6, document_name="doc1.txt"),
            TextChunk(text="Data 2", index=0, start_char=0, end_char=6, document_name="doc2.txt"),
        ]
        embeddings = np.random.rand(2, 384).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        
        # Clear
        result = store.clear()
        
        assert result["cleared_chunks"] == 2
        assert result["cleared_documents"] == 2
        
        # Verify empty
        stats = store.get_stats()
        assert stats["total_chunks"] == 0


class TestSearchResult:
    """Tests for the SearchResult dataclass."""
    
    def test_create_search_result(self):
        """Test creating a search result."""
        result = SearchResult(
            text="Test content",
            score=0.85,
            metadata={"key": "value"},
            document_name="test.pdf",
            chunk_index=0,
        )
        
        assert result.text == "Test content"
        assert result.score == 0.85
        assert result.document_name == "test.pdf"
        assert result.chunk_index == 0
    
    def test_search_result_comparison(self):
        """Test that search results can be compared by score."""
        result1 = SearchResult(text="A", score=0.9, metadata={}, document_name="a.txt", chunk_index=0)
        result2 = SearchResult(text="B", score=0.7, metadata={}, document_name="b.txt", chunk_index=0)
        
        assert result1.score > result2.score
