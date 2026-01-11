"""
Tests for the embeddings module.
"""
import pytest
import numpy as np

from blob_storage.embeddings import EmbeddingGenerator, get_embedding_generator


class TestEmbeddingGenerator:
    """Tests for the embedding generator."""
    
    @pytest.fixture(scope="class")
    def generator(self):
        """Create an embedding generator (shared across tests for speed)."""
        return EmbeddingGenerator()
    
    def test_initialization(self, generator):
        """Test generator initializes correctly."""
        assert generator is not None
        assert generator.model is not None
    
    def test_embed_single_text(self, generator):
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = generator.embed(text)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 1D array
        assert embedding.shape[0] == 384  # MiniLM dimension
    
    def test_embed_batch(self, generator):
        """Test embedding multiple texts."""
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]
        embeddings = generator.embed_batch(texts)
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
    
    def test_embed_empty_string(self, generator):
        """Test embedding empty string."""
        embedding = generator.embed("")
        
        assert embedding is not None
        assert embedding.shape[0] == 384
    
    def test_embedding_similarity(self, generator):
        """Test that similar texts have similar embeddings."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."
        text3 = "Python is a programming language."
        
        emb1 = generator.embed(text1)
        emb2 = generator.embed(text2)
        emb3 = generator.embed(text3)
        
        # Similar sentences should have higher similarity
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        assert sim_12 > sim_13  # Similar texts more similar
    
    def test_embedding_dimension(self, generator):
        """Test embedding dimension property."""
        assert generator.dimension == 384
    
    def test_get_embedding_generator_singleton(self):
        """Test that get_embedding_generator returns consistent instance."""
        gen1 = get_embedding_generator()
        gen2 = get_embedding_generator()
        
        # Should return the same or equivalent instances
        assert gen1.dimension == gen2.dimension


class TestEmbeddingNormalization:
    """Tests for embedding normalization."""
    
    @pytest.fixture(scope="class")
    def generator(self):
        """Create generator for tests."""
        return EmbeddingGenerator()
    
    def test_embedding_magnitude(self, generator):
        """Test that embeddings have reasonable magnitude."""
        text = "Test sentence for embedding."
        embedding = generator.embed(text)
        
        magnitude = np.linalg.norm(embedding)
        
        # Should be normalized or close to 1
        assert 0.5 < magnitude < 2.0
    
    def test_batch_consistency(self, generator):
        """Test that batch and single embeddings are consistent."""
        text = "Consistent embedding test."
        
        single_emb = generator.embed(text)
        batch_emb = generator.embed_batch([text])[0]
        
        # Should be very similar (may have small floating point differences)
        similarity = np.dot(single_emb, batch_emb) / (
            np.linalg.norm(single_emb) * np.linalg.norm(batch_emb)
        )
        
        assert similarity > 0.99
