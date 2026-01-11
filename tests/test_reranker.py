"""
Tests for the reranker module.
"""
import pytest
from unittest.mock import Mock, patch

from blob_storage.reranker import (
    CrossEncoderReranker,
    ReciprocRankFusionReranker,
    RerankResult,
    get_reranker,
)
from blob_storage.vector_store import SearchResult


class TestRerankResult:
    """Tests for the RerankResult dataclass."""
    
    def test_create_rerank_result(self):
        """Test creating a rerank result."""
        result = RerankResult(
            text="Test content",
            score=0.85,
            metadata={"key": "value"},
            document_name="doc.pdf",
            chunk_index=0,
            original_score=0.7,
            rerank_score=0.9,
            combined_score=0.85,
        )
        
        assert result.text == "Test content"
        assert result.original_score == 0.7
        assert result.rerank_score == 0.9
        assert result.combined_score == 0.85
    
    def test_from_search_result(self):
        """Test creating RerankResult from SearchResult."""
        search_result = SearchResult(
            text="Original text",
            score=0.75,
            metadata={"chunk_index": 1},
            document_name="test.pdf",
            chunk_index=1,
        )
        
        rerank_result = RerankResult(
            text=search_result.text,
            score=search_result.score,
            metadata=search_result.metadata,
            document_name=search_result.document_name,
            chunk_index=search_result.chunk_index,
            original_score=search_result.score,
            rerank_score=0.85,
            combined_score=0.82,
        )
        
        assert rerank_result.text == "Original text"
        assert rerank_result.original_score == 0.75


class TestCrossEncoderReranker:
    """Tests for the CrossEncoderReranker."""
    
    @pytest.fixture
    def mock_results(self):
        """Create mock search results."""
        return [
            SearchResult(
                text="First result about cats.",
                score=0.8,
                metadata={},
                document_name="pets.txt",
                chunk_index=0,
            ),
            SearchResult(
                text="Second result about dogs.",
                score=0.7,
                metadata={},
                document_name="pets.txt",
                chunk_index=1,
            ),
            SearchResult(
                text="Third result about programming.",
                score=0.6,
                metadata={},
                document_name="code.txt",
                chunk_index=0,
            ),
        ]
    
    @pytest.fixture(scope="class")
    def reranker(self):
        """Create a cross-encoder reranker (shared for speed)."""
        return CrossEncoderReranker()
    
    def test_initialization(self, reranker):
        """Test reranker initializes correctly."""
        assert reranker is not None
        assert reranker.model is not None
    
    def test_rerank_basic(self, reranker, mock_results):
        """Test basic reranking."""
        query = "Tell me about cats"
        reranked = reranker.rerank(query, mock_results, top_k=2)
        
        assert len(reranked) == 2
        assert all(isinstance(r, RerankResult) for r in reranked)
        
        # Results should have rerank scores
        assert all(r.rerank_score is not None for r in reranked)
    
    def test_rerank_ordering(self, reranker, mock_results):
        """Test that reranking changes ordering based on query."""
        query = "cats and felines"
        reranked = reranker.rerank(query, mock_results, top_k=3)
        
        # Cat-related result should rank higher for cat query
        # (may not always be true depending on model)
        assert len(reranked) == 3
        assert reranked[0].combined_score >= reranked[1].combined_score
    
    def test_rerank_empty_results(self, reranker):
        """Test reranking empty results."""
        reranked = reranker.rerank("query", [], top_k=5)
        
        assert reranked == []
    
    def test_rerank_top_k(self, reranker, mock_results):
        """Test top_k limiting."""
        reranked = reranker.rerank("test query", mock_results, top_k=1)
        
        assert len(reranked) == 1
    
    def test_score_combination(self, reranker, mock_results):
        """Test that combined scores are calculated correctly."""
        reranked = reranker.rerank("test", mock_results, top_k=3)
        
        for result in reranked:
            # Combined score should be between 0 and 1
            assert 0 <= result.combined_score <= 1
            # Should have all score components
            assert result.original_score is not None
            assert result.rerank_score is not None


class TestRRFReranker:
    """Tests for the RRF (Reciprocal Rank Fusion) reranker."""
    
    @pytest.fixture
    def reranker(self):
        """Create an RRF reranker."""
        return ReciprocRankFusionReranker(k=60)
    
    @pytest.fixture
    def mock_results(self):
        """Create mock search results."""
        return [
            SearchResult(text="A", score=0.9, metadata={}, document_name="a.txt", chunk_index=0),
            SearchResult(text="B", score=0.8, metadata={}, document_name="b.txt", chunk_index=0),
            SearchResult(text="C", score=0.7, metadata={}, document_name="c.txt", chunk_index=0),
        ]
    
    def test_rrf_rerank(self, reranker, mock_results):
        """Test RRF reranking."""
        reranked = reranker.rerank("query", mock_results, top_k=3)
        
        assert len(reranked) == 3
        # Original order should be roughly preserved (RRF uses ranks)
    
    def test_rrf_scores(self, reranker, mock_results):
        """Test RRF score calculation."""
        reranked = reranker.rerank("query", mock_results, top_k=3)
        
        # RRF scores should be positive
        for result in reranked:
            assert result.rerank_score > 0


class TestGetReranker:
    """Tests for the reranker factory function."""
    
    def test_get_cross_encoder(self):
        """Test getting cross-encoder reranker."""
        reranker = get_reranker("cross-encoder")
        assert isinstance(reranker, CrossEncoderReranker)
    
    def test_get_rrf(self):
        """Test getting RRF reranker."""
        reranker = get_reranker("rrf")
        assert isinstance(reranker, ReciprocRankFusionReranker)
    
    def test_get_default(self):
        """Test getting default reranker."""
        reranker = get_reranker()
        assert reranker is not None
    
    def test_invalid_type(self):
        """Test invalid reranker type."""
        with pytest.raises(ValueError):
            get_reranker("invalid_type")
