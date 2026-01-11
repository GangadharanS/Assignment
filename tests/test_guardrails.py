"""
Tests for the guardrails module.
"""
import pytest

from blob_storage.guardrails import (
    QueryGuardrails,
    RelevanceGuardrails,
    RAGGuardrails,
    FallbackResponses,
    SourceAttributionGuardrails,
    QueryType,
    ConfidenceLevel,
)


class TestQueryGuardrails:
    """Tests for query validation guardrails."""
    
    @pytest.fixture
    def guardrails(self):
        """Create query guardrails instance."""
        return QueryGuardrails(min_query_length=3)
    
    def test_valid_query(self, guardrails):
        """Test a valid query passes validation."""
        result = guardrails.check("What is the main topic of this document?")
        
        assert result.is_valid is True
        assert result.cleaned_query == "What is the main topic of this document?"
        assert len(result.issues) == 0
    
    def test_empty_query(self, guardrails):
        """Test empty query is rejected."""
        result = guardrails.check("")
        
        assert result.is_valid is False
        assert len(result.issues) > 0
    
    def test_short_query(self, guardrails):
        """Test too-short query is rejected."""
        result = guardrails.check("hi")
        
        assert result.is_valid is False
    
    def test_whitespace_query(self, guardrails):
        """Test whitespace-only query is rejected."""
        result = guardrails.check("   ")
        
        assert result.is_valid is False
    
    def test_query_classification_factual(self, guardrails):
        """Test factual query classification."""
        result = guardrails.check("What is the capital of France?")
        
        assert result.query_type == QueryType.FACTUAL
    
    def test_query_classification_greeting(self, guardrails):
        """Test greeting classification."""
        result = guardrails.check("Hello there!")
        
        assert result.query_type == QueryType.GREETING
    
    def test_query_trimming(self, guardrails):
        """Test that queries are trimmed."""
        result = guardrails.check("  What is this?  ")
        
        assert result.cleaned_query == "What is this?"


class TestRelevanceGuardrails:
    """Tests for relevance checking guardrails."""
    
    @pytest.fixture
    def guardrails(self):
        """Create relevance guardrails instance."""
        return RelevanceGuardrails(min_relevance=0.1, high_confidence=0.5)
    
    def test_high_relevance_sources(self, guardrails):
        """Test high relevance sources give high confidence."""
        sources = [
            {"relevance": 0.8, "text": "Relevant content"},
            {"relevance": 0.7, "text": "Also relevant"},
        ]
        
        result = guardrails.check(sources)
        
        assert result.has_relevant_context is True
        assert result.confidence == ConfidenceLevel.HIGH
    
    def test_medium_relevance_sources(self, guardrails):
        """Test medium relevance sources."""
        sources = [
            {"relevance": 0.3, "text": "Somewhat relevant"},
            {"relevance": 0.2, "text": "Less relevant"},
        ]
        
        result = guardrails.check(sources)
        
        assert result.has_relevant_context is True
        assert result.confidence in [ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]
    
    def test_low_relevance_sources(self, guardrails):
        """Test low relevance sources."""
        sources = [
            {"relevance": 0.05, "text": "Not relevant"},
        ]
        
        result = guardrails.check(sources)
        
        assert result.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.NONE]
    
    def test_no_sources(self, guardrails):
        """Test empty sources list."""
        result = guardrails.check([])
        
        assert result.has_relevant_context is False
        assert result.confidence == ConfidenceLevel.NONE
    
    def test_relevance_metrics(self, guardrails):
        """Test relevance metrics are calculated."""
        sources = [
            {"relevance": 0.8},
            {"relevance": 0.6},
            {"relevance": 0.4},
        ]
        
        result = guardrails.check(sources)
        
        assert result.top_relevance == 0.8
        assert result.avg_relevance == pytest.approx(0.6, rel=0.1)


class TestRAGGuardrails:
    """Tests for combined RAG guardrails."""
    
    @pytest.fixture
    def guardrails(self):
        """Create RAG guardrails instance."""
        return RAGGuardrails()
    
    def test_check_query(self, guardrails):
        """Test query checking."""
        result = guardrails.check_query("What is this document about?")
        
        assert result.is_valid is True
    
    def test_check_relevance(self, guardrails):
        """Test relevance checking."""
        sources = [{"relevance": 0.5, "text": "content"}]
        result = guardrails.check_relevance(sources, "query")
        
        assert result.has_relevant_context is True
    
    def test_full_check_valid(self, guardrails):
        """Test full guardrails check with valid input."""
        sources = [{"relevance": 0.6, "text": "Relevant content"}]
        result = guardrails.full_check(
            query="What is the topic?",
            sources=sources,
        )
        
        assert result.query_valid is True
        assert result.has_relevant_context is True
        assert result.should_use_fallback is False
    
    def test_full_check_no_context(self, guardrails):
        """Test full check with no relevant context."""
        result = guardrails.full_check(
            query="What is the topic?",
            sources=[],
        )
        
        assert result.should_use_fallback is True


class TestFallbackResponses:
    """Tests for fallback response messages."""
    
    def test_no_context_message(self):
        """Test no context fallback message exists."""
        assert FallbackResponses.NO_CONTEXT is not None
        assert len(FallbackResponses.NO_CONTEXT) > 0
    
    def test_low_confidence_message(self):
        """Test low confidence message exists."""
        assert FallbackResponses.LOW_CONFIDENCE is not None


class TestSourceAttributionGuardrails:
    """Tests for source attribution."""
    
    def test_format_citations_single_source(self):
        """Test formatting citations for single source."""
        sources = [{"document": "test.pdf", "chunk_index": 0}]
        citations = SourceAttributionGuardrails.format_citations(sources)
        
        assert "test.pdf" in citations
    
    def test_format_citations_multiple_sources(self):
        """Test formatting citations for multiple sources."""
        sources = [
            {"document": "doc1.pdf", "chunk_index": 0},
            {"document": "doc2.pdf", "chunk_index": 1},
        ]
        citations = SourceAttributionGuardrails.format_citations(sources)
        
        assert "doc1.pdf" in citations
        assert "doc2.pdf" in citations
    
    def test_format_citations_empty(self):
        """Test formatting empty citations."""
        citations = SourceAttributionGuardrails.format_citations([])
        
        assert citations == ""
    
    def test_format_citations_deduplicated(self):
        """Test that duplicate sources are deduplicated."""
        sources = [
            {"document": "same.pdf", "chunk_index": 0},
            {"document": "same.pdf", "chunk_index": 1},
            {"document": "same.pdf", "chunk_index": 2},
        ]
        citations = SourceAttributionGuardrails.format_citations(sources)
        
        # Should only appear once
        assert citations.count("same.pdf") == 1
