"""
Tests for the RAG pipeline module.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch

from blob_storage.rag import RAGPipeline, RAGResponse
from blob_storage.vector_store import SearchResult
from blob_storage.llm import LLMResponse


class TestRAGPipeline:
    """Tests for the RAG pipeline."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock()
        llm.generate.return_value = LLMResponse(
            content="This is the answer based on the context.",
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )
        llm.is_available.return_value = True
        return llm
    
    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results."""
        return [
            SearchResult(
                text="First relevant chunk about the topic.",
                score=0.85,
                metadata={"chunk_index": 0},
                document_name="doc1.pdf",
                chunk_index=0,
            ),
            SearchResult(
                text="Second relevant chunk with more details.",
                score=0.75,
                metadata={"chunk_index": 1},
                document_name="doc1.pdf",
                chunk_index=1,
            ),
        ]
    
    def test_initialization(self, mock_llm):
        """Test RAG pipeline initialization."""
        rag = RAGPipeline(llm=mock_llm)
        
        assert rag is not None
        assert rag.llm == mock_llm
    
    def test_initialization_with_custom_prompt(self, mock_llm):
        """Test initialization with custom system prompt."""
        custom_prompt = "You are a specialized assistant."
        rag = RAGPipeline(llm=mock_llm, system_prompt=custom_prompt)
        
        assert rag.system_prompt == custom_prompt
    
    def test_format_context(self, mock_llm, mock_search_results):
        """Test context formatting."""
        rag = RAGPipeline(llm=mock_llm)
        context = rag._format_context(mock_search_results, include_scores=True)
        
        assert "Chunk 1" in context
        assert "Chunk 2" in context
        assert "doc1.pdf" in context
        assert "First relevant chunk" in context
    
    def test_format_context_truncation(self, mock_llm):
        """Test context truncation for long chunks."""
        rag = RAGPipeline(llm=mock_llm)
        
        # Create a very long chunk
        long_results = [
            SearchResult(
                text="A" * 5000,  # Very long text
                score=0.9,
                metadata={},
                document_name="long.txt",
                chunk_index=0,
            ),
        ]
        
        context = rag._format_context(long_results, max_chunk_chars=500)
        
        # Should be truncated
        assert len(context) < 5000
        assert "truncated" in context.lower()
    
    def test_build_prompt(self, mock_llm):
        """Test prompt building."""
        rag = RAGPipeline(llm=mock_llm)
        
        context = "[Chunk 1] Source: doc.pdf\nTest content"
        query = "What is this about?"
        
        prompt = rag._build_prompt(query, context)
        
        assert "CONTEXT:" in prompt
        assert "Test content" in prompt
        assert "What is this about?" in prompt
    
    @patch("blob_storage.rag.get_vector_store")
    @patch("blob_storage.rag.get_embedding_generator")
    def test_query(self, mock_emb_gen, mock_vs, mock_llm, mock_search_results):
        """Test the query method."""
        # Setup mocks
        mock_store = Mock()
        mock_store.search_by_text.return_value = mock_search_results
        mock_vs.return_value = mock_store
        
        mock_generator = Mock()
        mock_generator.embed.return_value = [0.1] * 384
        mock_emb_gen.return_value = mock_generator
        
        rag = RAGPipeline(llm=mock_llm, enable_reranking=False)
        response = rag.query("What is the topic?", n_results=2)
        
        assert isinstance(response, RAGResponse)
        assert response.answer is not None
        assert len(response.sources) > 0
    
    def test_build_sources(self, mock_llm, mock_search_results):
        """Test source building."""
        rag = RAGPipeline(llm=mock_llm)
        sources = rag._build_sources(mock_search_results)
        
        assert len(sources) == 2
        assert sources[0]["document"] == "doc1.pdf"
        assert "relevance" in sources[0]


class TestRAGResponse:
    """Tests for the RAGResponse dataclass."""
    
    def test_create_response(self):
        """Test creating a RAG response."""
        response = RAGResponse(
            answer="The answer is 42.",
            query="What is the answer?",
            sources=[{"document": "guide.pdf", "relevance": 0.9}],
            model="llama3.2",
            context_chunks=1,
            usage={"total_tokens": 100},
        )
        
        assert response.answer == "The answer is 42."
        assert response.query == "What is the answer?"
        assert response.model == "llama3.2"
    
    def test_response_with_reranking(self):
        """Test response with reranking flag."""
        response = RAGResponse(
            answer="Answer",
            query="Query",
            sources=[],
            model="model",
            context_chunks=0,
            usage={},
            reranked=True,
        )
        
        assert response.reranked is True


class TestRAGPipelineEdgeCases:
    """Edge case tests for RAG pipeline."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock()
        llm.generate.return_value = LLMResponse(
            content="No relevant information found.",
            model="test-model",
            usage={},
        )
        return llm
    
    @patch("blob_storage.rag.get_vector_store")
    @patch("blob_storage.rag.get_embedding_generator")
    def test_query_no_results(self, mock_emb_gen, mock_vs, mock_llm):
        """Test query with no search results."""
        mock_store = Mock()
        mock_store.search_by_text.return_value = []
        mock_vs.return_value = mock_store
        
        mock_generator = Mock()
        mock_emb_gen.return_value = mock_generator
        
        rag = RAGPipeline(llm=mock_llm, enable_reranking=False)
        response = rag.query("Unknown topic", n_results=5)
        
        assert response.context_chunks == 0
    
    def test_format_context_empty(self, mock_llm):
        """Test formatting empty context."""
        rag = RAGPipeline(llm=mock_llm)
        context = rag._format_context([])
        
        assert "No relevant context" in context
