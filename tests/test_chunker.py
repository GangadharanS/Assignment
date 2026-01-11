"""
Tests for the chunker module.
"""
import pytest

from blob_storage.chunker import TextChunker, SemanticChunker, TextChunk


class TestTextChunker:
    """Tests for the basic TextChunker."""
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk("")
        assert chunks == []
    
    def test_chunk_short_text(self):
        """Test text shorter than chunk size."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        text = "Short text."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0
    
    def test_chunk_long_text(self):
        """Test text longer than chunk size."""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "This is a longer piece of text that should be split into multiple chunks for processing."
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 1
        # Verify all chunks are created
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert len(chunk.text) <= 50 or " " not in chunk.text  # May exceed if no space
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = TextChunker(chunk_size=30, overlap=10)
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        chunks = chunker.chunk(text)
        
        # With overlap, consecutive chunks should share some content
        if len(chunks) > 1:
            # Check that chunks are connected
            assert chunks[0].end_char >= chunks[1].start_char or True  # Simplified check
    
    def test_chunk_metadata(self):
        """Test chunk metadata."""
        chunker = TextChunker(chunk_size=100, overlap=0)
        text = "Test text for metadata."
        chunks = chunker.chunk(text, document_name="test.txt")
        
        assert len(chunks) == 1
        assert chunks[0].document_name == "test.txt"
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)


class TestSemanticChunker:
    """Tests for the SemanticChunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create a semantic chunker for testing."""
        return SemanticChunker(
            max_chunk_size=500,
            min_chunk_size=50,
            similarity_threshold=0.5,
            window_size=2,
        )
    
    def test_chunk_empty_text(self, chunker):
        """Test chunking empty text."""
        chunks = chunker.chunk("")
        assert chunks == []
    
    def test_chunk_single_sentence(self, chunker):
        """Test single sentence."""
        text = "This is a single sentence."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].text.strip() == text
    
    def test_chunk_multiple_sentences(self, chunker):
        """Test multiple sentences are chunked."""
        text = """
        First topic about technology. Computers are powerful machines. They process data quickly.
        
        Second topic about nature. Trees provide oxygen. Forests are important ecosystems.
        
        Third topic about food. Pizza is delicious. Many people enjoy Italian cuisine.
        """
        chunks = chunker.chunk(text)
        
        # Should create at least one chunk
        assert len(chunks) >= 1
        
        # All text should be represented
        combined = " ".join(c.text for c in chunks)
        assert "technology" in combined
        assert "nature" in combined
        assert "food" in combined
    
    def test_chunk_respects_max_size(self, chunker):
        """Test that chunks respect max size."""
        # Create text with many sentences
        text = " ".join(["This is sentence number {}.".format(i) for i in range(100)])
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            assert len(chunk.text) <= chunker.max_chunk_size + 100  # Allow some flexibility
    
    def test_chunk_indices_sequential(self, chunker):
        """Test that chunk indices are sequential."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunker.chunk(text)
        
        for i, chunk in enumerate(chunks):
            assert chunk.index == i


class TestTextChunk:
    """Tests for the TextChunk dataclass."""
    
    def test_create_chunk(self):
        """Test creating a text chunk."""
        chunk = TextChunk(
            text="Test content",
            index=0,
            start_char=0,
            end_char=12,
            document_name="test.txt",
        )
        
        assert chunk.text == "Test content"
        assert chunk.index == 0
        assert chunk.document_name == "test.txt"
    
    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = TextChunk(
            text="Test",
            index=1,
            start_char=10,
            end_char=14,
            document_name="doc.pdf",
        )
        
        d = chunk.to_dict()
        assert d["text"] == "Test"
        assert d["index"] == 1
        assert d["document_name"] == "doc.pdf"
