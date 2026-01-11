"""
Tests for the document processor module.
"""
import pytest
import tempfile
from pathlib import Path

from blob_storage.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Tests for the DocumentProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a document processor instance."""
        return DocumentProcessor()
    
    def test_process_txt_file(self, processor):
        """Test processing a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is test content.\nLine two.")
            temp_path = f.name
        
        try:
            text = processor.process_file(temp_path)
            assert "This is test content" in text
            assert "Line two" in text
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_process_txt_bytes(self, processor):
        """Test processing text from bytes."""
        content = b"Hello, World! This is a test."
        text = processor.process_bytes(content, "test.txt")
        
        assert "Hello, World" in text
    
    def test_process_csv_bytes(self, processor):
        """Test processing CSV content."""
        content = b"name,age\nAlice,30\nBob,25"
        text = processor.process_bytes(content, "data.csv")
        
        assert "name" in text or "Alice" in text
    
    def test_process_json_bytes(self, processor):
        """Test processing JSON content."""
        content = b'{"name": "Test", "value": 123}'
        text = processor.process_bytes(content, "data.json")
        
        assert "Test" in text or "name" in text
    
    def test_process_markdown_bytes(self, processor):
        """Test processing Markdown content."""
        content = b"# Title\n\nParagraph text.\n\n- Item 1\n- Item 2"
        text = processor.process_bytes(content, "doc.md")
        
        assert "Title" in text
        assert "Paragraph" in text
    
    def test_process_empty_content(self, processor):
        """Test processing empty content."""
        content = b""
        text = processor.process_bytes(content, "empty.txt")
        
        assert text == "" or text.strip() == ""
    
    def test_unsupported_format(self, processor):
        """Test handling unsupported file format."""
        content = b"\x00\x01\x02\x03"  # Binary content
        
        # Should not raise, might return empty or raw text
        text = processor.process_bytes(content, "file.xyz")
        assert text is not None
    
    def test_detect_file_type(self, processor):
        """Test file type detection."""
        assert processor._get_file_type("document.pdf") == "pdf"
        assert processor._get_file_type("report.docx") == "docx"
        assert processor._get_file_type("readme.txt") == "txt"
        assert processor._get_file_type("data.csv") == "csv"
        assert processor._get_file_type("config.json") == "json"
        assert processor._get_file_type("notes.md") == "md"


class TestDocumentProcessorEdgeCases:
    """Edge case tests for document processor."""
    
    @pytest.fixture
    def processor(self):
        """Create a document processor instance."""
        return DocumentProcessor()
    
    def test_unicode_content(self, processor):
        """Test processing Unicode content."""
        content = "Hello 世界! Привет мир! 🎉".encode('utf-8')
        text = processor.process_bytes(content, "unicode.txt")
        
        assert "Hello" in text
        # Unicode should be preserved or gracefully handled
    
    def test_large_content(self, processor):
        """Test processing large content."""
        content = ("Large text content. " * 10000).encode('utf-8')
        text = processor.process_bytes(content, "large.txt")
        
        assert len(text) > 0
        assert "Large text content" in text
    
    def test_special_characters(self, processor):
        """Test processing content with special characters."""
        content = b"Special chars: <>&\"'\\n\\t"
        text = processor.process_bytes(content, "special.txt")
        
        assert "Special chars" in text
    
    def test_multiple_newlines(self, processor):
        """Test processing content with multiple newlines."""
        content = b"Line 1\n\n\n\nLine 2\n\n\nLine 3"
        text = processor.process_bytes(content, "newlines.txt")
        
        assert "Line 1" in text
        assert "Line 2" in text
        assert "Line 3" in text
