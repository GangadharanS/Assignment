"""
Text chunking module for splitting documents into smaller pieces.
Includes semantic chunking using embeddings to find natural breakpoints.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TextChunk:
    """Represents a chunk of text from a document."""
    
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextChunker:
    """
    Splits text into chunks using various strategies.
    
    Strategies:
    - fixed_size: Split by character count with overlap
    - sentence: Split by sentences, respecting max size
    - paragraph: Split by paragraphs, respecting max size
    - semantic: Split by semantic similarity using embeddings (recommended)
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "semantic",
        similarity_threshold: float = 0.5,
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            strategy: Chunking strategy (fixed_size, sentence, paragraph, semantic)
            similarity_threshold: For semantic chunking, threshold below which to split
                                  (lower = more splits, higher = fewer splits)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self._embedding_model = None
    
    def _get_embedding_model(self):
        """Lazy load the embedding model for semantic chunking."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("Loading embedding model for semantic chunking...")
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for semantic chunking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedding_model
    
    def chunk(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """
        Split text into chunks.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        if self.strategy == "fixed_size":
            return self._chunk_fixed_size(text, metadata)
        elif self.strategy == "sentence":
            return self._chunk_by_sentence(text, metadata)
        elif self.strategy == "paragraph":
            return self._chunk_by_paragraph(text, metadata)
        elif self.strategy == "semantic":
            return self._chunk_semantic(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common sentence endings
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                cleaned.append(sent)
        
        return cleaned
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _chunk_semantic(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """
        Split text using semantic similarity between sentences.
        
        This method:
        1. Splits text into sentences
        2. Generates embeddings for each sentence
        3. Calculates similarity between consecutive sentences
        4. Splits where similarity drops below threshold
        5. Combines small chunks to respect chunk_size
        """
        model = self._get_embedding_model()
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            # Not enough sentences to chunk semantically
            return self._chunk_by_paragraph(text, metadata)
        
        # Generate embeddings for all sentences
        print(f"Generating embeddings for {len(sentences)} sentences...")
        embeddings = model.encode(sentences, show_progress_bar=len(sentences) > 20)
        
        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Find semantic breakpoints (where similarity drops)
        breakpoints = [0]  # Start of first chunk
        
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                breakpoints.append(i + 1)
        
        breakpoints.append(len(sentences))  # End of last chunk
        
        # Create initial chunks from breakpoints
        initial_chunks = []
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            initial_chunks.append(chunk_text)
        
        # Merge small chunks and split large ones to respect chunk_size
        final_chunks = self._balance_chunks(initial_chunks, metadata)
        
        return final_chunks
    
    def _balance_chunks(self, chunks: List[str], metadata: dict = None) -> List[TextChunk]:
        """
        Balance chunks to respect chunk_size by merging small chunks
        and splitting large ones.
        """
        result = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        start_char = 0
        
        for chunk_text in chunks:
            chunk_len = len(chunk_text)
            
            # If this chunk alone is too large, split it
            if chunk_len > self.chunk_size:
                # First, save current accumulated chunk
                if current_chunk:
                    text = " ".join(current_chunk)
                    result.append(TextChunk(
                        text=text,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(text),
                        metadata=metadata.copy() if metadata else {},
                    ))
                    chunk_index += 1
                    start_char += len(text) + 1
                    current_chunk = []
                    current_length = 0
                
                # Split the large chunk by sentences
                sub_chunks = self._chunk_by_sentence(chunk_text, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.start_char = start_char
                    sub_chunk.end_char = start_char + len(sub_chunk.text)
                    result.append(sub_chunk)
                    chunk_index += 1
                    start_char += len(sub_chunk.text) + 1
                continue
            
            # If adding this chunk would exceed limit, save current and start new
            if current_length + chunk_len > self.chunk_size and current_chunk:
                text = " ".join(current_chunk)
                result.append(TextChunk(
                    text=text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(text),
                    metadata=metadata.copy() if metadata else {},
                ))
                chunk_index += 1
                start_char += len(text) + 1
                current_chunk = []
                current_length = 0
            
            current_chunk.append(chunk_text)
            current_length += chunk_len + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk:
            text = " ".join(current_chunk)
            result.append(TextChunk(
                text=text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(text),
                metadata=metadata.copy() if metadata else {},
            ))
        
        return result
    
    def _chunk_fixed_size(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """Split text by fixed character count with overlap."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a word boundary
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=metadata.copy() if metadata else {},
                ))
                chunk_index += 1
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
            if start <= chunks[-1].start_char if chunks else 0:
                start = end  # Prevent infinite loop
        
        return chunks
    
    def _chunk_by_sentence(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """Split text by sentences, combining until chunk_size is reached."""
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata=metadata.copy() if metadata else {},
                ))
                chunk_index += 1
                
                # Keep some sentences for overlap
                overlap_text = ""
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if len(overlap_text) + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_text = " ".join(overlap_sentences)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = len(overlap_text)
                start_char = start_char + len(chunk_text) - len(overlap_text)
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata=metadata.copy() if metadata else {},
            ))
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """Split text by paragraphs, combining until chunk_size is reached."""
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        start_char = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            # If single paragraph exceeds chunk_size, split it further
            if para_length > self.chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        metadata=metadata.copy() if metadata else {},
                    ))
                    chunk_index += 1
                    start_char += len(chunk_text) + 2
                    current_chunk = []
                    current_length = 0
                
                # Split the long paragraph by sentences
                sub_chunks = self._chunk_by_sentence(para, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_index = chunk_index
                    chunks.append(sub_chunk)
                    chunk_index += 1
                continue
            
            # If adding this paragraph exceeds chunk_size, save current chunk
            if current_length + para_length > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata=metadata.copy() if metadata else {},
                ))
                chunk_index += 1
                start_char += len(chunk_text) + 2
                current_chunk = []
                current_length = 0
            
            current_chunk.append(para)
            current_length += para_length + 2  # +2 for paragraph separator
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata=metadata.copy() if metadata else {},
            ))
        
        return chunks


class SemanticChunker:
    """
    Advanced semantic chunker that uses embeddings to find optimal split points.
    
    This chunker:
    1. Splits text into sentences
    2. Creates sliding window of sentences
    3. Computes embeddings for each window
    4. Finds points where semantic similarity changes significantly
    5. Splits at those points to create coherent chunks
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.5,
        window_size: int = 3,
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            model_name: Sentence transformer model to use
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            similarity_threshold: Threshold for semantic similarity (0-1)
                                  Lower = more splits, Higher = fewer splits
            window_size: Number of sentences to consider together
        """
        self.model_name = model_name
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self._model = None
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading semantic chunking model: {self.model_name}...")
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: pip install sentence-transformers"
                )
        return self._model
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # More robust sentence splitting
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Split on sentence boundaries
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z"])'
        sentences = re.split(sentence_endings, text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for sentences."""
        return self.model.encode(sentences, show_progress_bar=len(sentences) > 20)
    
    def _calculate_breakpoints(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
    ) -> List[int]:
        """
        Calculate optimal breakpoints based on semantic similarity.
        
        Uses a sliding window approach to find where topic changes.
        """
        if len(sentences) <= self.window_size * 2:
            return []
        
        breakpoints = []
        
        # Calculate similarity between adjacent windows
        for i in range(self.window_size, len(sentences) - self.window_size):
            # Get embeddings for sentences before and after this point
            before_window = embeddings[i - self.window_size:i]
            after_window = embeddings[i:i + self.window_size]
            
            # Average embeddings for each window
            before_avg = np.mean(before_window, axis=0)
            after_avg = np.mean(after_window, axis=0)
            
            # Calculate cosine similarity
            similarity = np.dot(before_avg, after_avg) / (
                np.linalg.norm(before_avg) * np.linalg.norm(after_avg)
            )
            
            # If similarity is below threshold, this is a potential breakpoint
            if similarity < self.similarity_threshold:
                breakpoints.append(i)
        
        # Remove breakpoints that are too close together
        if breakpoints:
            filtered = [breakpoints[0]]
            for bp in breakpoints[1:]:
                if bp - filtered[-1] >= self.window_size:
                    filtered.append(bp)
            breakpoints = filtered
        
        return breakpoints
    
    def chunk(self, text: str, metadata: dict = None) -> List[TextChunk]:
        """
        Chunk text using semantic similarity.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [TextChunk(
                text=text.strip(),
                chunk_index=0,
                start_char=0,
                end_char=len(text),
                metadata=metadata.copy() if metadata else {},
            )]
        
        # Get embeddings for all sentences
        print(f"Computing embeddings for {len(sentences)} sentences...")
        embeddings = self._get_sentence_embeddings(sentences)
        
        # Find semantic breakpoints
        breakpoints = self._calculate_breakpoints(sentences, embeddings)
        
        # Add start and end
        all_points = [0] + breakpoints + [len(sentences)]
        
        # Create chunks from breakpoints
        chunks = []
        chunk_index = 0
        current_pos = 0
        
        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i + 1]
            
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            
            # If chunk is too large, split further
            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk_sentences, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.start_char = current_pos
                    sub_chunk.end_char = current_pos + len(sub_chunk.text)
                    chunks.append(sub_chunk)
                    chunk_index += 1
                    current_pos += len(sub_chunk.text) + 1
            else:
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=current_pos,
                    end_char=current_pos + len(chunk_text),
                    metadata=metadata.copy() if metadata else {},
                ))
                chunk_index += 1
                current_pos += len(chunk_text) + 1
        
        # Merge tiny chunks
        chunks = self._merge_small_chunks(chunks, metadata)
        
        return chunks
    
    def _split_large_chunk(
        self,
        sentences: List[str],
        metadata: dict = None,
    ) -> List[TextChunk]:
        """Split a large chunk into smaller ones by sentence count."""
        chunks = []
        current = []
        current_len = 0
        
        for sent in sentences:
            if current_len + len(sent) > self.max_chunk_size and current:
                chunks.append(TextChunk(
                    text=" ".join(current),
                    chunk_index=0,
                    start_char=0,
                    end_char=0,
                    metadata=metadata.copy() if metadata else {},
                ))
                current = []
                current_len = 0
            
            current.append(sent)
            current_len += len(sent) + 1
        
        if current:
            chunks.append(TextChunk(
                text=" ".join(current),
                chunk_index=0,
                start_char=0,
                end_char=0,
                metadata=metadata.copy() if metadata else {},
            ))
        
        return chunks
    
    def _merge_small_chunks(
        self,
        chunks: List[TextChunk],
        metadata: dict = None,
    ) -> List[TextChunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return chunks
        
        merged = []
        current_text = chunks[0].text
        current_start = chunks[0].start_char
        
        for i in range(1, len(chunks)):
            chunk = chunks[i]
            
            # If current is too small and combining wouldn't exceed max
            if (len(current_text) < self.min_chunk_size and 
                len(current_text) + len(chunk.text) <= self.max_chunk_size):
                current_text = current_text + " " + chunk.text
            else:
                # Save current chunk
                merged.append(TextChunk(
                    text=current_text,
                    chunk_index=len(merged),
                    start_char=current_start,
                    end_char=current_start + len(current_text),
                    metadata=metadata.copy() if metadata else {},
                ))
                current_text = chunk.text
                current_start = chunk.start_char
        
        # Don't forget the last one
        merged.append(TextChunk(
            text=current_text,
            chunk_index=len(merged),
            start_char=current_start,
            end_char=current_start + len(current_text),
            metadata=metadata.copy() if metadata else {},
        ))
        
        return merged


# Default chunker instances
default_chunker = TextChunker(chunk_size=1000, chunk_overlap=200, strategy="semantic")
semantic_chunker = SemanticChunker(max_chunk_size=1000, similarity_threshold=0.5)
