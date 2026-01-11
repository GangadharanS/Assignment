"""
Embedding generation module using sentence-transformers.
"""
from typing import List, Optional, Union

import numpy as np

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class EmbeddingGenerator:
    """
    Generates semantic embeddings for text using sentence-transformers.
    
    Default model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
    Alternative models:
    - all-mpnet-base-v2: Higher quality, 768 dimensions, slower
    - paraphrase-MiniLM-L6-v2: Good for paraphrase detection
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
    
    @property
    def model(self) -> "SentenceTransformer":
        """Lazy load the model."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
        return self._model
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for one or more texts.
        
        Args:
            texts: A single text string or list of text strings
            
        Returns:
            Numpy array of embeddings. Shape: (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
        )
        
        return embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


# Singleton instance (lazy loaded)
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator(model_name: str = None) -> EmbeddingGenerator:
    """Get or create the embedding generator instance."""
    global _embedding_generator
    
    if _embedding_generator is None or (model_name and model_name != _embedding_generator.model_name):
        _embedding_generator = EmbeddingGenerator(model_name)
    
    return _embedding_generator


