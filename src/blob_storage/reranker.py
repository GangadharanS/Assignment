"""
Re-ranking module for improving retrieval quality.

Uses cross-encoder models to re-score and reorder retrieved chunks
based on their relevance to the query.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from blob_storage.vector_store import SearchResult


@dataclass
class RerankResult:
    """Result after re-ranking."""
    
    text: str
    original_score: float
    rerank_score: float
    combined_score: float
    document_name: str
    chunk_index: int
    metadata: Dict[str, Any]


class CrossEncoderReranker:
    """
    Re-ranker using cross-encoder models from sentence-transformers.
    
    Cross-encoders are more accurate than bi-encoders (used for initial retrieval)
    because they process the query and document together, allowing for
    better understanding of their relationship.
    
    Popular models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced)
    - BAAI/bge-reranker-base (high quality)
    - BAAI/bge-reranker-large (highest quality, slower)
    """
    
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = 32,
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device to run on ("cpu", "cuda", "mps")
            batch_size: Batch size for scoring
        """
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", self.DEFAULT_MODEL
        )
        self.device = device
        self.batch_size = batch_size
        self._model = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with: pip install sentence-transformers"
            )
        
        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
        )
        print(f"✓ Loaded reranker model: {self.model_name}")
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = None,
        score_threshold: float = None,
        alpha: float = 0.7,
    ) -> List[RerankResult]:
        """
        Re-rank search results using cross-encoder.
        
        Args:
            query: The search query
            results: Initial search results from vector store
            top_k: Number of top results to return (None = all)
            score_threshold: Minimum rerank score to include
            alpha: Weight for combining scores (alpha * rerank + (1-alpha) * original)
            
        Returns:
            Re-ranked results sorted by combined score
        """
        if not results:
            return []
        
        self._load_model()
        
        # Create query-document pairs for scoring
        pairs = [(query, r.text) for r in results]
        
        # Get cross-encoder scores
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        
        # Normalize scores to 0-1 range using sigmoid
        import numpy as np
        normalized_scores = 1 / (1 + np.exp(-scores))
        
        # Create reranked results
        reranked = []
        for i, (result, rerank_score) in enumerate(zip(results, normalized_scores)):
            # Combine original and rerank scores
            combined_score = (
                alpha * float(rerank_score) + 
                (1 - alpha) * result.score
            )
            
            reranked.append(RerankResult(
                text=result.text,
                original_score=result.score,
                rerank_score=float(rerank_score),
                combined_score=combined_score,
                document_name=result.document_name,
                chunk_index=result.chunk_index,
                metadata=result.metadata,
            ))
        
        # Sort by combined score (descending)
        reranked.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Apply threshold filter
        if score_threshold is not None:
            reranked = [r for r in reranked if r.rerank_score >= score_threshold]
        
        # Apply top_k limit
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked


class CohereReranker:
    """
    Re-ranker using Cohere's rerank API.
    
    Requires COHERE_API_KEY environment variable.
    This is a paid service but provides high-quality reranking.
    """
    
    def __init__(self, api_key: str = None, model: str = "rerank-english-v2.0"):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key (or set COHERE_API_KEY env var)
            model: Cohere rerank model name
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Get or create Cohere client."""
        if self._client is not None:
            return self._client
        
        if not self.api_key:
            raise ValueError("Cohere API key required. Set COHERE_API_KEY env var.")
        
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere is required for Cohere reranking. "
                "Install with: pip install cohere"
            )
        
        self._client = cohere.Client(self.api_key)
        return self._client
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = None,
        score_threshold: float = None,
        alpha: float = 0.7,
    ) -> List[RerankResult]:
        """Re-rank using Cohere API."""
        if not results:
            return []
        
        client = self._get_client()
        
        # Call Cohere rerank API
        documents = [r.text for r in results]
        response = client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_k or len(documents),
        )
        
        # Map back to results
        reranked = []
        for item in response.results:
            original_result = results[item.index]
            rerank_score = item.relevance_score
            
            combined_score = (
                alpha * rerank_score + 
                (1 - alpha) * original_result.score
            )
            
            reranked.append(RerankResult(
                text=original_result.text,
                original_score=original_result.score,
                rerank_score=rerank_score,
                combined_score=combined_score,
                document_name=original_result.document_name,
                chunk_index=original_result.chunk_index,
                metadata=original_result.metadata,
            ))
        
        # Apply threshold filter
        if score_threshold is not None:
            reranked = [r for r in reranked if r.rerank_score >= score_threshold]
        
        return reranked


class ReciprocRankFusionReranker:
    """
    Reciprocal Rank Fusion (RRF) reranker.
    
    A simple but effective method that doesn't require a model.
    Combines rankings from multiple sources (e.g., semantic + keyword search).
    
    Formula: RRF(d) = Σ 1/(k + rank(d))
    where k is a constant (typically 60) and rank is the position in each list.
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF reranker.
        
        Args:
            k: Constant for RRF formula (higher = more even weighting)
        """
        self.k = k
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = None,
        score_threshold: float = None,
        alpha: float = 0.7,
    ) -> List[RerankResult]:
        """
        Re-rank using RRF formula.
        
        For a single result list, this effectively normalizes scores
        based on rank position.
        """
        if not results:
            return []
        
        # Calculate RRF scores based on position
        reranked = []
        for rank, result in enumerate(results, start=1):
            rrf_score = 1.0 / (self.k + rank)
            
            # Normalize to 0-1 range (approximate)
            max_rrf = 1.0 / (self.k + 1)
            normalized_rrf = rrf_score / max_rrf
            
            combined_score = (
                alpha * normalized_rrf + 
                (1 - alpha) * result.score
            )
            
            reranked.append(RerankResult(
                text=result.text,
                original_score=result.score,
                rerank_score=normalized_rrf,
                combined_score=combined_score,
                document_name=result.document_name,
                chunk_index=result.chunk_index,
                metadata=result.metadata,
            ))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Apply filters
        if score_threshold is not None:
            reranked = [r for r in reranked if r.rerank_score >= score_threshold]
        
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked


# Reranker types
RERANKER_TYPES = {
    "cross-encoder": CrossEncoderReranker,
    "cohere": CohereReranker,
    "rrf": ReciprocRankFusionReranker,
    "none": None,
}


def get_reranker(
    reranker_type: str = "cross-encoder",
    **kwargs,
):
    """
    Get a reranker instance.
    
    Args:
        reranker_type: Type of reranker ("cross-encoder", "cohere", "rrf", "none")
        **kwargs: Arguments for the reranker
        
    Returns:
        Reranker instance or None
    """
    reranker_type = reranker_type.lower()
    
    if reranker_type == "none" or reranker_type not in RERANKER_TYPES:
        return None
    
    reranker_class = RERANKER_TYPES[reranker_type]
    return reranker_class(**kwargs)


# Singleton instance
_reranker = None


def get_default_reranker() -> CrossEncoderReranker:
    """Get the default cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


