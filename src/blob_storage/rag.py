"""
RAG (Retrieval-Augmented Generation) module.

Combines vector search with LLM generation for question answering.
Includes re-ranking for improved retrieval quality.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from blob_storage.embeddings import EmbeddingGenerator, get_embedding_generator
from blob_storage.llm import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    LLMBase,
    LLMResponse,
    Message,
    OllamaLLM,
    get_default_llm,
)
from blob_storage.vector_store import SearchResult, VectorStore, get_vector_store


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    model: str
    context_chunks: int
    usage: Dict[str, int] = None
    reranked: bool = False


class RAGPipeline:
    """
    RAG Pipeline for document-based question answering.
    
    Flow:
    1. Embed the user query
    2. Search vector store for similar chunks
    3. Re-rank results using cross-encoder (optional)
    4. Build context from top chunks
    5. Send context + query to LLM
    6. Return answer with sources
    """
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        embedding_generator: EmbeddingGenerator = None,
        llm: LLMBase = None,
        system_prompt: str = None,
        enable_reranking: bool = True,
        reranker_type: str = "cross-encoder",
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store for retrieving chunks
            embedding_generator: For embedding queries
            llm: Language model for generation
            system_prompt: Custom system prompt for the LLM
            enable_reranking: Whether to use re-ranking (default: True)
            reranker_type: Type of reranker ("cross-encoder", "cohere", "rrf", "none")
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedding_generator = embedding_generator or get_embedding_generator()
        self.llm = llm or get_default_llm()
        self.system_prompt = system_prompt or DEFAULT_RAG_SYSTEM_PROMPT
        self.enable_reranking = enable_reranking
        self.reranker_type = reranker_type
        self._reranker = None
    
    @property
    def reranker(self):
        """Lazy load the reranker."""
        if self._reranker is None and self.enable_reranking:
            try:
                from blob_storage.reranker import get_reranker
                self._reranker = get_reranker(self.reranker_type)
            except Exception as e:
                print(f"Warning: Could not load reranker: {e}")
                self._reranker = None
        return self._reranker
    
    def _format_context(
        self,
        results: Union[List[SearchResult], List[Any]],
        include_scores: bool = False,
        max_chunk_chars: int = 1500,  # Limit each chunk to prevent timeout
        max_total_chars: int = 8000,  # Limit total context size
    ) -> str:
        """
        Format search results into context for the LLM.
        
        Args:
            results: Search results (SearchResult or RerankResult)
            include_scores: Whether to include similarity scores
            max_chunk_chars: Maximum characters per chunk (truncates if longer)
            max_total_chars: Maximum total context characters
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        context_parts = []
        total_chars = 0
        
        for i, result in enumerate(results, 1):
            header = f"[Chunk {i}] Source: {result.document_name}"
            if include_scores:
                # Check if this is a rerank result (has combined_score)
                if hasattr(result, 'combined_score'):
                    header += f" (relevance: {result.combined_score:.2%})"
                else:
                    header += f" (relevance: {result.score:.2%})"
            
            # Truncate long chunks
            text = result.text
            if len(text) > max_chunk_chars:
                text = text[:max_chunk_chars] + "... [truncated]"
            
            chunk_content = f"{header}\n{text}"
            
            # Check if adding this chunk would exceed total limit
            if total_chars + len(chunk_content) > max_total_chars:
                if context_parts:  # Only break if we have at least one chunk
                    break
            
            context_parts.append(chunk_content)
            total_chars += len(chunk_content)
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Build the user prompt with context.
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        return f"""CONTEXT:
{context}

---

QUESTION: {query}

Please answer the question based on the context provided above."""
    
    def query(
        self,
        query: str,
        n_results: int = 5,
        filter_document: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        include_scores: bool = True,
        min_relevance: float = 0.0,
        use_reranking: bool = None,
        rerank_top_k: int = None,
    ) -> RAGResponse:
        """
        Query the RAG pipeline.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            filter_document: Optional document filter
            temperature: LLM temperature
            max_tokens: Max response tokens
            include_scores: Include relevance scores in context
            min_relevance: Minimum relevance score for chunks
            use_reranking: Override default reranking setting
            rerank_top_k: Number of top results after reranking
            
        Returns:
            RAGResponse with answer and sources
        """
        # Step 1: Embed the query
        query_embedding = self.embedding_generator.embed_single(query)
        
        # Step 2: Search for similar chunks (retrieve more if reranking)
        should_rerank = use_reranking if use_reranking is not None else self.enable_reranking
        retrieve_count = n_results * 3 if should_rerank else n_results
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=retrieve_count,
            filter_document=filter_document,
        )
        
        # Filter by minimum relevance if specified
        if min_relevance > 0:
            results = [r for r in results if r.score >= min_relevance]
        
        # Step 3: Re-rank if enabled
        reranked = False
        if should_rerank and self.reranker and results:
            top_k = rerank_top_k or n_results
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_k=top_k,
            )
            reranked = True
        else:
            # Limit to n_results if not reranking
            results = results[:n_results]
        
        # Step 4: Format context
        context = self._format_context(results, include_scores)
        
        # Step 5: Build prompt and generate response
        user_prompt = self._build_prompt(query, context)
        
        llm_response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Step 6: Build response with sources
        sources = self._build_sources(results, reranked)
        
        return RAGResponse(
            answer=llm_response.content,
            sources=sources,
            query=query,
            model=llm_response.model,
            context_chunks=len(results),
            usage=llm_response.usage,
            reranked=reranked,
        )
    
    def _build_sources(
        self,
        results: List[Any],
        reranked: bool = False,
    ) -> List[Dict[str, Any]]:
        """Build source list from results."""
        sources = []
        for r in results:
            source = {
                "document": r.document_name,
                "chunk_index": r.chunk_index,
                "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
            }
            
            if reranked and hasattr(r, 'combined_score'):
                source["relevance"] = round(r.combined_score, 4)
                source["original_score"] = round(r.original_score, 4)
                source["rerank_score"] = round(r.rerank_score, 4)
            else:
                source["relevance"] = round(r.score, 4)
            
            sources.append(source)
        
        return sources
    
    async def aquery(
        self,
        query: str,
        n_results: int = 5,
        filter_document: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        include_scores: bool = True,
        min_relevance: float = 0.0,
        use_reranking: bool = None,
        rerank_top_k: int = None,
    ) -> RAGResponse:
        """Async version of query."""
        # Step 1: Embed the query
        query_embedding = self.embedding_generator.embed_single(query)
        
        # Step 2: Search for similar chunks (retrieve more if reranking)
        should_rerank = use_reranking if use_reranking is not None else self.enable_reranking
        retrieve_count = n_results * 3 if should_rerank else n_results
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=retrieve_count,
            filter_document=filter_document,
        )
        
        # Filter by minimum relevance if specified
        if min_relevance > 0:
            results = [r for r in results if r.score >= min_relevance]
        
        # Step 3: Re-rank if enabled
        reranked = False
        if should_rerank and self.reranker and results:
            top_k = rerank_top_k or n_results
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_k=top_k,
            )
            reranked = True
        else:
            # Limit to n_results if not reranking
            results = results[:n_results]
        
        # Step 4: Format context
        context = self._format_context(results, include_scores)
        
        # Step 5: Build prompt and generate response
        user_prompt = self._build_prompt(query, context)
        
        # Use async LLM call if available
        if hasattr(self.llm, 'agenerate'):
            llm_response = await self.llm.agenerate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            llm_response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        # Step 6: Build response with sources
        sources = self._build_sources(results, reranked)
        
        return RAGResponse(
            answer=llm_response.content,
            sources=sources,
            query=query,
            model=llm_response.model,
            context_chunks=len(results),
            usage=llm_response.usage,
            reranked=reranked,
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        n_results: int = 5,
        filter_document: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> RAGResponse:
        """
        Multi-turn chat with RAG.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            n_results: Number of chunks to retrieve
            filter_document: Optional document filter
            temperature: LLM temperature
            max_tokens: Max response tokens
            
        Returns:
            RAGResponse with answer and sources
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Get the last user message for retrieval
        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        if not last_user_msg:
            raise ValueError("No user message found")
        
        # Retrieve relevant chunks based on last user message
        query_embedding = self.embedding_generator.embed_single(last_user_msg)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            filter_document=filter_document,
        )
        
        # Build context
        context = self._format_context(results, include_scores=True)
        
        # Build message list with system prompt and context
        llm_messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=f"CONTEXT:\n{context}\n\n---"),
        ]
        
        # Add conversation history
        for msg in messages:
            llm_messages.append(
                Message(role=msg["role"], content=msg["content"])
            )
        
        # Generate response
        llm_response = self.llm.chat(
            messages=llm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Build response
        sources = [
            {
                "document": r.document_name,
                "chunk_index": r.chunk_index,
                "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                "relevance": round(r.score, 4),
            }
            for r in results
        ]
        
        return RAGResponse(
            answer=llm_response.content,
            sources=sources,
            query=last_user_msg,
            model=llm_response.model,
            context_chunks=len(results),
            usage=llm_response.usage,
        )


# Singleton instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline(
    system_prompt: str = None,
    llm_provider: str = "ollama",
    llm_model: str = None,
) -> RAGPipeline:
    """
    Get or create the RAG pipeline instance.
    
    Args:
        system_prompt: Custom system prompt
        llm_provider: LLM provider ("ollama" or "huggingface")
        llm_model: Specific model to use
        
    Returns:
        RAGPipeline instance
    """
    global _rag_pipeline
    
    if _rag_pipeline is None:
        from blob_storage.llm import get_llm
        
        llm = get_llm(provider=llm_provider, model=llm_model)
        _rag_pipeline = RAGPipeline(
            llm=llm,
            system_prompt=system_prompt,
        )
    
    return _rag_pipeline

