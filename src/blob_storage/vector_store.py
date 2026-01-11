"""
Vector store module using ChromaDB for storing and searching embeddings.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try importing chromadb
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from blob_storage.chunker import TextChunk
from blob_storage.config import config


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""
    
    text: str
    score: float
    metadata: Dict[str, Any]
    document_name: str
    chunk_index: int


class VectorStore:
    """
    Vector store using ChromaDB for persistent storage and similarity search.
    """
    
    DEFAULT_COLLECTION = "documents"
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        if not HAS_CHROMADB:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install with: pip install chromadb"
            )
        
        self.persist_directory = persist_directory or os.path.join(
            config.LOCAL_STORAGE_PATH, "vector_db"
        )
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        
        # Create persist directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self._collection = None
    
    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks with embeddings"},
            )
        return self._collection
    
    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
        document_name: str,
        document_metadata: Dict[str, Any] = None,
    ) -> List[str]:
        """
        Add document chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors (one per chunk)
            document_name: Name of the source document
            document_metadata: Additional metadata for the document
            
        Returns:
            List of chunk IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if not chunks:
            return []
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate unique ID
            chunk_id = f"{document_name}::chunk_{chunk.chunk_index}_{i}"
            ids.append(chunk_id)
            documents.append(chunk.text)
            
            # Build metadata
            metadata = {
                "document_name": document_name,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "indexed_at": datetime.now().isoformat(),
            }
            
            # Add document metadata
            if document_metadata:
                for key, value in document_metadata.items():
                    # ChromaDB only supports str, int, float, bool
                    if isinstance(value, (str, int, float, bool)):
                        metadata[f"doc_{key}"] = value
            
            # Add chunk metadata
            if chunk.metadata:
                for key, value in chunk.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[f"chunk_{key}"] = value
            
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        
        print(f"✓ Added {len(chunks)} chunks from '{document_name}' to vector store")
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_document: str = None,
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return
            filter_document: Optional document name to filter results
            
        Returns:
            List of SearchResult objects
        """
        where_filter = None
        if filter_document:
            where_filter = {"document_name": filter_document}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        search_results = []
        
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                
                # Convert distance to similarity score (ChromaDB uses L2 distance)
                # Lower distance = higher similarity
                score = 1 / (1 + distance)
                
                search_results.append(SearchResult(
                    text=doc,
                    score=score,
                    metadata=metadata,
                    document_name=metadata.get("document_name", "unknown"),
                    chunk_index=metadata.get("chunk_index", 0),
                ))
        
        return search_results
    
    def search_by_text(
        self,
        query_text: str,
        embedding_generator,
        n_results: int = 5,
        filter_document: str = None,
    ) -> List[SearchResult]:
        """
        Search using text query (generates embedding automatically).
        
        Args:
            query_text: The search query text
            embedding_generator: EmbeddingGenerator instance
            n_results: Number of results to return
            filter_document: Optional document name to filter results
            
        Returns:
            List of SearchResult objects
        """
        query_embedding = embedding_generator.embed_single(query_text)
        return self.search(query_embedding, n_results, filter_document)
    
    def delete_document(self, document_name: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_name: Name of the document to delete
            
        Returns:
            Number of chunks deleted
        """
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"document_name": document_name},
            include=[],
        )
        
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            count = len(results["ids"])
            print(f"✓ Deleted {count} chunks for document '{document_name}'")
            return count
        
        return 0
    
    def get_document_chunks(self, document_name: str) -> List[Dict]:
        """
        Get all chunks for a document.
        
        Args:
            document_name: Name of the document
            
        Returns:
            List of chunk data dictionaries
        """
        results = self.collection.get(
            where={"document_name": document_name},
            include=["documents", "metadatas"],
        )
        
        chunks = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                chunks.append({
                    "id": results["ids"][i],
                    "text": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        
        return chunks
    
    def list_documents(self) -> List[Dict]:
        """
        List all indexed documents.
        
        Returns:
            List of document info dictionaries
        """
        # Get all items
        results = self.collection.get(include=["metadatas"])
        
        # Extract unique documents
        documents = {}
        if results and results["metadatas"]:
            for metadata in results["metadatas"]:
                doc_name = metadata.get("document_name", "unknown")
                if doc_name not in documents:
                    documents[doc_name] = {
                        "name": doc_name,
                        "chunk_count": 0,
                        "indexed_at": metadata.get("indexed_at"),
                    }
                documents[doc_name]["chunk_count"] += 1
        
        return list(documents.values())
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        count = self.collection.count()
        documents = self.list_documents()
        
        return {
            "total_chunks": count,
            "total_documents": len(documents),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }
    
    def clear(self) -> Dict:
        """
        Clear all documents from the vector store.
        
        Returns:
            Dict with cleared count info
        """
        count_before = self.collection.count()
        documents_before = len(self.list_documents())
        
        # Delete the collection and recreate it
        self.client.delete_collection(self.collection_name)
        self._collection = None  # Reset cached collection
        
        # Recreate empty collection
        _ = self.collection  # This will recreate it
        
        return {
            "cleared_chunks": count_before,
            "cleared_documents": documents_before,
            "message": f"Cleared {count_before} chunks from {documents_before} documents",
        }


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store(
    persist_directory: str = None,
    collection_name: str = None,
) -> VectorStore:
    """Get or create the vector store instance."""
    global _vector_store
    
    if _vector_store is None:
        _vector_store = VectorStore(persist_directory, collection_name)
    
    return _vector_store


