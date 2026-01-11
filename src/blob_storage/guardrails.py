"""
Guardrails module for RAG safety and quality control.

Implements:
- Query validation and preprocessing
- Relevance checking
- Fallback responses
- Source attribution verification
- Ambiguous query handling
- Content safety checks
- Hallucination detection
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Minimum query length to be considered valid
MIN_QUERY_LENGTH = 3
# Minimum relevance score to consider context useful
# Lowered to 0.10 to handle typos and semantic variations better
MIN_RELEVANCE_THRESHOLD = 0.10
# High confidence threshold
HIGH_CONFIDENCE_THRESHOLD = 0.5
# Maximum query length
MAX_QUERY_LENGTH = 2000


class QueryType(Enum):
    """Types of user queries."""
    FACTUAL = "factual"           # Seeking specific facts
    EXPLORATORY = "exploratory"   # Open-ended exploration
    COMPARATIVE = "comparative"   # Comparing things
    PROCEDURAL = "procedural"     # How-to questions
    CLARIFICATION = "clarification"  # Follow-up clarification
    AMBIGUOUS = "ambiguous"       # Unclear intent
    OFF_TOPIC = "off_topic"       # Not related to documents
    GREETING = "greeting"         # Social/greeting message
    INVALID = "invalid"           # Too short or nonsensical


class ConfidenceLevel(Enum):
    """Confidence levels for responses."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class QueryValidation:
    """Result of query validation."""
    is_valid: bool
    query_type: QueryType
    cleaned_query: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class RelevanceCheck:
    """Result of relevance checking."""
    has_relevant_context: bool
    confidence: ConfidenceLevel
    avg_relevance: float
    top_relevance: float
    relevant_chunks: int
    total_chunks: int
    issues: List[str] = field(default_factory=list)


@dataclass
class GuardrailResult:
    """Complete guardrail check result."""
    passed: bool
    query_validation: QueryValidation
    relevance_check: Optional[RelevanceCheck] = None
    should_use_fallback: bool = False
    fallback_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryGuardrails:
    """
    Guardrails for validating and preprocessing user queries.
    """
    
    # Greeting patterns
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))[\s!.,]*$",
        r"^how\s+are\s+you",
        r"^what'?s?\s+up",
    ]
    
    # Question word patterns for detecting query type
    FACTUAL_PATTERNS = [
        r"^(what|who|when|where|which)\s+(is|are|was|were|did|does|do)",
        r"^(tell\s+me|explain)\s+(what|who|when|where)",
    ]
    
    PROCEDURAL_PATTERNS = [
        r"^how\s+(do|can|should|would|to)",
        r"^(steps?\s+to|process\s+(of|for)|guide\s+(to|for))",
    ]
    
    COMPARATIVE_PATTERNS = [
        r"(compare|comparison|difference|between|versus|vs\.?)",
        r"(better|worse|more|less)\s+than",
    ]
    
    # Ambiguous/vague query indicators
    VAGUE_PATTERNS = [
        r"^(it|this|that|these|those)$",
        r"^(something|anything|stuff|things?)$",
        r"^(idk|dunno|unsure)$",
    ]
    
    def __init__(
        self,
        min_query_length: int = MIN_QUERY_LENGTH,
        max_query_length: int = MAX_QUERY_LENGTH,
    ):
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length
    
    def validate(self, query: str) -> QueryValidation:
        """
        Validate and classify a user query.
        
        Args:
            query: The user's query string
            
        Returns:
            QueryValidation with results
        """
        issues = []
        suggestions = []
        
        # Clean the query
        cleaned = self._clean_query(query)
        
        # Check length
        if len(cleaned) < self.min_query_length:
            return QueryValidation(
                is_valid=False,
                query_type=QueryType.INVALID,
                cleaned_query=cleaned,
                issues=["Query is too short. Please provide more detail."],
                suggestions=["Try asking a complete question about your documents."],
            )
        
        if len(cleaned) > self.max_query_length:
            cleaned = cleaned[:self.max_query_length]
            issues.append("Query was truncated due to length.")
        
        # Check for greeting
        if self._is_greeting(cleaned):
            return QueryValidation(
                is_valid=True,
                query_type=QueryType.GREETING,
                cleaned_query=cleaned,
                issues=[],
                suggestions=[],
            )
        
        # Check for ambiguous/vague queries
        if self._is_vague(cleaned):
            return QueryValidation(
                is_valid=True,
                query_type=QueryType.AMBIGUOUS,
                cleaned_query=cleaned,
                issues=["Query seems vague or ambiguous."],
                suggestions=[
                    "Could you be more specific about what you're looking for?",
                    "Try including more context or keywords.",
                ],
            )
        
        # Classify query type
        query_type = self._classify_query(cleaned)
        
        return QueryValidation(
            is_valid=True,
            query_type=query_type,
            cleaned_query=cleaned,
            issues=issues,
            suggestions=suggestions,
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize a query."""
        # Strip whitespace
        cleaned = query.strip()
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned
    
    def _is_greeting(self, query: str) -> bool:
        """Check if query is a greeting."""
        query_lower = query.lower()
        for pattern in self.GREETING_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return True
        return False
    
    def _is_vague(self, query: str) -> bool:
        """Check if query is too vague."""
        query_lower = query.lower()
        for pattern in self.VAGUE_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return True
        return False
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify the type of query."""
        query_lower = query.lower()
        
        # Check procedural first (how-to)
        for pattern in self.PROCEDURAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryType.PROCEDURAL
        
        # Check comparative
        for pattern in self.COMPARATIVE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryType.COMPARATIVE
        
        # Check factual
        for pattern in self.FACTUAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryType.FACTUAL
        
        # Default to exploratory
        return QueryType.EXPLORATORY


class RelevanceGuardrails:
    """
    Guardrails for checking relevance of retrieved context.
    """
    
    def __init__(
        self,
        min_relevance: float = MIN_RELEVANCE_THRESHOLD,
        high_confidence: float = HIGH_CONFIDENCE_THRESHOLD,
    ):
        self.min_relevance = min_relevance
        self.high_confidence = high_confidence
    
    def check(
        self,
        sources: List[Dict[str, Any]],
        query: str = None,
    ) -> RelevanceCheck:
        """
        Check if retrieved sources are relevant enough.
        
        Args:
            sources: List of source dicts with 'relevance' scores
            query: Optional query for additional checks
            
        Returns:
            RelevanceCheck with results
        """
        if not sources:
            return RelevanceCheck(
                has_relevant_context=False,
                confidence=ConfidenceLevel.NONE,
                avg_relevance=0.0,
                top_relevance=0.0,
                relevant_chunks=0,
                total_chunks=0,
                issues=["No relevant documents found in the knowledge base."],
            )
        
        # Calculate metrics
        relevance_scores = [s.get("relevance", 0) for s in sources]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        top_relevance = max(relevance_scores)
        
        # Count relevant chunks
        relevant_chunks = sum(1 for s in relevance_scores if s >= self.min_relevance)
        
        # Determine confidence level
        issues = []
        if top_relevance >= self.high_confidence and avg_relevance >= self.min_relevance:
            confidence = ConfidenceLevel.HIGH
        elif top_relevance >= self.min_relevance:
            confidence = ConfidenceLevel.MEDIUM
            if avg_relevance < self.min_relevance:
                issues.append("Some retrieved context may not be directly relevant.")
        elif top_relevance > 0.1:
            confidence = ConfidenceLevel.LOW
            issues.append("Retrieved context has low relevance. Answer may be unreliable.")
        else:
            confidence = ConfidenceLevel.NONE
            issues.append("No sufficiently relevant context found.")
        
        return RelevanceCheck(
            has_relevant_context=relevant_chunks > 0,
            confidence=confidence,
            avg_relevance=avg_relevance,
            top_relevance=top_relevance,
            relevant_chunks=relevant_chunks,
            total_chunks=len(sources),
            issues=issues,
        )


class FallbackResponses:
    """
    Fallback responses for various scenarios.
    """
    
    NO_CONTEXT = (
        "I couldn't find relevant information in the documents to answer your question. "
        "Could you try rephrasing or asking about a different topic?"
    )
    
    LOW_CONFIDENCE = (
        "Based on the available documents, here's what I found, though I'm not entirely "
        "confident this fully answers your question: "
    )
    
    AMBIGUOUS_QUERY = (
        "Your question seems a bit broad. Could you provide more specific details? "
        "For example:\n"
        "- What specific topic are you interested in?\n"
        "- Are you looking for a definition, process, or comparison?"
    )
    
    GREETING_RESPONSE = (
        "Hello! I'm your document assistant. I can answer questions about the documents "
        "in your knowledge base. What would you like to know?"
    )
    
    OFF_TOPIC = (
        "This question doesn't seem to relate to the documents in the knowledge base. "
        "I can only answer questions based on the uploaded documents. "
        "Would you like to ask something else?"
    )
    
    ERROR_RESPONSE = (
        "I encountered an issue processing your request. Please try again or "
        "rephrase your question."
    )
    
    @classmethod
    def get_fallback(
        cls,
        query_type: QueryType,
        confidence: ConfidenceLevel,
    ) -> Optional[str]:
        """Get appropriate fallback response."""
        if query_type == QueryType.GREETING:
            return cls.GREETING_RESPONSE
        
        if query_type == QueryType.AMBIGUOUS:
            return cls.AMBIGUOUS_QUERY
        
        if query_type == QueryType.INVALID:
            return "Please provide a more detailed question."
        
        if confidence == ConfidenceLevel.NONE:
            return cls.NO_CONTEXT
        
        return None
    
    @classmethod
    def get_confidence_prefix(cls, confidence: ConfidenceLevel) -> str:
        """Get a prefix based on confidence level."""
        if confidence == ConfidenceLevel.LOW:
            return cls.LOW_CONFIDENCE
        return ""


class SourceAttributionGuardrails:
    """
    Guardrails for source attribution in responses.
    """
    
    @staticmethod
    def format_citations(sources: List[Dict[str, Any]]) -> str:
        """Format sources as citations."""
        if not sources:
            return ""
        
        unique_docs = list(set(s.get("document", "Unknown") for s in sources))
        
        if len(unique_docs) == 1:
            return f"\n\n📄 Source: {unique_docs[0]}"
        else:
            citation = "\n\n📄 Sources:\n"
            for doc in unique_docs:
                citation += f"  • {doc}\n"
            return citation.rstrip()
    
    @staticmethod
    def verify_attribution(
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str]]:
        """
        Verify that the answer is grounded in the sources.
        
        This is a simple heuristic check - production systems
        would use more sophisticated NLI models.
        """
        issues = []
        
        if not sources:
            return False, ["No sources available for verification."]
        
        # Get all source text
        source_text = " ".join(s.get("text", "") for s in sources).lower()
        
        # Check if answer mentions things not in sources
        # This is a simplified check
        answer_words = set(answer.lower().split())
        source_words = set(source_text.split())
        
        # Calculate overlap (very simplified)
        overlap = len(answer_words & source_words)
        answer_unique = len(answer_words - source_words)
        
        if overlap < 5 and len(answer_words) > 10:
            issues.append("Answer may contain information not found in sources.")
        
        # Check for hedging language (indicates potential hallucination)
        hedging_phrases = [
            "i think", "probably", "might be", "could be", 
            "i believe", "it seems", "possibly"
        ]
        for phrase in hedging_phrases:
            if phrase in answer.lower():
                issues.append(f"Answer contains hedging language: '{phrase}'")
                break
        
        return len(issues) == 0, issues


class RAGGuardrails:
    """
    Complete guardrails system for RAG pipeline.
    
    Combines all guardrail types for comprehensive protection.
    """
    
    def __init__(
        self,
        min_relevance: float = MIN_RELEVANCE_THRESHOLD,
        high_confidence: float = HIGH_CONFIDENCE_THRESHOLD,
        min_query_length: int = MIN_QUERY_LENGTH,
    ):
        self.query_guardrails = QueryGuardrails(min_query_length=min_query_length)
        self.relevance_guardrails = RelevanceGuardrails(
            min_relevance=min_relevance,
            high_confidence=high_confidence,
        )
    
    def check_query(self, query: str) -> QueryValidation:
        """Validate a query before processing."""
        return self.query_guardrails.validate(query)
    
    def check_relevance(
        self,
        sources: List[Dict[str, Any]],
        query: str = None,
    ) -> RelevanceCheck:
        """Check relevance of retrieved context."""
        return self.relevance_guardrails.check(sources, query)
    
    def full_check(
        self,
        query: str,
        sources: List[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """
        Perform complete guardrail check.
        
        Args:
            query: User's query
            sources: Retrieved source chunks (optional, for pre-response check)
            
        Returns:
            GuardrailResult with comprehensive check results
        """
        warnings = []
        
        # Step 1: Validate query
        query_validation = self.check_query(query)
        
        if not query_validation.is_valid:
            return GuardrailResult(
                passed=False,
                query_validation=query_validation,
                should_use_fallback=True,
                fallback_message=FallbackResponses.get_fallback(
                    query_validation.query_type,
                    ConfidenceLevel.NONE,
                ),
                warnings=warnings,
            )
        
        # Handle special query types
        if query_validation.query_type == QueryType.GREETING:
            return GuardrailResult(
                passed=True,
                query_validation=query_validation,
                should_use_fallback=True,
                fallback_message=FallbackResponses.GREETING_RESPONSE,
                warnings=warnings,
            )
        
        if query_validation.query_type == QueryType.AMBIGUOUS:
            warnings.append("Query is ambiguous - results may be imprecise.")
        
        # Step 2: Check relevance if sources provided
        relevance_check = None
        should_use_fallback = False
        fallback_message = None
        
        if sources is not None:
            relevance_check = self.check_relevance(sources, query)
            
            if not relevance_check.has_relevant_context:
                should_use_fallback = True
                fallback_message = FallbackResponses.NO_CONTEXT
            elif relevance_check.confidence == ConfidenceLevel.LOW:
                warnings.append("Low confidence in retrieved context.")
            
            warnings.extend(relevance_check.issues)
        
        return GuardrailResult(
            passed=True,
            query_validation=query_validation,
            relevance_check=relevance_check,
            should_use_fallback=should_use_fallback,
            fallback_message=fallback_message,
            warnings=warnings,
            metadata={
                "query_type": query_validation.query_type.value,
                "confidence": relevance_check.confidence.value if relevance_check else None,
            },
        )
    
    def post_process_response(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        add_citations: bool = True,
        verify_grounding: bool = True,
    ) -> Tuple[str, List[str]]:
        """
        Post-process a response with guardrails.
        
        Args:
            answer: The generated answer
            sources: Source chunks used
            add_citations: Whether to add source citations
            verify_grounding: Whether to verify answer is grounded
            
        Returns:
            Tuple of (processed_answer, warnings)
        """
        warnings = []
        processed = answer
        
        # Verify grounding
        if verify_grounding:
            is_grounded, grounding_issues = SourceAttributionGuardrails.verify_attribution(
                answer, sources
            )
            if not is_grounded:
                warnings.extend(grounding_issues)
        
        # Add citations
        if add_citations and sources:
            citations = SourceAttributionGuardrails.format_citations(sources)
            processed = answer + citations
        
        return processed, warnings


# Singleton instance
_guardrails: Optional[RAGGuardrails] = None


def get_guardrails() -> RAGGuardrails:
    """Get the default guardrails instance."""
    global _guardrails
    if _guardrails is None:
        _guardrails = RAGGuardrails()
    return _guardrails


