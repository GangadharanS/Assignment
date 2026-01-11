"""
LLM integration module for interacting with open-source language models.

Supports:
- Ollama (local LLM server)
- Hugging Face Transformers (local models)
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from blob_storage.config import config


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class LLMResponse:
    """Represents an LLM response."""
    content: str
    model: str
    usage: Dict[str, int] = None
    raw_response: Dict[str, Any] = None


class LLMBase(ABC):
    """Base class for LLM integrations."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Have a multi-turn conversation with the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        pass


class OllamaLLM(LLMBase):
    """
    Ollama LLM integration for running open-source models locally.
    
    Ollama supports models like:
    - llama2, llama3
    - mistral, mixtral
    - codellama
    - phi
    - gemma
    - qwen
    And many more: https://ollama.ai/library
    """
    
    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        timeout: float = 120.0,
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            model: Model name (e.g., "llama3.2", "mistral", "phi")
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
        """
        self.model = model or os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)
        self.timeout = timeout
        self._client = None
    
    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = self.client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = self.client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Generate a response from Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Randomness (0-1)
            max_tokens: Maximum response length
            
        Returns:
            LLMResponse with the generated text
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        
        return self.chat(messages, temperature, max_tokens)
    
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Have a chat conversation with Ollama.
        
        Args:
            messages: List of Message objects
            temperature: Randomness (0-1)
            max_tokens: Maximum response length
            
        Returns:
            LLMResponse with the generated text
        """
        # Format messages for Ollama
        formatted_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        response = self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=self.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
            raw_response=data,
        )
    
    async def agenerate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Async version of generate."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        
        return await self.achat(messages, temperature, max_tokens)
    
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Async version of chat."""
        formatted_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        ) as client:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=self.model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                },
                raw_response=data,
            )


class HuggingFaceLLM(LLMBase):
    """
    Hugging Face Transformers LLM integration for running models locally.
    
    Note: Requires additional installation:
        pip install transformers torch accelerate
    """
    
    DEFAULT_MODEL = "microsoft/phi-2"
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "auto",
        load_in_8bit: bool = False,
    ):
        """
        Initialize Hugging Face LLM.
        
        Args:
            model_name: Model name from Hugging Face Hub
            device: Device to run on ("auto", "cpu", "cuda", "mps")
            load_in_8bit: Use 8-bit quantization to reduce memory
        """
        self.model_name = model_name or os.getenv("HF_MODEL", self.DEFAULT_MODEL)
        self.device = device
        self.load_in_8bit = load_in_8bit
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._pipeline is not None:
            return
        
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFaceLLM. "
                "Install with: pip install transformers torch"
            )
        
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device_map=self.device if self.device != "auto" else None,
            torch_dtype="auto",
        )
    
    def is_available(self) -> bool:
        """Check if transformers is available."""
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a response using Hugging Face."""
        self._load_model()
        
        # Build the full prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\n\nAssistant:"
        
        outputs = self._pipeline(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,
        )
        
        content = outputs[0]["generated_text"].strip()
        
        return LLMResponse(
            content=content,
            model=self.model_name,
            usage={"total_tokens": len(full_prompt.split()) + len(content.split())},
        )
    
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Chat using Hugging Face (converts to single prompt)."""
        # Convert messages to a single prompt
        parts = []
        system_prompt = None
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        
        prompt = "\n\n".join(parts)
        return self.generate(prompt, system_prompt, temperature, max_tokens)


# Default system prompts for RAG
DEFAULT_RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

INSTRUCTIONS:
1. Answer the user's question using ONLY the information from the provided context
2. If the context doesn't contain enough information to answer, say so clearly
3. Be concise and direct in your responses
4. If you quote from the context, indicate which document it came from
5. Do not make up information that isn't in the context

CONTEXT FORMAT:
The context will be provided as chunks from various documents, with source information."""

DETAILED_RAG_SYSTEM_PROMPT = """You are an expert AI assistant specialized in answering questions based on document content.

Your role is to:
1. Carefully analyze the provided context chunks from the document database
2. Synthesize information from multiple chunks when relevant
3. Provide accurate, well-structured answers

Guidelines:
- ONLY use information present in the provided context
- If the context is insufficient, clearly state what information is missing
- When multiple chunks contain relevant info, synthesize them coherently  
- Cite document sources when referencing specific information
- Be precise and avoid speculation beyond what the context supports
- Format responses with clear structure when appropriate

Response format:
- Start with a direct answer to the question
- Provide supporting details from the context
- Note any limitations or caveats if applicable"""

CONVERSATIONAL_RAG_SYSTEM_PROMPT = """You are a friendly and knowledgeable assistant helping users find information in their documents.

When answering:
- Be conversational but accurate
- Draw only from the provided context
- Acknowledge when you're unsure or the context is limited
- Suggest follow-up questions if appropriate
- Keep responses clear and well-organized"""


# Factory function for getting LLM
def get_llm(
    provider: str = "ollama",
    model: str = None,
    **kwargs,
) -> LLMBase:
    """
    Get an LLM instance.
    
    Args:
        provider: "ollama" or "huggingface"
        model: Model name
        **kwargs: Additional arguments for the LLM
        
    Returns:
        LLM instance
    """
    provider = provider.lower()
    
    if provider == "ollama":
        return OllamaLLM(model=model, **kwargs)
    elif provider in ("huggingface", "hf", "transformers"):
        return HuggingFaceLLM(model_name=model, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# Singleton instance
_llm: Optional[LLMBase] = None


def get_default_llm() -> LLMBase:
    """Get the default LLM instance (Ollama)."""
    global _llm
    if _llm is None:
        _llm = OllamaLLM()
    return _llm


