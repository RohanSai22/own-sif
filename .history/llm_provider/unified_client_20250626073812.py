"""Unified LLM client for Prometheus 2.0 supporting multiple providers."""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

# Import LLM clients
try:
    import openai
except ImportError:
    openai = None

try:
    import groq
except ImportError:
    groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from config import llm_config

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    response_time: Optional[float] = None

class UnifiedLLMClient:
    """Unified client for multiple LLM providers with fallback support."""
    
    def __init__(self):
        self.providers = {}
        self.default_model = llm_config.default_model
        self.fallback_model = llm_config.fallback_model
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers."""
        
        # Initialize OpenAI
        if openai and llm_config.openai_api_key:
            try:
                self.providers["openai"] = openai.OpenAI(
                    api_key=llm_config.openai_api_key
                )
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Initialize Groq
        if groq and llm_config.groq_api_key:
            try:
                self.providers["groq"] = groq.Groq(
                    api_key=llm_config.groq_api_key
                )
                logger.info("Groq client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
        
        # Initialize Gemini
        if genai and llm_config.gemini_api_key:
            try:
                genai.configure(api_key=llm_config.gemini_api_key)
                self.providers["gemini"] = genai
                logger.info("Gemini client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        
        if not self.providers:
            raise RuntimeError("No LLM providers could be initialized. Check your API keys.")
        
        logger.info(f"Initialized providers: {list(self.providers.keys())}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response using the specified or default model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (format: "provider/model_name")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt to prepend
            
        Returns:
            LLMResponse object with the generated content
        """
        # Use default model if none specified
        if not model:
            model = self.default_model
        
        # Parse provider and model name
        if "/" in model:
            provider_name, model_name = model.split("/", 1)
        else:
            # Try to infer provider from model name
            provider_name = self._infer_provider(model)
            model_name = model
        
        # Add system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Set defaults
        temperature = temperature or llm_config.temperature
        max_tokens = max_tokens or llm_config.max_tokens
        
        # Try the requested provider first
        try:
            return self._generate_with_provider(
                provider_name, model_name, messages, temperature, max_tokens
            )
        except Exception as e:
            logger.warning(f"Failed to generate with {provider_name}/{model_name}: {e}")
            
            # Try fallback model if different from the failed one
            if model != self.fallback_model:
                try:
                    fallback_provider, fallback_model_name = self.fallback_model.split("/", 1)
                    logger.info(f"Trying fallback model: {self.fallback_model}")
                    return self._generate_with_provider(
                        fallback_provider, fallback_model_name, messages, temperature, max_tokens
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            # If all else fails, try any available provider
            for available_provider in self.providers.keys():
                if available_provider != provider_name:
                    try:
                        logger.info(f"Trying alternative provider: {available_provider}")
                        return self._generate_with_provider(
                            available_provider, model_name, messages, temperature, max_tokens
                        )
                    except Exception as alt_error:
                        logger.warning(f"Alternative provider {available_provider} failed: {alt_error}")
                        continue
            
            raise RuntimeError(f"All LLM providers failed. Last error: {e}")
    
    def _generate_with_provider(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Generate with a specific provider."""
        start_time = time.time()
        
        if provider == "openai":
            return self._generate_openai(model, messages, temperature, max_tokens, start_time)
        elif provider == "groq":
            return self._generate_groq(model, messages, temperature, max_tokens, start_time)
        elif provider == "gemini":
            return self._generate_gemini(model, messages, temperature, max_tokens, start_time)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _generate_openai(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> LLMResponse:
        """Generate using OpenAI API."""
        client = self.providers["openai"]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            provider="openai",
            tokens_used=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason,
            response_time=time.time() - start_time
        )
    
    def _generate_groq(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> LLMResponse:
        """Generate using Groq API."""
        client = self.providers["groq"]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            provider="groq",
            tokens_used=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason,
            response_time=time.time() - start_time
        )
    
    def _generate_gemini(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> LLMResponse:
        """Generate using Gemini API."""
        genai = self.providers["gemini"]
        
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        prompt = "\n\n".join(prompt_parts)
        
        # Use appropriate Gemini model
        if "gemini" not in model.lower():
            model = "gemini-pro"
        
        model_instance = genai.GenerativeModel(model)
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        response = model_instance.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return LLMResponse(
            content=response.text,
            model=model,
            provider="gemini",
            tokens_used=None,  # Gemini doesn't provide token count easily
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
            response_time=time.time() - start_time
        )
    
    def _infer_provider(self, model: str) -> str:
        """Infer provider from model name."""
        model_lower = model.lower()
        
        if "gpt" in model_lower or "davinci" in model_lower:
            return "openai"
        elif "llama" in model_lower or "mixtral" in model_lower or "qwen" in model_lower:
            return "groq"
        elif "gemini" in model_lower:
            return "gemini"
        else:
            # Default to first available provider
            return list(self.providers.keys())[0]
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models for each provider."""
        models = {}
        
        if "openai" in self.providers:
            models["openai"] = [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ]
        
        if "groq" in self.providers:
            models["groq"] = [
                "qwen/qwen3-32b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768"
            ]
        
        if "gemini" in self.providers:
            models["gemini"] = [
                "gemini-pro",
                "gemini-pro-vision"
            ]
        
        return models
    
    def test_providers(self) -> Dict[str, Dict[str, Any]]:
        """Test all available providers."""
        results = {}
        test_messages = [{"role": "user", "content": "Say 'Hello, I am working!' in exactly those words."}]
        
        for provider_name in self.providers.keys():
            try:
                # Get a simple model for each provider
                if provider_name == "openai":
                    test_model = "gpt-3.5-turbo"
                elif provider_name == "groq":
                    test_model = "meta-llama/llama-4-maverick-17b-128e-instruct"
                elif provider_name == "gemini":
                    test_model = "gemini-pro"
                else:
                    continue
                
                response = self._generate_with_provider(
                    provider_name, test_model, test_messages, 0.1, 50
                )
                
                results[provider_name] = {
                    "status": "success",
                    "model": test_model,
                    "response_time": response.response_time,
                    "content": response.content
                }
                
            except Exception as e:
                results[provider_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results

# Global client instance
llm_client = UnifiedLLMClient()

if __name__ == "__main__":
    # Test the client
    import os
    
    # Load test environment
    os.environ["GROQ_API_KEY"] = "your_groq_key"  # Replace with actual key for testing
    
    try:
        client = UnifiedLLMClient()
        
        # Test generation
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = client.generate(messages)
        print(f"Response from {response.provider}/{response.model}:")
        print(response.content)
        print(f"Tokens used: {response.tokens_used}")
        print(f"Response time: {response.response_time:.2f}s")
        
        # Test all providers
        print("\nTesting all providers:")
        test_results = client.test_providers()
        for provider, result in test_results.items():
            print(f"{provider}: {result['status']}")
            if result['status'] == 'success':
                print(f"  Response: {result['content'][:50]}...")
        
    except Exception as e:
        print(f"Error: {e}")
