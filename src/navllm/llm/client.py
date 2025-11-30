"""
LLM Client: Interface for calling external LLM APIs.

Handles preference scoring and explanation generation as described in Section 5.3.
The LLM operates only on schema-level descriptions and summary statistics.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    raw_response: Any
    success: bool
    error: Optional[str] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Send prompt to LLM and get response."""
        pass
    
    @abstractmethod
    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Send prompt and parse JSON response."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client (GPT-4, GPT-3.5)."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo", 
                 temperature: float = 0.3, max_retries: int = 2):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Send prompt to OpenAI API."""
        client = self._get_client()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", 1000),
                )
                content = response.choices[0].message.content
                return LLMResponse(content=content, raw_response=response, success=True)
            except Exception as e:
                logger.warning(f"LLM API call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    return LLMResponse(
                        content="", raw_response=None, 
                        success=False, error=str(e)
                    )
        return LLMResponse(content="", raw_response=None, success=False, error="Max retries exceeded")
    
    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Send prompt and parse JSON response."""
        response = self.complete(prompt, **kwargs)
        
        if not response.success:
            logger.error(f"LLM call failed: {response.error}")
            return {}
        
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response.content}")
            return {}


class DeepSeekClient(LLMClient):
    """DeepSeek API client (OpenAI-compatible)."""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 temperature: float = 0.3, max_retries: int = 2):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        client = self._get_client()
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", 1000),
                )
                content = response.choices[0].message.content
                return LLMResponse(content=content, raw_response=response, success=True)
            except Exception as e:
                logger.warning(f"DeepSeek API call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    return LLMResponse(content="", raw_response=None, success=False, error=str(e))
        return LLMResponse(content="", raw_response=None, success=False, error="Max retries exceeded")
    
    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        response = self.complete(prompt, **kwargs)
        if not response.success:
            logger.error(f"DeepSeek call failed: {response.error}")
            return {}
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {}


class GeminiClient(LLMClient):
    """Google Gemini API client."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash",
                 temperature: float = 0.3, max_retries: int = 2,
                 requests_per_minute: int = 14):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.requests_per_minute = requests_per_minute
        self._client = None
        self._last_request_time = 0
        self._min_interval = 60.0 / requests_per_minute
    
    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        return self._client
    
    def _rate_limit(self):
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
        return self._client
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        client = self._get_client()
        
        for attempt in range(self.max_retries + 1):
            try:
                self._rate_limit()
                response = client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": kwargs.get("temperature", self.temperature),
                        "max_output_tokens": kwargs.get("max_tokens", 1000),
                    }
                )
                content = response.text
                return LLMResponse(content=content, raw_response=response, success=True)
            except Exception as e:
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    return LLMResponse(content="", raw_response=None, success=False, error=str(e))
        return LLMResponse(content="", raw_response=None, success=False, error="Max retries exceeded")
    
    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        response = self.complete(prompt, **kwargs)
        if not response.success:
            logger.error(f"Gemini call failed: {response.error}")
            return {}
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {}


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls."""
    
    def __init__(self, default_scores: List[float] = None, simulate_preferences: bool = True):
        self.default_scores = default_scores or [0.5]
        self.simulate_preferences = simulate_preferences
        self.call_count = 0
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            content="Mock response",
            raw_response=None,
            success=True
        )
    
    def complete_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        self.call_count += 1
        
        # Count candidates in prompt
        num_candidates = prompt.lower().count("grouped by") + prompt.lower().count("drill down") + prompt.lower().count("roll up")
        if num_candidates == 0:
            num_candidates = max(1, prompt.count("\n") // 3)
        
        if self.simulate_preferences:
            # Simulate realistic preference scoring based on prompt content
            import random
            scores = []
            prompt_lower = prompt.lower()
            
            for i in range(num_candidates):
                base_score = 0.5 + random.uniform(-0.2, 0.2)
                
                # Boost score if drill-down is mentioned (users often want more detail)
                if "drill" in prompt_lower:
                    base_score += 0.1
                
                # Boost if keywords match
                keywords = ["sales", "revenue", "decline", "drop", "increase", "pattern", 
                           "anomaly", "defect", "quality", "region", "time", "product"]
                for kw in keywords:
                    if kw in prompt_lower:
                        base_score += random.uniform(0, 0.05)
                
                # Add variation between candidates
                base_score += (num_candidates - i) * 0.05
                
                scores.append(min(1.0, max(0.0, base_score)))
            
            return {"scores": scores}
        else:
            scores = self.default_scores * num_candidates
            return {"scores": scores[:num_candidates] if num_candidates > 0 else self.default_scores}
