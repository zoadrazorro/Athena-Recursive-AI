"""
LM Studio API client for interfacing with local model endpoints.

Provides async HTTP communication with LM Studio's OpenAI-compatible API.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
import aiohttp
from loguru import logger

from .schemas import HealthCheck


class LMStudioClient:
    """
    Async client for communicating with LM Studio model endpoints.

    Supports OpenAI-compatible chat completion API with streaming,
    health checks, and automatic retry logic.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Initialize the LM Studio client.

        Args:
            base_url: Base URL for the LM Studio endpoint (e.g., http://localhost:1234/v1)
            model_name: Name of the model loaded in LM Studio
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def ensure_session(self):
        """Ensure an active session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self):
        """Close the client session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to LM Studio.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional parameters for the API

        Returns:
            Response dict from the API

        Raises:
            aiohttp.ClientError: If the request fails after retries
        """
        await self.ensure_session()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }

        url = f"{self.base_url}/chat/completions"

        for attempt in range(self.max_retries):
            try:
                start_time = datetime.utcnow()

                async with self.session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.warning(
                            f"LM Studio request failed (attempt {attempt + 1}): "
                            f"Status {response.status}, Error: {error_text}"
                        )

                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            raise aiohttp.ClientError(
                                f"Request failed with status {response.status}: {error_text}"
                            )

                    result = await response.json()

                    # Log timing
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    logger.debug(
                        f"LM Studio request completed in {elapsed:.2f}s "
                        f"(model: {self.model_name})"
                    )

                    return result

            except asyncio.TimeoutError:
                logger.warning(
                    f"LM Studio request timeout (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

            except aiohttp.ClientError as e:
                logger.error(
                    f"LM Studio client error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError(f"Failed to complete request after {self.max_retries} attempts")

    async def health_check(self) -> HealthCheck:
        """
        Check the health of this LM Studio endpoint.

        Returns:
            HealthCheck object with status information
        """
        await self.ensure_session()

        start_time = datetime.utcnow()

        try:
            # Try a simple completion request
            messages = [{"role": "user", "content": "Hi"}]

            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 5
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                elapsed = (datetime.utcnow() - start_time).total_seconds()

                if response.status == 200:
                    return HealthCheck(
                        endpoint=self.base_url,
                        model_name=self.model_name,
                        status="healthy",
                        response_time=elapsed
                    )
                else:
                    error_text = await response.text()
                    return HealthCheck(
                        endpoint=self.base_url,
                        model_name=self.model_name,
                        status="degraded",
                        response_time=elapsed,
                        error=f"Status {response.status}: {error_text}"
                    )

        except asyncio.TimeoutError:
            return HealthCheck(
                endpoint=self.base_url,
                model_name=self.model_name,
                status="degraded",
                error="Health check timeout"
            )

        except Exception as e:
            return HealthCheck(
                endpoint=self.base_url,
                model_name=self.model_name,
                status="unavailable",
                error=str(e)
            )

    async def extract_response_text(self, response: Dict[str, Any]) -> str:
        """
        Extract the text content from an API response.

        Args:
            response: Response dict from chat_completion

        Returns:
            Extracted text content
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract response text: {e}")
            logger.debug(f"Response structure: {response}")
            raise ValueError(f"Invalid response structure: {e}")


class ModelPool:
    """
    Manages a pool of LM Studio clients for different models.

    Provides connection pooling, health monitoring, and load balancing
    for multiple model endpoints.
    """

    def __init__(self):
        """Initialize the model pool."""
        self.clients: Dict[str, LMStudioClient] = {}
        self._health_status: Dict[str, HealthCheck] = {}

    def register_client(
        self,
        name: str,
        base_url: str,
        model_name: str,
        **kwargs
    ) -> None:
        """
        Register a new LM Studio client in the pool.

        Args:
            name: Unique identifier for this client
            base_url: Base URL for the endpoint
            model_name: Name of the model
            **kwargs: Additional arguments for LMStudioClient
        """
        if name in self.clients:
            logger.warning(f"Overwriting existing client: {name}")

        self.clients[name] = LMStudioClient(
            base_url=base_url,
            model_name=model_name,
            **kwargs
        )

        logger.info(f"Registered client '{name}' at {base_url} (model: {model_name})")

    def get_client(self, name: str) -> LMStudioClient:
        """
        Get a client from the pool.

        Args:
            name: Client identifier

        Returns:
            LMStudioClient instance

        Raises:
            KeyError: If client not found
        """
        if name not in self.clients:
            raise KeyError(f"Client '{name}' not found in pool")

        return self.clients[name]

    async def health_check_all(self) -> Dict[str, HealthCheck]:
        """
        Run health checks on all registered clients.

        Returns:
            Dict mapping client names to HealthCheck results
        """
        tasks = {}

        for name, client in self.clients.items():
            tasks[name] = client.health_check()

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                self._health_status[name] = HealthCheck(
                    endpoint=self.clients[name].base_url,
                    model_name=self.clients[name].model_name,
                    status="unavailable",
                    error=str(result)
                )
            else:
                self._health_status[name] = result

        return self._health_status

    def get_health_status(self, name: str) -> Optional[HealthCheck]:
        """Get the last health check result for a client."""
        return self._health_status.get(name)

    async def close_all(self) -> None:
        """Close all client sessions."""
        tasks = [client.close() for client in self.clients.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Closed all client sessions")
