"""Async client for Ollama's /api/chat, narrowed to what the service needs."""

from __future__ import annotations

from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Ollama is down, refused the request, or sent something unusable. `status` is the HTTP status it answered with."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class OllamaClient:
    def __init__(self, url: str, model: str, timeout: float = 120.0, keep_alive: str = "30m",
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model = model
        self.num_thread: int | None = None  # CPU threads Ollama uses per request; None = Ollama decides
        self._keep_alive = keep_alive
        self._http = httpx.AsyncClient(base_url=url.rstrip("/"), timeout=timeout, transport=transport)

    async def close(self) -> None:
        await self._http.aclose()

    async def model_ready(self) -> bool:
        """True when Ollama answers and has the model pulled."""
        try:
            response = await self._http.get("/api/tags", timeout=3.0)
            names = {entry.get("name") for entry in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return False
        return self.model in names or f"{self.model}:latest" in names

    async def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
                   schema: dict[str, Any] | None, thinking: bool, temperature: float, max_tokens: int) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "think": thinking,
                                "keep_alive": self._keep_alive,
                                "options": {"temperature": temperature, "num_predict": max_tokens}}
        if self.num_thread:
            body["options"]["num_thread"] = self.num_thread
        if tools:
            body["tools"] = tools
        if schema is not None:
            body["format"] = schema
        try:
            response = await self._http.post("/api/chat", json=body)
        except httpx.TimeoutException as error:
            raise OllamaError("Ollama timed out", 504) from error
        except httpx.HTTPError as error:
            raise OllamaError(f"Ollama could not be reached: {error}") from error
        if not response.is_success:
            try:
                detail = response.json().get("error")
            except ValueError:
                detail = None
            raise OllamaError(f"Ollama refused the request ({response.status_code}): {detail or response.text[:200]}",
                              response.status_code)
        return response.json()
