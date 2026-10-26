"""HTTP face of MiniCPM5-1B (served by Ollama).

POST /v1/chat {"messages", "tools", "schema", "thinking", "temperature", "max_tokens"} runs one model turn. With
`tools` the model may answer with `tool_calls` for the caller to run; the caller appends the results as `tool`
messages and calls again, so the loop lives in the caller. With a JSON `schema` the reply is constrained to it and
returned parsed. Thinking is on unless the caller turns it off. GET /health reports whether Ollama has the model."""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from minicpm_service.ollama import OllamaClient, OllamaError

_log = logging.getLogger(__name__)

DEFAULT_MODEL = "hf.co/openbmb/MiniCPM5-1B-GGUF:Q4_K_M"


class ToolCall(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """`tool_calls` belongs to an assistant turn; a `tool` message carries one call's result (`tool_name` names it)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field("", max_length=100_000)
    tool_calls: list[ToolCall] | None = None
    tool_name: str | None = Field(None, max_length=64)


class Tool(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    description: str = Field("", max_length=2000)
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    messages: list[Message] = Field(min_length=1, max_length=64)
    tools: list[Tool] | None = Field(None, max_length=32)
    json_schema: dict[str, Any] | None = Field(None, alias="schema")
    thinking: bool = True
    temperature: float = Field(0.0, ge=0, le=2)
    max_tokens: int = Field(2048, ge=1, le=8192)


def _ollama_message(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [{"function": {"name": call.name, "arguments": call.arguments}} for call in message.tool_calls]
    if message.tool_name:
        out["tool_name"] = message.tool_name
    return out


def _ollama_tool(tool: Tool) -> dict[str, Any]:
    return {"type": "function", "function": tool.model_dump()}


def create_app(client: OllamaClient | None = None, token: str | None = None) -> FastAPI:
    """`token` (MINICPM_TOKEN), when set, must arrive in the X-Internal-Token header."""
    client = client or OllamaClient(os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
                                    os.environ.get("MINICPM_MODEL", DEFAULT_MODEL))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await client.close()

    def authorize(x_internal_token: str = Header("")) -> None:
        if token and not hmac.compare_digest(x_internal_token, token):
            raise HTTPException(401, "Invalid token")

    app = FastAPI(title="MiniCPM5-1B service", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": client.model, "ready": await client.model_ready()}

    @app.post("/v1/chat", dependencies=[Depends(authorize)])
    async def chat(body: ChatRequest) -> dict[str, Any]:
        if body.tools and body.json_schema is not None:
            raise HTTPException(422, "Send either tools or a schema, not both")
        started = time.perf_counter()
        try:
            reply = await client.chat([_ollama_message(m) for m in body.messages],
                                      tools=[_ollama_tool(t) for t in body.tools] if body.tools else None,
                                      schema=body.json_schema, thinking=body.thinking,
                                      temperature=body.temperature, max_tokens=body.max_tokens)
        except OllamaError as error:
            if error.status == 400:
                raise HTTPException(422, str(error)) from None
            raise HTTPException(504 if error.status == 504 else 502, str(error)) from None
        except Exception:
            _log.exception("MiniCPM chat failed")
            raise HTTPException(500, "MiniCPM could not answer") from None
        message = reply.get("message") or {}
        content = message.get("content") or ""
        calls = [{"id": call.get("id") or f"call_{index}", "name": call["function"]["name"],
                  "arguments": call["function"].get("arguments") or {}}
                 for index, call in enumerate(message.get("tool_calls") or [])]
        # `length`: the token budget ran out, often inside the thinking, so `content` may be empty.
        finish = "tool_calls" if calls else "length" if reply.get("done_reason") == "length" else "stop"
        result: dict[str, Any] = {
            "content": content, "thinking": message.get("thinking") or None, "tool_calls": calls,
            "finish_reason": finish, "model": client.model,
            "usage": {"input_tokens": reply.get("prompt_eval_count", 0), "output_tokens": reply.get("eval_count", 0)},
            "duration_ms": round((time.perf_counter() - started) * 1000)}
        if body.json_schema is not None:
            try:
                result["parsed"] = json.loads(content)
            except ValueError:
                # Cut off by max_tokens or malformed; the caller can still see `content`.
                raise HTTPException(502, {"message": "The model did not return valid JSON", "content": content}) from None
        return result

    return app


def main() -> None:
    import uvicorn

    app = create_app(token=os.environ.get("MINICPM_TOKEN") or None)
    uvicorn.run(app, host=os.environ.get("MINICPM_HOST", "127.0.0.1"), port=int(os.environ.get("MINICPM_PORT", "9130")))


if __name__ == "__main__":
    main()
