import json

import httpx
from fastapi.testclient import TestClient

from minicpm_service.app import create_app
from minicpm_service.ollama import OllamaClient

BODY = {"messages": [{"role": "user", "content": "hi"}]}


def client(handler, token=None):
    ollama = OllamaClient("http://ollama", "m", transport=httpx.MockTransport(handler))
    return TestClient(create_app(ollama, token))


def reply(content="hello", **extra):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content, **extra},
                                     "prompt_eval_count": 3, "eval_count": 5})


def test_chat_forwards_options_and_reports_usage():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return reply(thinking="hmm")

    body = client(handler).post("/v1/chat", json={**BODY, "thinking": True, "temperature": 0.3, "max_tokens": 50}).json()
    assert body["content"] == "hello" and body["thinking"] == "hmm"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 5}
    assert seen["think"] is True and seen["options"] == {"temperature": 0.3, "num_predict": 50} and "format" not in seen


def test_thinking_is_on_by_default_and_can_be_turned_off():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["think"])
        return reply()

    c = client(handler)
    c.post("/v1/chat", json=BODY)
    c.post("/v1/chat", json={**BODY, "thinking": False})
    assert seen == [True, False]


WEATHER = {"name": "get_weather", "description": "Weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}


def test_tools_are_sent_and_tool_calls_come_back_normalised():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}})

    body = client(handler).post("/v1/chat", json={**BODY, "tools": [WEATHER]}).json()
    assert seen["tools"] == [{"type": "function", "function": {**WEATHER}}]
    assert body["tool_calls"] == [{"id": "call_0", "name": "get_weather", "arguments": {"city": "Paris"}}]
    assert body["finish_reason"] == "tool_calls"


def test_tool_round_trip_messages_reach_ollama():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return reply("It is sunny.")

    messages = [{"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "tool_calls": [{"id": "call_0", "name": "get_weather", "arguments": {"city": "Paris"}}]},
                {"role": "tool", "tool_name": "get_weather", "content": "sunny"}]
    body = client(handler).post("/v1/chat", json={"messages": messages, "tools": [WEATHER]}).json()
    assert seen["messages"][1] == {"role": "assistant", "content": "",
                                   "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}
    assert seen["messages"][2] == {"role": "tool", "content": "sunny", "tool_name": "get_weather"}
    assert body["tool_calls"] == [] and body["finish_reason"] == "stop"


def test_tools_and_schema_together_are_refused():
    r = client(lambda request: reply()).post("/v1/chat", json={**BODY, "tools": [WEATHER], "schema": {"type": "object"}})
    assert r.status_code == 422


def test_running_out_of_tokens_is_reported():
    r = client(lambda request: httpx.Response(200, json={"message": {"content": "", "thinking": "hm"}, "done_reason": "length"}))
    assert r.post("/v1/chat", json=BODY).json()["finish_reason"] == "length"


def test_schema_constrains_and_parses_the_reply():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return reply('{"ok": true}')

    body = client(handler).post("/v1/chat", json={**BODY, "schema": schema}).json()
    assert body["parsed"] == {"ok": True} and seen["format"] == schema


def test_invalid_json_under_a_schema_is_502_with_the_content():
    r = client(lambda request: reply('{"ok": tr')).post("/v1/chat", json={**BODY, "schema": {"type": "object"}})
    assert r.status_code == 502 and r.json()["detail"]["content"] == '{"ok": tr'


def test_token_is_required_when_configured():
    c = client(lambda request: reply(), token="s3cret")
    assert c.post("/v1/chat", json=BODY).status_code == 401
    assert c.post("/v1/chat", json=BODY, headers={"X-Internal-Token": "s3cret"}).status_code == 200


def test_ollama_errors_map_to_http_statuses():
    def refuse(code):
        return client(lambda request: httpx.Response(code, json={"error": "nope"}))

    assert refuse(400).post("/v1/chat", json=BODY).status_code == 422
    assert refuse(404).post("/v1/chat", json=BODY).status_code == 502

    def down(request):
        raise httpx.ConnectError("refused")

    def slow(request):
        raise httpx.ReadTimeout("slow")

    assert client(down).post("/v1/chat", json=BODY).status_code == 502
    assert client(slow).post("/v1/chat", json=BODY).status_code == 504


def test_request_is_validated():
    c = client(lambda request: reply())
    assert c.post("/v1/chat", json={"messages": []}).status_code == 422
    assert c.post("/v1/chat", json={**BODY, "max_tokens": 0}).status_code == 422


def test_health_reports_model_readiness():
    ready = client(lambda request: httpx.Response(200, json={"models": [{"name": "m:latest"}]}))
    assert ready.get("/health").json() == {"status": "ok", "model": "m", "ready": True}
    assert client(lambda request: httpx.Response(200, json={"models": []})).get("/health").json()["ready"] is False
