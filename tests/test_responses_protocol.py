from __future__ import annotations

import json

import pytest

from muse2api.errors import UpstreamError
from muse2api.services.responses_state import ResponseSnapshot, ResponsesState

TOOL = {
    "type": "function",
    "name": "probe_tool",
    "description": "Return a value.",
    "parameters": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
}


def _enable_tools(client) -> None:
    client.app.state.services.settings.responses_tools_enabled = True


def _script_gateway(client, *, initial=None, continuation="TOOL_RESULT_OK"):
    prompts = []

    async def scripted(req):
        prompts.append(req.prompt)
        result = continuation if "[Tool]\n" in req.prompt else (
            initial if "tool-selection protocol" in req.prompt else continuation)
        for part in (result or "").splitlines(keepends=True):
            if part:
                yield part

    client.app.state.services.gateway.chat_stream = scripted
    return prompts


def _first_call(client, auth, *, stream=False, choice="required"):
    return client.post("/v1/responses", headers=auth, json={
        "model": "gpt-4o",
        "input": "Call probe_tool with value OK.",
        "tools": [TOOL],
        "tool_choice": choice,
        "stream": stream,
    })


@pytest.mark.parametrize("mode", ["output-only", "full-history", "previous-response-id"])
def test_structured_tool_continuations(client, auth, mode):
    _enable_tools(client)
    prompts = _script_gateway(
        client,
        initial='{"type":"function_call","name":"probe_tool","arguments":{"value":"OK"}}',
    )
    first = _first_call(client, auth)
    assert first.status_code == 200
    first_body = first.json()
    call = first_body["output"][0]
    assert call["type"] == "function_call"
    assert call["name"] == "probe_tool"
    assert json.loads(call["arguments"]) == {"value": "OK"}
    assert first_body["parallel_tool_calls"] is False

    output = {"type": "function_call_output", "call_id": call["call_id"],
              "output": '{"value":"OK"}'}
    payload = {"model": "gpt-4o"}
    if mode == "output-only":
        payload["input"] = [output]
    elif mode == "full-history":
        payload["input"] = [
            {"type": "message", "role": "user", "content": "Call probe_tool with value OK."},
            call,
            output,
        ]
    else:
        payload["previous_response_id"] = first_body["id"]
        payload["input"] = [output]
    continued = client.post("/v1/responses", headers=auth, json=payload)
    assert continued.status_code == 200, continued.text
    assert continued.json()["output"][0]["content"][0]["text"] == "TOOL_RESULT_OK"
    assert "[Function call probe_tool:" in prompts[-1]
    assert "[Tool]\n{\"value\":\"OK\"}" in prompts[-1]


@pytest.mark.parametrize("choice", ["auto", "required", {"type": "function", "name": "probe_tool"}])
def test_tool_choice_returns_structured_call(client, auth, choice):
    _enable_tools(client)
    _script_gateway(
        client,
        initial='{"type":"function_call","name":"probe_tool","arguments":{"value":"OK"}}',
    )
    response = _first_call(client, auth, choice=choice)
    assert response.status_code == 200
    assert response.json()["output"][0]["type"] == "function_call"


def test_tool_choice_none_never_parses_tool_envelope(client, auth):
    _enable_tools(client)
    envelope = '{"type":"function_call","name":"probe_tool","arguments":{"value":"OK"}}'
    _script_gateway(
        client,
        initial=envelope,
        continuation=envelope,
    )
    response = _first_call(client, auth, choice="none")
    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["content"][0]["text"].startswith('{"type":"function_call"')
    assert body["parallel_tool_calls"] is False


def test_required_tool_rejects_malformed_upstream_envelope(client, auth):
    _enable_tools(client)
    _script_gateway(client, initial="I will call the tool now.")
    response = _first_call(client, auth)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_tool_call"


def test_tools_are_off_by_default_and_state_ids_are_not_guessed(client, auth):
    disabled = client.post("/v1/responses", headers=auth, json={
        "input": "hello", "tools": [TOOL], "tool_choice": "required",
    })
    assert disabled.status_code == 501
    unknown = client.post("/v1/responses", headers=auth, json={
        "input": "continue", "previous_response_id": "resp_unknown",
    })
    assert unknown.status_code == 404


def test_streaming_tool_call_has_responses_lifecycle(client, auth):
    _enable_tools(client)
    _script_gateway(
        client,
        initial='{"type":"function_call","name":"probe_tool","arguments":{"value":"OK"}}',
    )
    response = _first_call(client, auth, stream=True)
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: ")]
    event_types = [event["type"] for event in events]
    assert event_types == [
        "response.created", "response.in_progress", "response.output_item.added",
        "response.function_call_arguments.delta", "response.function_call_arguments.done",
        "response.output_item.done", "response.completed",
    ]
    assert events[-1]["response"]["output"][0]["type"] == "function_call"


def test_tool_output_call_ids_are_one_use(client, auth):
    _enable_tools(client)
    _script_gateway(
        client,
        initial='{"type":"function_call","name":"probe_tool","arguments":{"value":"OK"}}',
    )
    first = _first_call(client, auth)
    call = first.json()["output"][0]
    output = {"type": "function_call_output", "call_id": call["call_id"],
              "output": '{"value":"OK"}'}
    continued = client.post("/v1/responses", headers=auth, json={
        "model": "gpt-4o", "input": [output],
    })
    assert continued.status_code == 200
    replay = client.post("/v1/responses", headers=auth, json={
        "model": "gpt-4o", "input": [output],
    })
    assert replay.status_code == 404
    assert replay.json()["error"]["code"] == "response_not_found"


def test_stream_prefetch_error_releases_call_reservation_and_closes_upstream(client, auth):
    _enable_tools(client)
    _script_gateway(
        client,
        initial='{"type":"function_call","name":"probe_tool","arguments":{"value":"OK"}}',
    )
    first = _first_call(client, auth)
    call = first.json()["output"][0]
    output = {"type": "function_call_output", "call_id": call["call_id"],
              "output": '{"value":"OK"}'}
    closed = False

    async def fail_first_delta(req):
        nonlocal closed
        try:
            raise UpstreamError("temporary failure")
            yield "unreachable"
        finally:
            closed = True

    client.app.state.services.gateway.chat_stream = fail_first_delta
    failed = client.post("/v1/responses", headers=auth, json={
        "input": [output], "stream": True,
    })
    assert failed.status_code == 502
    assert closed

    _script_gateway(client, continuation="TOOL_RESULT_OK")
    retried = client.post("/v1/responses", headers=auth, json={"input": [output]})
    assert retried.status_code == 200
    assert retried.json()["output"][0]["content"][0]["text"] == "TOOL_RESULT_OK"


def test_heartbeat_stream_failure_uses_response_failed_terminal_event(client, auth):
    client.app.state.services.settings.stream_heartbeat_interval = 0.001

    async def fail_after_delta(req):
        yield "partial"
        raise UpstreamError("temporary failure")

    client.app.state.services.gateway.chat_stream = fail_after_delta
    response = client.post("/v1/responses", headers=auth, json={
        "input": "stream with late failure", "stream": True,
    })
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: ")]
    types = [event["type"] for event in events]
    assert types[0] == "response.created"
    assert types[-1] == "response.failed"
    assert events[-1]["response"]["status"] == "failed"
    assert events[-1]["response"]["error"]["message"] == "temporary failure"


def test_responses_state_ttl_and_owner_isolation():
    state = ResponsesState(ttl_seconds=30, max_entries=16)
    snapshot = ResponseSnapshot("key-a", "resp-a", "model", 100, 0,
                                [{"role": "user", "content": "hello"}],
                                pending_calls={"call-a": {"name": "probe_tool"}})
    state.put(snapshot, now=100)
    assert state.get("key-a", "resp-a", now=110) is snapshot
    assert state.get("key-b", "resp-a", now=110) is None
    assert state.get_for_call("key-a", "call-a", now=110) is snapshot
    assert state.get("key-a", "resp-a", now=131) is None
    assert state.get_for_call("key-a", "call-a", now=131) is None


def test_responses_state_reservations_are_atomic_and_released():
    state = ResponsesState(ttl_seconds=30, max_entries=16)
    snapshot = ResponseSnapshot("key-a", "resp-a", "model", 100, 0, [],
                                pending_calls={"call-a": {"name": "probe_tool"}})
    state.put(snapshot, now=100)
    assert state.reserve_calls("key-a", "resp-a", {"call-a"}, now=101)
    assert not state.reserve_calls("key-a", "resp-a", {"call-a"}, now=101)
    state.release_calls("key-a", "resp-a", {"call-a"})
    assert state.reserve_calls("key-a", "resp-a", {"call-a"}, now=101)
    state.consume_calls("key-a", "resp-a", {"call-a"}, now=101)
    assert state.get_for_call("key-a", "call-a", now=101) is None
