"""OpenAI Responses API (``/v1/responses``).

Maps ``input`` / ``instructions`` onto ``flatten_messages`` and reuses
``Gateway.chat_stream``. Streaming emits the standard Responses events
(``response.created`` / ``response.output_text.delta`` / ``response.completed``).

muse.ai has no function calling, so tool items (``function_call``,
``function_call_output``) are rejected with a clear error instead of being
silently dropped (see C3).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from ...core.media import load_image_ref
from ...core.models import resolve_model
from ...core.prompt import flatten_messages, message_turns
from ...drivers.base import ChatRequest, InputImage
from ...errors import FeatureNotImplemented, InvalidRequest, Muse2APIError, NotFound, UpstreamError
from ...services.container import Services
from ...services.responses_state import ResponseSnapshot, ResponsesState
from ..deps import get_services, require_api_key

log = logging.getLogger(__name__)
router = APIRouter(tags=["responses"], dependencies=[Depends(require_api_key)])


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _sse(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n".encode()


class ResponseRequest(BaseModel):
    model: str | None = None
    input: Any = None
    instructions: str | None = None
    stream: bool = False
    user: str | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None

    model_config = ConfigDict(extra="allow")


# Item types the official SDK round-trips but muse.ai cannot fulfil.
_TOOL_ITEMS = {"function_call", "function_call_output", "reasoning", "computer_call",
               "file_search_call", "web_search_call"}


def _has_tool_items(body: ResponseRequest) -> bool:
    return isinstance(body.input, list) and any(
        isinstance(item, dict) and item.get("type") in {"function_call", "function_call_output"}
        for item in body.input
    )


def _tool_choice_name(choice: Any) -> str:
    if choice is None:
        return "none"
    if isinstance(choice, str):
        return choice if choice in {"none", "auto", "required"} else "invalid"
    if isinstance(choice, dict) and choice.get("type") == "function":
        return "named"
    return "invalid"


def _effective_tool_choice(body: ResponseRequest) -> Any:
    if body.tool_choice is not None:
        return body.tool_choice
    return "auto" if body.tools else "none"


def _tool_name(choice: Any) -> str | None:
    if isinstance(choice, dict) and choice.get("type") == "function":
        function = choice.get("function")
        return function.get("name") if isinstance(function, dict) else choice.get("name")
    return None


def _validate_tools(tools: list[dict[str, Any]] | None, choice: Any) -> None:
    if tools is None:
        return
    names: set[str] = set()
    for tool in tools:
        if (not isinstance(tool, dict) or tool.get("type") != "function"
                or not isinstance(tool.get("name"), str) or not tool["name"]
                or not isinstance(tool.get("parameters", {}), dict)):
            raise InvalidRequest("only function tools with an object parameters schema are supported")
        if tool["name"] in names:
            raise InvalidRequest("function tool names must be unique")
        names.add(tool["name"])
    selected = _tool_name(choice)
    if selected and selected not in names:
        raise InvalidRequest("tool_choice names a function that is not in tools")


def _tool_instruction(tools: list[dict[str, Any]], choice: Any) -> str:
    schema = [{"name": t["name"], "description": t.get("description", ""),
               "parameters": t.get("parameters", {})} for t in tools]
    selected = _tool_name(choice)
    return (
        "This is a tool-selection protocol. Do not claim that any tool has run. "
        "When calling a tool, output exactly one JSON object with this shape: "
        '{"type":"function_call","name":"TOOL_NAME","arguments":{}}. '
        "Use only a listed function and valid JSON arguments. Do not wrap the object "
        "in markdown. Available functions: " + json.dumps(schema, ensure_ascii=False)
        + (f". You must call {selected}." if selected else "")
        + (". You must call one function." if choice == "required" else "")
        + (". If no tool is needed, answer normally." if choice == "auto" else "")
    )


async def _text_chunks(text: str) -> AsyncIterator[str]:
    for start in range(0, len(text), 48):
        yield text[start:start + 48]


def _validate_tool_request(body: ResponseRequest, enabled: bool) -> None:
    has_tools = bool(body.tools)
    effective = _effective_tool_choice(body)
    choice = _tool_choice_name(effective)
    _validate_tools(body.tools, effective)
    if choice == "invalid":
        raise InvalidRequest("tool_choice must be 'none', 'auto', 'required', or a function choice")
    if _has_tool_items(body) and not enabled:
        raise InvalidRequest("function calling items require responses_tools_enabled")
    if has_tools and choice in {"auto", "required", "named"} and not enabled:
        raise FeatureNotImplemented(
            "structured tool calling is disabled; enable responses_tools_enabled only after live validation",
            code="tool_calling_disabled",
        )
    if choice == "none" and _has_tool_items(body) and not enabled:
        raise InvalidRequest("function calling items require responses_tools_enabled")
    if choice == "required" and not has_tools:
        raise InvalidRequest("tool_choice='required' needs at least one tool")


def _tool_call_from_text(text: str, tools: list[dict[str, Any]], choice: Any) -> dict[str, Any] | None:
    """Parse only the explicit JSON envelope used by opt-in prompt emulation."""
    if not text.strip():
        return None
    raw = text.strip()
    candidates = [raw]
    if raw.startswith("```"):
        candidates.append(raw.strip("`").removeprefix("json").strip())
    decoder = json.JSONDecoder()
    for start, char in enumerate(raw):
        if char == "{":
            try:
                candidates.append(raw[start: start + len(raw)].strip())
                candidates.append(decoder.raw_decode(raw[start:])[0])
            except (TypeError, ValueError):
                continue
    value = next((item for item in candidates if isinstance(item, dict)
                  and item.get("type") == "function_call"), None)
    if value is None:
        return None
    name = value.get("name")
    selected = _tool_name(choice)
    if selected and name != selected:
        return None
    if not isinstance(name, str) or name not in {
        t.get("name") for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str)
    }:
        return None
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return None
    if not isinstance(arguments, dict):
        return None
    return {"type": "function_call", "id": "fc_" + uuid.uuid4().hex[:24],
            "call_id": "call_" + uuid.uuid4().hex[:24], "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))}


def _tool_history(body: ResponseRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ordinary messages and Responses tool items from the current input."""
    if not isinstance(body.input, list):
        return _to_messages(body), []
    tool_items = [item for item in body.input if isinstance(item, dict)
                  and item.get("type") in {"function_call", "function_call_output"}]
    normal_input = [item for item in body.input if not isinstance(item, dict)
                    or item.get("type") not in _TOOL_ITEMS]
    normal_body = body.model_copy(update={"input": normal_input})
    return _to_messages(normal_body), tool_items


def _same_message(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("role") == right.get("role") and left.get("content") == right.get("content")


def _merge_continuation(body: ResponseRequest, previous: ResponseSnapshot | None,
                        owner_id: str, state_store: ResponsesState,
                        tools_enabled: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                      list[dict[str, Any]]]:
    unsupported = [item.get("type") for item in (body.input if isinstance(body.input, list) else [])
                   if isinstance(item, dict) and item.get("type") in _TOOL_ITEMS
                   and item.get("type") not in {"function_call", "function_call_output"}]
    if unsupported:
        raise InvalidRequest(f"input item type '{unsupported[0]}' is not supported")
    if not tools_enabled and _has_tool_items(body):
        raise InvalidRequest("function calling items require responses_tools_enabled",
                             code="tool_calling_disabled")
    normal, tool_items = _tool_history(body)
    if previous is None:
        if tool_items:
            raise NotFound("tool item does not refer to a known response", code="response_not_found")
        return normal, body.tools or [], []

    messages = list(previous.messages)
    calls = dict(previous.pending_calls)
    replayed: dict[str, dict[str, Any]] = {}
    outputs: dict[str, str] = {}
    for item in tool_items:
        call_id = item.get("call_id")
        if item.get("type") == "function_call":
            call_id = item.get("call_id") or item.get("id")
            if not isinstance(call_id, str) or call_id not in calls:
                raise InvalidRequest("function_call item does not match the previous response")
            saved = calls[call_id]
            if (item.get("name") != saved.get("name")
                    or str(item.get("arguments", "{}")) != saved.get("arguments")):
                raise InvalidRequest("function_call item differs from the previous response")
            replayed[call_id] = item
        else:
            if not isinstance(call_id, str) or call_id not in calls:
                raise InvalidRequest("function_call_output has no matching pending call")
            if call_id in outputs:
                raise InvalidRequest("duplicate function_call_output")
            output = item.get("output", "")
            if not isinstance(output, str):
                raise InvalidRequest("function_call_output.output must be a string")
            outputs[call_id] = output

    if calls and set(outputs) != set(calls):
        raise InvalidRequest("every pending function call needs exactly one output")
    if not calls and outputs:
        raise InvalidRequest("previous response has no pending function call")

    if body.previous_response_id:
        messages.extend(normal)
    elif normal:
        # Full-history tool replay is accepted only when its ordinary message
        # prefix agrees with server-retained history. This prevents clients
        # from smuggling a different conversation under an old call ID.
        expected = [m for m in previous.messages if m.get("role") != "system"]
        observed = [m for m in normal if m.get("role") != "system"]
        if len(observed) > len(expected) or any(
            not _same_message(a, b) for a, b in zip(observed, expected, strict=False)
        ):
            raise InvalidRequest("full-history input does not match the previous response")

    for call_id, output in outputs.items():
        call = calls[call_id]
        messages.append({"role": "assistant", "content":
                         f"[Function call {call['name']}: {call['arguments']}]"})
        messages.append({"role": "tool", "content": output})
    tools = body.tools if body.tools is not None else previous.tools
    return messages, tools or [], tool_items


def _to_messages(body: ResponseRequest, *, allow_tools: bool = False) -> list[dict]:
    """Convert Responses ``input`` / ``instructions`` to chat-style messages."""
    messages: list[dict] = []
    if body.instructions:
        messages.append({"role": "system", "content": body.instructions})
    inp = body.input
    if inp is None or (isinstance(inp, str) and not inp.strip()):
        pass
    elif isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                raise InvalidRequest("input items must be objects")
            itype = item.get("type")
            if itype in _TOOL_ITEMS:
                if allow_tools and itype == "function_call":
                    name = item.get("name")
                    arguments = item.get("arguments", "{}")
                    messages.append({"role": "assistant", "content":
                                     f"[Function call {name}: {arguments}]",
                                     "_function_call": item})
                    continue
                if allow_tools and itype == "function_call_output":
                    call_id = item.get("call_id")
                    output = item.get("output", "")
                    messages.append({"role": "tool", "content": str(output),
                                     "_function_call_output": {"call_id": call_id}})
                    continue
                raise InvalidRequest(
                    f"input item type '{itype}' is not supported: "
                    "muse.ai has no function calling")
            if "role" in item or itype in ("message", None):
                messages.append({"role": item.get("role", "user"),
                                 "content": item.get("content")})
            else:
                raise InvalidRequest(f"unsupported input item type '{itype}'")
    else:
        raise InvalidRequest("input must be a string or a list of items")
    return messages


def _owner_id(request: Request) -> str:
    return str(getattr(request.state, "key_id", "legacy"))


def _tool_continuation_state(body: ResponseRequest, owner_id: str,
                             state_store: ResponsesState) -> ResponseSnapshot | None:
    if body.previous_response_id:
        state = state_store.get(owner_id, body.previous_response_id)
        if state is None:
            raise NotFound("previous_response_id is unknown or expired", code="response_not_found")
        return state
    if isinstance(body.input, list):
        for item in body.input:
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                call_id = item.get("call_id")
                if isinstance(call_id, str):
                    state = state_store.get_for_call(owner_id, call_id)
                    if state is None:
                        raise NotFound("function_call_output refers to an unknown or expired call",
                                        code="response_not_found")
                    return state
    return None


def _message_item(msg_id: str, text: str) -> dict:
    return {"id": msg_id, "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "annotations": [], "text": text}]}


def _function_call_item(call: dict[str, Any], *, status: str = "completed") -> dict[str, Any]:
    return {**call, "status": status}


def _usage(input_text: str, output_text: str) -> dict:
    i, o = _approx_tokens(input_text), _approx_tokens(output_text)
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0}}


def _response_object(resp_id: str, msg_id: str, created: int, model: str, text: str,
                     input_text: str, *, status: str = "completed",
                     output: list[dict[str, Any]] | None = None,
                     tools: list[dict[str, Any]] | None = None,
                     tool_choice: Any = None) -> dict:
    advertised_tools = tools or []
    choice = tool_choice if tool_choice is not None else ("auto" if advertised_tools else "none")
    return {"id": resp_id, "object": "response", "created_at": created, "model": model,
            "status": status,
            "output": output if output is not None else (
                [_message_item(msg_id, text)] if status == "completed" else []),
            "usage": _usage(input_text, text),
            # Prompt-emulated Muse calls are serial and return at most one call.
            "parallel_tool_calls": False,
            "tool_choice": choice, "tools": advertised_tools}


def _failed_event(resp_id: str, msg_id: str, created: int, model: str,
                  input_text: str, sequence: int, error: dict[str, Any]) -> bytes:
    response = _response_object(resp_id, msg_id, created, model, "", input_text,
                                status="failed", output=[])
    response["error"] = error
    return _sse({"type": "response.failed", "sequence_number": sequence,
                 "response": response})


async def _tool_events(resp_id: str, created: int, model: str, call: dict[str, Any],
                       tools: list[dict[str, Any]], tool_choice: Any) -> AsyncIterator[bytes]:
    seq = 0
    yield _sse({"type": "response.created", "sequence_number": seq,
                "response": _response_object(resp_id, "", created, model, "", "",
                                             status="in_progress", output=[], tools=tools,
                                             tool_choice=tool_choice)})
    seq += 1
    yield _sse({"type": "response.in_progress", "sequence_number": seq,
                "response": _response_object(resp_id, "", created, model, "", "",
                                             status="in_progress", output=[], tools=tools,
                                             tool_choice=tool_choice)})
    seq += 1
    partial = {**call, "status": "in_progress", "arguments": ""}
    yield _sse({"type": "response.output_item.added", "sequence_number": seq,
                "output_index": 0, "item": partial})
    seq += 1
    args = call["arguments"]
    for start in range(0, len(args), 24):
        delta = args[start:start + 24]
        yield _sse({"type": "response.function_call_arguments.delta", "sequence_number": seq,
                    "item_id": call["id"], "output_index": 0, "delta": delta})
        seq += 1
    yield _sse({"type": "response.function_call_arguments.done", "sequence_number": seq,
                "item_id": call["id"], "output_index": 0, "arguments": args})
    seq += 1
    yield _sse({"type": "response.output_item.done", "sequence_number": seq,
                "output_index": 0, "item": _function_call_item(call)})
    seq += 1
    final = _function_call_item(call)
    yield _sse({"type": "response.completed", "sequence_number": seq,
                "response": _response_object(resp_id, "", created, model, "", "",
                                             output=[final], tools=tools,
                                             tool_choice=tool_choice)})


@router.post("/v1/responses")
async def create_response(body: ResponseRequest, request: Request,
                          svc: Services = Depends(get_services)):
    request.state.model = body.model  # logged even if it does not resolve
    spec = resolve_model(body.model, "chat")
    request.state.model = body.model or spec.id
    tools_enabled = svc.settings.responses_tools_enabled
    owner_id = _owner_id(request)
    previous = _tool_continuation_state(body, owner_id, svc.responses)
    messages, tools, tool_items = _merge_continuation(
        body, previous, owner_id, svc.responses, tools_enabled)
    effective_choice = _effective_tool_choice(body)
    has_tool_output = any(item.get("type") == "function_call_output" for item in tool_items)
    consumed_call_ids = {
        item.get("call_id") for item in tool_items
        if item.get("type") == "function_call_output"
        and isinstance(item.get("call_id"), str)
    }
    if previous is not None and body.tools is None and body.tool_choice is None:
        effective_choice = "none" if has_tool_output or not tools else "auto"
    if has_tool_output and effective_choice == "none":
        messages.append({"role": "system", "content":
                         "The external function call has completed. Use its tool result above "
                         "to answer the original user. Do not claim that the tool was executed "
                         "by this service."})
    protocol_body = body.model_copy(update={"tools": tools, "tool_choice": effective_choice})
    _validate_tool_request(protocol_body, tools_enabled)
    tool_choice = _tool_choice_name(effective_choice)
    conversation_messages = list(messages)
    if tools_enabled and tools and tool_choice in {"auto", "required", "named"}:
        messages.append({"role": "system", "content": _tool_instruction(tools, effective_choice)})
    flat = flatten_messages(messages)
    if not flat.text and not flat.images:
        raise InvalidRequest("input contains no text or images")

    images = [InputImage(*(await load_image_ref(ref))) for ref in flat.images]
    reserved_call_ids: set[str] = set()
    if previous is not None and consumed_call_ids:
        if not svc.responses.reserve_calls(owner_id, previous.response_id, consumed_call_ids):
            raise InvalidRequest("function_call_output has already been submitted",
                                 code="duplicate_function_call_output")
        reserved_call_ids = consumed_call_ids
    cancel = asyncio.Event()
    req = ChatRequest(
        prompt=flat.text,
        model=spec.id,
        images=images,
        timeout=svc.settings.chat_timeout,
        first_token_timeout=svc.settings.first_token_timeout,
        conversation_hint=body.user,
        turns=message_turns(messages),
        cancel=cancel,
    )
    try:
        stream = svc.gateway.chat_stream(req)
    except BaseException:
        if reserved_call_ids:
            svc.responses.release_calls(owner_id, previous.response_id, reserved_call_ids)
        raise
    resp_id = "resp_" + uuid.uuid4().hex[:24]
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    created = int(time.time())
    model_name = body.model or spec.id

    choice = _tool_choice_name(effective_choice)
    tool_call: dict[str, Any] | None = None
    generated_text: str | None = None
    if tools_enabled and tools and choice in {"auto", "required", "named"}:
        # Prompt emulation is intentionally strict. The upstream must return
        # one JSON function_call envelope; arbitrary prose is never advertised
        # as a structured call.
        try:
            generated_text = "".join([delta async for delta in stream])
        except BaseException:
            if reserved_call_ids:
                svc.responses.release_calls(owner_id, previous.response_id, reserved_call_ids)
            raise
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
        tool_call = _tool_call_from_text(generated_text, tools, effective_choice)
        if tool_call is None and choice == "required":
            if reserved_call_ids:
                svc.responses.release_calls(owner_id, previous.response_id, reserved_call_ids)
            raise UpstreamError("upstream did not produce a valid structured tool call",
                                code="upstream_invalid_tool_call")
        if tool_call is not None:
            snapshot = ResponseSnapshot(
                owner_id=owner_id, response_id=resp_id, model=model_name,
                created_at=created, expires_at=0,
                messages=conversation_messages,
                tools=tools, pending_calls={tool_call["call_id"]: tool_call},
            )
            svc.responses.put(snapshot)
            if previous is not None and consumed_call_ids:
                svc.responses.consume_calls(owner_id, previous.response_id, consumed_call_ids)
            call_output = [_function_call_item(tool_call)]
            if not body.stream:
                return _response_object(resp_id, msg_id, created, model_name, "", flat.text,
                                        output=call_output, tools=tools,
                                        tool_choice=effective_choice)
            return StreamingResponse(_tool_events(resp_id, created, model_name, tool_call,
                                                  tools, effective_choice),
                                     media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache",
                                              "X-Accel-Buffering": "no"})

    # Save text responses too, so a later previous_response_id continuation
    # has a well-defined, bounded history even when tools are disabled.
    async def save_text_state(text: str) -> None:
        svc.responses.put(ResponseSnapshot(
            owner_id=owner_id, response_id=resp_id, model=model_name,
            created_at=created, expires_at=0,
            messages=[*conversation_messages, {"role": "assistant", "content": text}],
            tools=tools, pending_calls={},
        ))

    if generated_text is not None:
        stream = _text_chunks(generated_text)

    if not body.stream:
        try:
            text = generated_text if generated_text is not None else "".join([d async for d in stream])
            await save_text_state(text)
            if previous is not None and consumed_call_ids:
                svc.responses.consume_calls(owner_id, previous.response_id, consumed_call_ids)
            return _response_object(resp_id, msg_id, created, model_name, text, flat.text,
                                    tools=tools, tool_choice=effective_choice)
        except BaseException:
            if reserved_call_ids:
                svc.responses.release_calls(owner_id, previous.response_id, reserved_call_ids)
            raise
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

    async def events(first: str | None) -> AsyncIterator[bytes]:
        # ``first`` is the pre-fetched first delta when heartbeats are disabled;
        # when enabled it is None and the first token is awaited here, after the
        # 200 (headers) has already been committed to the client.
        seq = 0
        try:
            yield _sse({"type": "response.created", "sequence_number": seq,
                        "response": _response_object(resp_id, msg_id, created, model_name,
                                                     "", flat.text, status="in_progress",
                                                     tools=tools, tool_choice=effective_choice)})
            seq += 1
            # The official SDK builds its accumulated snapshot from item/part
            # lifecycle events, so emit them around the text deltas.
            yield _sse({"type": "response.in_progress", "sequence_number": seq,
                        "response": _response_object(resp_id, msg_id, created, model_name,
                                                     "", flat.text, status="in_progress",
                                                     tools=tools, tool_choice=effective_choice)})
            seq += 1
            yield _sse({"type": "response.output_item.added", "sequence_number": seq,
                        "output_index": 0,
                        "item": {"id": msg_id, "type": "message", "status": "in_progress",
                                 "role": "assistant", "content": []}})
            seq += 1
            yield _sse({"type": "response.content_part.added", "sequence_number": seq,
                        "item_id": msg_id, "output_index": 0, "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []}})
            seq += 1
            if first is None and svc.settings.stream_heartbeat_interval > 0:
                # Emit lifecycle setup before waiting so failures still have a
                # complete response.failed terminal event sequence.
                pending = asyncio.ensure_future(anext(stream))
                try:
                    while True:
                        done, _ = await asyncio.wait(
                            {pending},
                            timeout=svc.settings.stream_heartbeat_interval)
                        if done:
                            try:
                                first = pending.result()
                            except StopAsyncIteration:
                                first = ""
                            break
                        yield b": keepalive\n\n"
                finally:
                    if not pending.done():
                        pending.cancel()
            produced = first or ""

            def delta_event(delta: str) -> dict:
                return {"type": "response.output_text.delta", "sequence_number": seq,
                        "item_id": msg_id, "output_index": 0, "content_index": 0,
                        "delta": delta, "logprobs": []}

            if produced:
                yield _sse(delta_event(produced))
                seq += 1
            async for delta in stream:
                produced += delta
                yield _sse(delta_event(delta))
                seq += 1
            yield _sse({"type": "response.output_text.done", "sequence_number": seq,
                        "item_id": msg_id, "output_index": 0, "content_index": 0,
                        "text": produced, "logprobs": []})
            seq += 1
            yield _sse({"type": "response.content_part.done", "sequence_number": seq,
                        "item_id": msg_id, "output_index": 0, "content_index": 0,
                        "part": {"type": "output_text", "text": produced, "annotations": []}})
            seq += 1
            # Persist and consume before the terminal event so an immediate
            # previous_response_id continuation observes a complete snapshot.
            await save_text_state(produced)
            if previous is not None and consumed_call_ids:
                svc.responses.consume_calls(owner_id, previous.response_id, consumed_call_ids)
            yield _sse({"type": "response.output_item.done", "sequence_number": seq,
                        "output_index": 0, "item": _message_item(msg_id, produced)})
            seq += 1
            yield _sse({"type": "response.completed", "sequence_number": seq,
                        "response": _response_object(resp_id, msg_id, created, model_name,
                                                     produced, flat.text, tools=tools,
                                                     tool_choice=effective_choice)})
        except Muse2APIError as exc:
            if reserved_call_ids:
                svc.responses.release_calls(owner_id, previous.response_id, reserved_call_ids)
            yield _failed_event(resp_id, msg_id, created, model_name, flat.text, seq, exc.to_body()["error"])
        except Exception as exc:  # noqa: BLE001
            if reserved_call_ids:
                svc.responses.release_calls(owner_id, previous.response_id, reserved_call_ids)
            yield _failed_event(resp_id, msg_id, created, model_name, flat.text, seq,
                               {"message": "internal error", "type": "server_error",
                                "code": "internal_error"})
            log.error("responses stream crashed (%s)", type(exc).__name__)
        finally:
            if reserved_call_ids:
                svc.responses.release_calls(owner_id, previous.response_id, reserved_call_ids)
            cancel.set()
            with contextlib.suppress(Exception):
                await stream.aclose()

    # With stream_heartbeat_interval set (C4), the 200 goes out immediately;
    # upstream errors then surface as SSE error events instead of HTTP codes.
    if svc.settings.stream_heartbeat_interval > 0:
        return StreamingResponse(events(None), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # Default: pull the first delta before committing to a 200 so that
    # account/upstream failures surface as proper HTTP errors instead of a
    # broken event stream.
    try:
        first = await anext(stream)
    except StopAsyncIteration:
        first = ""
    except BaseException:
        if reserved_call_ids:
            svc.responses.release_calls(owner_id, previous.response_id, reserved_call_ids)
        with contextlib.suppress(Exception):
            await stream.aclose()
        raise

    return StreamingResponse(events(first), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
