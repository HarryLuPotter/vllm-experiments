# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only protocol tests; run with --confcutdir=tests/tool_time."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "round_trip_under_test", ROOT / "vllm/entrypoints/openai/tool_time.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
PREDICTION = module.PREDICTION
OBSERVATION = module.OBSERVATION


def response(name="bash", prediction=3.0, call_id="call_1"):
    return {
        "id": "request_1",
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            PREDICTION: prediction,
                            "function": {"name": name, "arguments": "{}"},
                        }
                    ]
                }
            }
        ],
    }


def history(wire):
    return [
        {"role": "assistant", **wire["choices"][0]["message"]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]


@pytest.fixture
def tracker(tmp_path):
    now = [100.0]
    tracker = module.ToolTimeTracker(tmp_path, clock=lambda: now[0])
    yield tracker, now
    tracker.close()


def test_standard_messages_round_trip_and_idempotent_history(tracker):
    t, now = tracker
    wire = response()
    calls = t.filter_response(wire)
    assert PREDICTION not in json.dumps(wire)
    t.sent(calls)
    messages = history(wire)
    now[0] = 150  # Processing/queue delay must not affect ingress time.
    enriched = t.prepare(messages, arrived_at=104)
    assert enriched[1][OBSERVATION] == 4
    assert enriched[0]["tool_calls"][0][PREDICTION] == 3
    assert OBSERVATION not in messages[1]
    assert t.prepare(messages, arrived_at=130) == enriched
    assert t.prepare(messages, arrived_at=None) == enriched
    assert t.sequence == 2


def test_render_does_not_complete_call_and_ignores_client_observations(tracker):
    t, _ = tracker
    wire = response()
    t.sent(t.filter_response(wire))
    messages = history(wire)
    messages[1]["tool_exec_time"] = 999
    messages[1][OBSERVATION] = 888
    restored = t.prepare(messages, arrived_at=None)
    assert OBSERVATION not in restored[1]
    assert "tool_exec_time" not in restored[1]
    assert t.records["call_1"].observed is None
    assert t.sequence == 1


@pytest.mark.parametrize("change", ["name", "arguments", "id"])
def test_mismatched_history_does_not_complete(tracker, change):
    t, _ = tracker
    wire = response()
    t.sent(t.filter_response(wire))
    messages = history(wire)
    call = messages[0]["tool_calls"][0]
    if change == "id":
        call["id"] = "unknown"
    else:
        call["function"][change] = "other" if change == "name" else '{"x":1}'
    assert OBSERVATION not in t.prepare(messages, arrived_at=104)[1]
    assert t.records["call_1"].observed is None


@pytest.mark.parametrize("prediction", [None, -1, float("nan"), float("inf"), True])
def test_missing_prediction_still_measures_return(tracker, prediction):
    t, _ = tracker
    wire = response(prediction=prediction)
    t.sent(t.filter_response(wire))
    enriched = t.prepare(history(wire), arrived_at=104)
    assert PREDICTION not in enriched[0]["tool_calls"][0]
    assert enriched[1][OBSERVATION] == 4


def test_all_tool_names_multiple_and_send_failure(tracker):
    t, _ = tracker
    wire = response(name="finish")
    wire["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        json.dumps({PREDICTION: 2})
    )
    t.sent(t.filter_response(wire))
    assert t.records["call_1"].prediction == 3
    assert (
        json.loads(
            wire["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        == {}
    )
    t.prepare(history(wire), arrived_at=104)
    assert t.records["call_1"].observed == 4
    t.records.clear()
    calls = t.filter_response(response())
    t.sent(calls + t.filter_response(response(call_id="call_2")))
    assert not t.records
    t.sent(calls)
    t.send_failed(calls)
    assert not t.records


def test_retention_capacity_and_shutdown_log(tmp_path):
    now = [0]
    t = module.ToolTimeTracker(tmp_path, clock=lambda: now[0])
    t.capacity = 1
    t.sent(t.filter_response(response()))
    t.sent(t.filter_response(response(call_id="call_2")))
    assert list(t.records) == ["call_2"]
    now[0] = 86401
    t.prune()
    assert not t.records
    t.sent(t.filter_response(response(call_id="call_3")))
    t.close()
    events = [json.loads(line) for line in t.path.read_text().splitlines()]
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert events[-1]["event"] == "unobserved_at_shutdown"
    assert events[-1][OBSERVATION] is None
    assert "arguments" not in events[0]


def test_asgi_boundary_and_failure(tracker):
    t, now = tracker

    async def run(fail):
        scope = {"type": "http"}

        async def app(scope, receive, send):
            assert "tool_time_arrived_at" in scope["state"]
            scope["state"]["tool_time_tracker"] = t
            scope["state"]["tool_time_calls"] = t.filter_response(response())
            await send({"type": "http.response.start", "status": 200})
            now[0] = 200
            await send({"type": "http.response.body", "body": b"{}"})

        async def send(message):
            if message["type"] == "http.response.body":
                assert t.records["call_1"].sent_at == 200
                if fail:
                    raise OSError("disconnected")

        await module.ToolTimeMiddleware(app)(scope, None, send)

    asyncio.run(run(False))
    assert t.records["call_1"].sent_at == 200
    with pytest.raises(OSError):
        asyncio.run(run(True))
    assert not t.records


def test_writer_failure_is_visible(tmp_path):
    t = module.ToolTimeTracker(tmp_path)
    t.failure = OSError("disk full")
    with pytest.raises(RuntimeError, match="logging failed"):
        t.prepare([], arrived_at=None)
    with pytest.raises(RuntimeError, match="logging failed"):
        t.close()


@pytest.mark.parametrize("tool_name", ["bash", "finish", "exit_workflow"])
def test_template_restores_feedback_once(tracker, tool_name):
    from jinja2 import Environment

    template_path = ROOT.parent / "template/template_round_trip.jinja"
    if not template_path.exists():
        pytest.skip("Deploy sibling template/template_round_trip.jinja first")
    template = Environment().from_string(template_path.read_text())
    t, _ = tracker
    wire = response(name=tool_name)
    t.sent(t.filter_response(wire))
    messages = t.prepare(history(wire), arrived_at=104)
    for message in messages:
        for call in message.get("tool_calls", []):
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    prompt = template.render(messages=messages, add_generation_prompt=True)
    assert prompt.count(f"<parameter={PREDICTION}>") == 1
    assert prompt.count(f"<{OBSERVATION}>") == 1
    assert "<actual_tool_execution_time_seconds>" not in prompt
    assert template.render(messages=messages, add_generation_prompt=True) == prompt


def test_real_mini_agent_loop(tracker):
    pytest.importorskip("mini_agent")
    from mini_agent.agent.silent import SilentAgent
    from mini_agent.llm import LLM
    from mini_agent.llm.openai import OpenAILLM
    from mini_agent.tool import Tool, ToolMetadata, ToolOutcome, ToolResult
    from mini_agent.tool.finish import FinishTool
    from openai.types.chat import ChatCompletion

    t, now = tracker

    class Bash(Tool):
        meta_data = ToolMetadata(
            name="bash",
            description="test",
            param_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

        async def execute(self, arguments):
            assert not arguments
            now[0] += 4
            return ToolResult(
                outcome=ToolOutcome.CONTINUE, duration=3, observation="ok"
            )

    class FakeServer(LLM):
        count = 0

        async def query(self, request):
            messages = OpenAILLM._convert_to_openai_messages(request)
            assert PREDICTION not in json.dumps(messages)
            restored = t.prepare(messages, arrived_at=now[0])
            if self.count:
                assert restored[-1][OBSERVATION] == 4
                assert restored[-2]["tool_calls"][0][PREDICTION] == 3
            name = "bash" if self.count == 0 else "finish"
            wire = response(name=name, call_id=f"call_{self.count}")
            calls = t.filter_response(wire)
            t.sent(calls)
            wire.update(object="chat.completion", created=0, model="test")
            wire["choices"][0].update(index=0, finish_reason="tool_calls")
            wire["choices"][0]["message"].update(role="assistant", content=None)
            self.count += 1
            return OpenAILLM._parse_openai_response(ChatCompletion.model_validate(wire))

    result = asyncio.run(
        SilentAgent(llm=FakeServer(), tools=[Bash(), FinishTool()]).run("test")
    )
    assert result.stop.reason.value == "completed"
    assert result.steps == 2
    assert t.records["call_0"].observed == 4
    assert t.records["call_1"].prediction == 3
    assert t.records["call_1"].observed is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0", 0.0),
        ("12.5", 12.5),
        ("-1", None),
        ("NaN", None),
        ("inf", None),
        ("1e3", None),
        ("1-2", None),
        ("3 seconds", None),
    ],
)
def test_core_parser_cpu_only(value, expected):
    spec = importlib.util.spec_from_file_location(
        "core_tool_time", ROOT / "vllm/v1/core/tool_time_parser.py"
    )
    assert spec is not None and spec.loader is not None
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    class Tokenizer:
        output = (
            f"<tool_call><function=bash><parameter={PREDICTION}>{value}"
            "</parameter></function></tool_call>"
        )

        def decode(self, tokens, **kwargs):
            return self.output

    tokenizer = Tokenizer()
    parser = core.ToolTimePredictionParser(tokenizer)
    assert parser.parse([1]) == expected
    tokenizer.output = tokenizer.output.replace("<function=bash>", "<function=finish>")
    assert parser.parse([1]) == expected
    tokenizer.output = tokenizer.output.replace("<function=finish>", "<function=bash>")
    tokenizer.output += "<tool_call></tool_call>"
    assert parser.parse([1]) is None
