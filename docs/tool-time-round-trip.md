# Server-side tool round-trip experiment (`yzl-2`)

This branch extends `yzl-1` for a single API process, Qwen + `qwen3_coder`,
non-streaming Chat Completions, one choice and one tool call per response.
The client sends standard OpenAI messages. No client timing field, custom
header, prediction echo, telemetry service, or instance ID is required.

## Start and collect

Copy the sibling `template/template_round_trip.jinja` to the server's template
mount. Keep the existing model, GPU, cache and batching settings. Add/change:

```bash
export VLLM_TOOL_TIME_LOG_DIR=/workspace/tool-time-logs
# Arguments to the existing vllm serve command:
# --api-server-count 1
# --enable-auto-tool-choice --tool-call-parser qwen3_coder
# --chat-template /chat-templates/template_round_trip.jinja
```

In Docker, pass the environment variable to the container and mount the log
directory if it should survive container removal. The default is
`./tool-time-logs` relative to the API process. This configures logging, not an
optimization toggle. The old template remains available for `yzl-1`.

API startup creates `tool-time-<server_run_id>.jsonl`. Each JSON line contains
`schema_version`, `server_run_id`, `sequence`, UTC `timestamp`, and `event`.
Call events contain `request_id`, `tool_call_id`, `tool_name`,
`predicted_tool_round_trip_seconds`, `prediction_valid`, and
`observed_tool_round_trip_seconds`.
Null values mean unobserved/invalid, never zero seconds. There are no full
prompts, arguments or tool outputs in this log.

Events:

| Event | Meaning |
| --- | --- |
| `tool_sent` | Prediction captured at the response body send boundary |
| `tool_returned` | First matching result arrived; observation fixed |
| `unsupported_multiple_tools` | Multiple calls; no independent timings |
| `send_failed` | Response send failed; discard this timing sample |
| `record_evicted` | Retention/capacity removal; `returned` says if observed |
| `unobserved_at_shutdown` | Pending call at normal shutdown |

The writer batches up to 256 events per write and flushes each batch. Normal
shutdown drains the queue. Disk/queue failures are explicitly logged and make
subsequent tracker operations fail rather than silently losing samples. An
abrupt kill can lose queued events: unmatched `tool_sent` events are censored
observations, not measured durations.

## Measurement and prompt semantics

The clock starts immediately before sending the final HTTP response body and
ends at the next HTTP request's ingress, before parsing/rendering/Engine queueing.
This is a server boundary approximation: it includes transport, tool execution,
client processing and client-side telemetry overhead. It is not pure execution
time. Invalid HTTP/schema requests do not complete calls.

Only a returned tool message preceded by a matching assistant call (ID, name,
canonical business arguments) completes a record. Retransmitted history never
updates a completed duration. Records are process-local, limited to 100,000 and
24 hours, reclaimed on response sends; expired records are ignored during
restoration. A restart loses online records. `/render` can restore existing
history but never records a return or starts a timer.

The model parameter and both API/Core parsers consistently use
`predicted_tool_round_trip_seconds` to predict round-trip seconds.
The API removes it from both business arguments and response extension fields.
Before rendering, the server restores its own predictions and
`observed_tool_round_trip_seconds` on a copy of the submitted history. Client
timing and prediction extension fields are not trusted as feedback.
Use the new template; the old template describes a different prediction target.

Core still sets `deadline = monotonic_now + prediction` before freeing KV.
It does not consult the API table. The offset between Core release and API
response send is an accepted approximation. All tool names follow the same
prediction, timing and deadline rules. Calls with no subsequent result remain
unobserved, including any tool that happens to terminate a client's workflow.
Multiple tool calls have no predicted deadline. Scheduling and eviction
selection are otherwise unchanged from `yzl-1`.

## Mini projects and manual analysis

`mini-agent` already preserves `tool_calls[].id` and `tool_call_id` in standard
messages. Both `bash` and empty-argument `finish` keep their original schemas
from the client's perspective. No change to the three mini repositories is
required.

`mini-telemetry` logs `tool_started`, `tool_completed`, and `tool_failed` with
`tool_call_id`, `instance_id` and `step`. Its completed tool `duration` is the
client's execution time. To inspect one call from both files:

```bash
jq -c 'select(.tool_call_id == "call_xxx")' tool-time-*.jsonl
jq -c 'select(.tool_call_id == "call_xxx")' trace.jsonl
```

Match by tool-call ID and retain `server_run_id` when combining server runs.
Coverage is valid predictions / single-tool calls sent; exclude
multiple-call and failed-send records. Accuracy uses only matched
`tool_returned` observations; report missing predictions and unobserved calls
separately. Treat a later `send_failed` as invalidating that call's sample.

`mini-tbench-harness` stops admitting new tasks at the telemetry window end but
lets active agents continue. Server logging continues throughout. For manual
window analysis select calls visible in telemetry, and label returns after the
window separately; do not silently combine full-server totals with window-only
harness metrics. UTC timestamps across hosts require synchronized clocks if
used for window boundaries; the measured durations do not.

`instances.completed` counts all instance stop events in the window;
`instances.stop_reasons.completed` counts agents calling finish. Neither is a
verifier pass rate. Keep the original metrics.json and trace.jsonl intact.

## Validation

CPU-only tracker/ASGI tests (no model download):

```bash
.venv/bin/python -m pytest --confcutdir=tests/tool_time tests/tool_time -v
```

In a fully installed vLLM environment also run:

```bash
.venv/bin/python -m pytest tests/v1/core/test_tool_time_parser.py tests/v1/core/test_prefix_caching.py tests/tool_parsers/test_qwen3coder_tool_parser.py tests/entrypoints/openai/chat_completion/test_serving_chat.py -v
```

The optional mini integration test runs when `mini-agent` is installed in the
test environment. Remote smoke: run one mini-tbench task, inspect filtered tool
responses, restored history and paired logs, then run a small concurrent batch.
Verify `finish` stops normally and prefix/eviction counters still export.
