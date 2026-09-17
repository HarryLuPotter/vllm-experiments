# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-API-process tool round-trip observations for the Qwen experiment."""

import copy
import hashlib
import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PREDICTION = "predicted_tool_round_trip_seconds"
OBSERVATION = "observed_tool_round_trip_seconds"
logger = logging.getLogger(__name__)


def arguments_digest(arguments: str | dict) -> str:
    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(parsed, dict):
        raise TypeError("Tool arguments must be a JSON object")
    business_arguments = dict(parsed)
    business_arguments.pop(PREDICTION, None)
    return hashlib.sha256(
        json.dumps(business_arguments, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class CallRecord:
    request_id: str
    tool_call_id: str
    tool_name: str
    arguments_digest: str
    prediction: float | None
    sent_at: float
    observed: float | None = None


class ToolTimeTracker:
    def __init__(self, directory: str | Path, *, clock=time.monotonic):
        self.clock = clock
        self.server_run_id = uuid.uuid4().hex
        self.records: OrderedDict[str, CallRecord] = OrderedDict()
        self.capacity = 100_000
        self.retention = 24 * 3600
        self.sequence = 0
        self.failure: BaseException | None = None
        self.closed = False
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"tool-time-{self.server_run_id}.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
        self.queue: queue.Queue = queue.Queue(maxsize=100_000)
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()

    def _write(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                batch = [item]
                stop = False
                for _ in range(255):
                    try:
                        item = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        stop = True
                        break
                    batch.append(item)
                self.file.writelines(json.dumps(x) + "\n" for x in batch)
                self.file.flush()
                if stop:
                    break
        except BaseException as exc:
            self.failure = exc
            logger.exception("Tool-time JSONL writer failed; observations incomplete")
        finally:
            self.file.close()

    def check(self):
        if self.failure is not None:
            raise RuntimeError("Tool-time JSONL logging failed") from self.failure
        if self.closed:
            raise RuntimeError("Tool-time tracker is closed")

    def event(self, event: str, record: CallRecord | None = None, **fields):
        self.check()
        self.sequence += 1
        data = {
            "schema_version": 1,
            "server_run_id": self.server_run_id,
            "sequence": self.sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        if record is not None:
            data.update(
                request_id=record.request_id,
                tool_call_id=record.tool_call_id,
                tool_name=record.tool_name,
                predicted_tool_round_trip_seconds=record.prediction,
                prediction_valid=record.prediction is not None,
                observed_tool_round_trip_seconds=record.observed,
            )
        data.update(fields)
        try:
            self.queue.put_nowait(data)
        except queue.Full as exc:
            self.failure = exc
            logger.error("Tool-time JSONL queue full; observations incomplete")
            raise RuntimeError("Tool-time JSONL queue full") from exc

    def prune(self, reserve: int = 0):
        now = self.clock()
        while self.records:
            record = next(iter(self.records.values()))
            if (
                now - record.sent_at < self.retention
                and len(self.records) <= self.capacity - reserve
            ):
                break
            self.records.pop(record.tool_call_id)
            self.event("record_evicted", record, returned=record.observed is not None)

    def prepare(self, messages: list[dict], *, arrived_at: float | None) -> list[dict]:
        """Copy and enrich history; None arrival means a read-only render."""
        self.check()
        result = copy.deepcopy(messages)
        matched: dict[str, CallRecord] = {}
        now = self.clock()
        for message in result:
            message.pop("tool_exec_time", None)
            message.pop(OBSERVATION, None)
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    call.pop(PREDICTION, None)
                    function = call.get("function", {})
                    matched.pop(call.get("id"), None)
                    record = self.records.get(call.get("id"))
                    try:
                        digest = arguments_digest(function.get("arguments", {}))
                        arguments = function.get("arguments", {})
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                            if PREDICTION in arguments:
                                arguments.pop(PREDICTION)
                                function["arguments"] = json.dumps(arguments)
                        elif isinstance(arguments, dict):
                            arguments.pop(PREDICTION, None)
                    except (ValueError, TypeError):
                        continue
                    if (
                        record is not None
                        and now - record.sent_at < self.retention
                        and record.tool_name == function.get("name")
                        and record.arguments_digest == digest
                    ):
                        matched[record.tool_call_id] = record
                        if record.prediction is not None:
                            call[PREDICTION] = record.prediction
            elif message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                record = (
                    matched.get(tool_call_id) if isinstance(tool_call_id, str) else None
                )
                if record is None:
                    continue
                if (
                    record.observed is None
                    and arrived_at is not None
                    and arrived_at >= record.sent_at
                ):
                    record.observed = arrived_at - record.sent_at
                    self.event("tool_returned", record)
                if record.observed is not None:
                    message[OBSERVATION] = record.observed
        return result

    def filter_response(self, response: dict) -> list[dict]:
        """Remove private fields from the wire; return metadata for send time."""
        calls = []
        for choice in response.get("choices", []):
            for call in choice.get("message", {}).get("tool_calls") or []:
                prediction = call.pop(PREDICTION, None)
                function = call["function"]
                arguments = json.loads(function["arguments"])
                if PREDICTION in arguments:
                    arguments.pop(PREDICTION)
                    function["arguments"] = json.dumps(arguments)
                if (
                    isinstance(prediction, bool)
                    or not isinstance(prediction, (int, float))
                    or not math.isfinite(prediction)
                    or prediction < 0
                ):
                    prediction = None
                calls.append(
                    dict(
                        request_id=response["id"],
                        tool_call_id=call["id"],
                        tool_name=function["name"],
                        arguments_digest=arguments_digest(arguments),
                        prediction=prediction,
                    )
                )
        return calls

    def sent(self, calls: list[dict]):
        self.check()
        reserve = int(len(calls) == 1)
        self.prune(reserve)
        now = self.clock()
        for call in calls:
            record = CallRecord(**call, sent_at=now)
            if len(calls) != 1:
                self.event("unsupported_multiple_tools", record)
            else:
                self.records[record.tool_call_id] = record
                self.event("tool_sent", record)

    def send_failed(self, calls: list[dict]):
        for call in calls:
            record = self.records.pop(call["tool_call_id"], None)
            if record is None:
                record = CallRecord(**call, sent_at=self.clock())
            self.event("send_failed", record)

    def close(self):
        try:
            for record in self.records.values():
                if record.observed is None:
                    self.event("unobserved_at_shutdown", record)
            self.records.clear()
        finally:
            # Timed puts also notice a failed writer instead of hanging shutdown.
            while self.writer.is_alive():
                try:
                    self.queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self.writer.join()
            self.closed = True
        if self.failure is not None:
            raise RuntimeError("Tool-time JSONL logging failed") from self.failure


class ToolTimeMiddleware:
    """Timestamp before HTTP parsing and at the final ASGI response body."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        state["tool_time_arrived_at"] = time.monotonic()

        async def timed_send(message):
            calls = state.get("tool_time_calls")
            tracker = state.get("tool_time_tracker")
            if (
                calls is not None
                and tracker is not None
                and message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                tracker.sent(calls)
                await send(message)
            else:
                await send(message)

        try:
            await self.app(scope, receive, timed_send)
        except BaseException:
            tracker = state.get("tool_time_tracker")
            if tracker is not None:
                tracker.send_failed(state.get("tool_time_calls", []))
            raise


def get_tracker(request):
    return getattr(request.app.state, "tool_time_tracker", None)


def log_directory():
    return os.environ.get("VLLM_TOOL_TIME_LOG_DIR", "./tool-time-logs")
