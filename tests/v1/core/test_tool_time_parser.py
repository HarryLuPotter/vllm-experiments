# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.tool_time_parser import ToolTimePredictionParser

pytestmark = pytest.mark.cpu_test


class FakeTokenizer:
    def __init__(self, output: str):
        self.output = output

    def decode(self, token_ids: list[int], **kwargs) -> str:
        assert token_ids == [1, 2, 3]
        assert kwargs == {"skip_special_tokens": False}
        return self.output


def parse(output: str) -> float | None:
    parser = ToolTimePredictionParser(FakeTokenizer(output))  # type: ignore[arg-type]
    return parser.parse([1, 2, 3])


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0.0), ("12", 12.0), ("12.5", 12.5), ("120.0", 120.0)],
)
def test_parse_valid_prediction(value: str, expected: float):
    output = (
        "<think>reasoning</think>\n<tool_call>\n<function=bash>\n"
        "<parameter=command>\nls -la\n</parameter>\n"
        "<parameter=predicted_tool_round_trip_seconds>\n"
        f"{value}\n"
        "</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    assert parse(output) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "-1",
        "NaN",
        "inf",
        "1e3",
        "12 seconds",
        "1-2",
        ".5",
        "01",
        "9" * 400,
    ],
)
def test_invalid_prediction_is_omitted(value: str):
    output = f"<parameter=predicted_tool_round_trip_seconds>{value}</parameter>"
    assert parse(output) is None


def test_missing_or_duplicate_prediction_is_omitted():
    assert parse("<tool_call></tool_call>") is None
    prediction = "<parameter=predicted_tool_round_trip_seconds>1</parameter>"
    assert parse(prediction + prediction) is None


def test_incomplete_prediction_is_omitted():
    assert parse("<parameter=predicted_tool_round_trip_seconds>12.5") is None


def test_empty_tokens_are_omitted_without_decoding():
    parser = ToolTimePredictionParser(FakeTokenizer("unused"))  # type: ignore[arg-type]
    assert parser.parse([]) is None


def test_tool_names_are_uniform_but_multiple_calls_have_no_deadline():
    prediction = "<parameter=predicted_tool_round_trip_seconds>1</parameter>"
    assert (
        parse(f"<tool_call><function=finish>{prediction}</function></tool_call>") == 1.0
    )
    assert parse(f"<tool_call>{prediction}</tool_call><tool_call></tool_call>") is None
