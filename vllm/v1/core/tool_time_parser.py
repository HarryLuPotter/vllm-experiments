# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract tool execution-time predictions from completed model output."""

import math
import re
from collections.abc import Sequence

from vllm.tokenizers import TokenizerLike

_PREDICTION_PARAMETER = "predicted_tool_execution_time_seconds"
_OPEN_TAG = f"<parameter={_PREDICTION_PARAMETER}>"
_CLOSE_TAG = "</parameter>"
_NON_NEGATIVE_DECIMAL = r"(?:0|[1-9]\d*)(?:\.\d+)?"
_PREDICTION_PATTERN = re.compile(
    re.escape(_OPEN_TAG)
    + r"\s*("
    + _NON_NEGATIVE_DECIMAL
    + r")\s*"
    + re.escape(_CLOSE_TAG),
    re.DOTALL,
)


class ToolTimePredictionParser:
    """Qwen3 Coder parser used by the KV-cache eviction policy."""

    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer

    def parse(self, output_token_ids: Sequence[int]) -> float | None:
        if not output_token_ids:
            return None

        model_output = self.tokenizer.decode(
            list(output_token_ids), skip_special_tokens=False
        )
        if model_output.count(_OPEN_TAG) != 1:
            return None

        matches = _PREDICTION_PATTERN.findall(model_output)
        if len(matches) != 1:
            return None

        value = float(matches[0])
        return value if math.isfinite(value) else None
