# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("prediction", "expected_deadline"), [(12.5, 112.5), (None, None)]
)
def test_prediction_is_parsed_before_normal_request_is_freed(
    monkeypatch, prediction: float | None, expected_deadline: float | None
):
    scheduler = create_scheduler(skip_tokenizer_init=True)
    parser = Mock()
    parser.parse.return_value = prediction
    scheduler.tool_time_prediction_parser = parser
    monkeypatch.setattr(
        "vllm.v1.core.sched.scheduler.time.monotonic", lambda: 100.0
    )

    request = create_requests(num_requests=1, max_tokens=10)[0]
    request.num_computed_tokens = request.num_tokens
    request.status = RequestStatus.RUNNING
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[EOS_TOKEN_ID]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(scheduler_output, model_output)

    parser.parse.assert_called_once_with(request.output_token_ids)
    assert request.predicted_reuse_deadline == expected_deadline
    assert request.request_id not in scheduler.requests


def test_abort_does_not_parse_prediction():
    scheduler = create_scheduler(skip_tokenizer_init=True)
    parser = Mock()
    scheduler.tool_time_prediction_parser = parser
    request = create_requests(num_requests=1)[0]
    scheduler.add_request(request)

    scheduler.finish_requests(
        request.request_id, RequestStatus.FINISHED_ABORTED
    )

    parser.parse.assert_not_called()
    assert request.predicted_reuse_deadline is None
