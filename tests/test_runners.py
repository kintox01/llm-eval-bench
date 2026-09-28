from types import SimpleNamespace

import pytest
from pydantic_ai.models.test import TestModel

from evalbench.pricing import estimate_cost
from evalbench.runners.base import InvalidOutput
from evalbench.runners.jev_runner import build_questions, to_raw_result
from evalbench.runners.pydantic_ai_runner import PydanticAIRunner
from evalbench.schemas import EvalCase, TicketLabel

CASE = EvalCase(
    id="t00",
    text="J'ai été débité deux fois",
    expected=TicketLabel(category="billing", urgency="high", refund_requested=True),
)


async def test_pydantic_ai_runner_returns_a_typed_prediction():
    model = TestModel(custom_output_args={"category": "billing", "urgency": "high", "refund_requested": True})
    prediction = await PydanticAIRunner(model, "Label the ticket.").predict(CASE)
    assert prediction.error is None
    assert prediction.output == CASE.expected
    assert prediction.input_tokens and prediction.output_tokens


async def test_pydantic_ai_runner_flags_invalid_output():
    model = TestModel(custom_output_args={"category": "sales", "urgency": "high", "refund_requested": True})
    prediction = await PydanticAIRunner(model, "Label the ticket.", retries=1).predict(CASE)
    assert prediction.output is None
    assert prediction.error_kind == "invalid_output"


def test_jev_questions_follow_the_task_definitions():
    questions = build_questions()
    assert set(questions) == {"category", "urgency", "refund_requested"}
    assert len(questions["urgency"].criteria) == 3


def jev_response(choice="billing", score=1.6, noul=0.8):
    return SimpleNamespace(
        model="jev-test",
        usage=SimpleNamespace(input_tokens=120, output_tokens=0),
        answers={
            "category": SimpleNamespace(choice=choice, confidence=0.7, probabilities={choice: 0.7}),
            "urgency": SimpleNamespace(score=score, confidence=0.6),
            "refund_requested": SimpleNamespace(noul=noul),
        },
    )


def test_jev_answers_map_to_a_ticket_label():
    result = to_raw_result(jev_response(score=1.6, noul=0.8))
    assert result.output == TicketLabel(category="billing", urgency="high", refund_requested=True)
    assert result.confidence == pytest.approx({"category": 0.7, "urgency": 0.6, "refund_requested": 0.8})

    low = to_raw_result(jev_response(score=0.2, noul=0.1))
    assert low.output.urgency == "low" and low.output.refund_requested is False
    assert low.confidence["refund_requested"] == pytest.approx(0.9)


def test_jev_unknown_choice_is_invalid_output():
    with pytest.raises(InvalidOutput):
        to_raw_result(jev_response(choice="sales"))


def test_costs():
    assert estimate_cost("anthropic:claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)
    assert estimate_cost("typesafe:jev-latest", 1_000_000, 0) == pytest.approx(0.042)
    assert estimate_cost("unknown:model", 10, 10) is None
