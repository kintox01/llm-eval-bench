from pydantic import ValidationError
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score, SystemOneResponse

from evalbench.runners.base import InvalidOutput, RawResult, Runner
from evalbench.schemas import TicketLabel
from evalbench.task import CATEGORIES, REFUND_DEFINITION, URGENCY_LEVELS

URGENCY_ORDER = list(URGENCY_LEVELS)


def build_questions() -> dict[str, Choice | Score | Noul]:
    return {
        "category": Choice(
            instructions="Which team should handle this support ticket",
            criteria=CATEGORIES,
        ),
        "urgency": Score(
            instructions="How urgent this support ticket is",
            criteria=list(URGENCY_LEVELS.values()),
        ),
        "refund_requested": Noul(instructions=REFUND_DEFINITION),
    }


def to_raw_result(response: SystemOneResponse) -> RawResult:
    category = response.answers["category"]
    urgency = response.answers["urgency"]
    refund = response.answers["refund_requested"]

    # score is a float (e.g. 1.4)
    level = min(max(round(urgency.score), 0), len(URGENCY_ORDER) - 1)
    try:
        label = TicketLabel(
            category=category.choice,
            urgency=URGENCY_ORDER[level],
            refund_requested=refund.noul >= 0.5,
        )
    except ValidationError as exc:
        raise InvalidOutput(str(exc)) from exc

    return RawResult(
        output=label,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        confidence={
            "category": category.confidence,
            "urgency": urgency.confidence,
            "refund_requested": max(refund.noul, 1 - refund.noul),
        },
        extra={
            "model_version": response.model,
            "category_probabilities": dict(category.probabilities),
            "urgency_score": urgency.score,
            "refund_probability": refund.noul,
        },
    )


class JevRunner(Runner):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = AsyncTypeSafeClient(model=model.removeprefix("typesafe:"))
        self.questions = build_questions()

    async def _call(self, text: str) -> RawResult:
        response = await self.client.system_one(state={"ticket": text}, questions=self.questions)
        return to_raw_result(response)

    async def aclose(self) -> None:
        await self.client.aclose()
