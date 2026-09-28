from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "bug", "account", "feature_request", "other"]
Urgency = Literal["low", "medium", "high"]

FIELDS = ("category", "urgency", "refund_requested")


class TicketLabel(BaseModel):
    category: Category
    urgency: Urgency
    refund_requested: bool = Field(
        description="True only if the customer explicitly asks for money back."
    )


class EvalCase(BaseModel):
    id: str
    text: str
    expected: TicketLabel
    note: str | None = None
