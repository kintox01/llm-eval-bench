import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from evalbench.pricing import estimate_cost
from evalbench.schemas import EvalCase, TicketLabel

ErrorKind = Literal["invalid_output", "api_error"]


class InvalidOutput(Exception):
    pass


@dataclass
class RawResult:
    output: TicketLabel
    input_tokens: int | None = None
    output_tokens: int | None = None
    confidence: dict[str, float] | None = None
    extra: dict[str, Any] | None = None


@dataclass
class Prediction:
    case_id: str
    output: TicketLabel | None
    error_kind: ErrorKind | None = None
    error: str | None = None
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    confidence: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Runner(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def _call(self, text: str) -> RawResult: ...

    async def aclose(self) -> None:
        pass

    async def predict(self, case: EvalCase) -> Prediction:
        start = time.perf_counter()
        try:
            result = await self._call(case.text)
        except InvalidOutput as exc:
            return self._failure(case, "invalid_output", exc, start)
        except Exception as exc:
            return self._failure(case, "api_error", exc, start)
        return Prediction(
            case_id=case.id,
            output=result.output,
            latency_s=time.perf_counter() - start,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=estimate_cost(self.model, result.input_tokens, result.output_tokens),
            confidence=result.confidence,
            extra=result.extra or {},
        )

    def _failure(self, case: EvalCase, kind: ErrorKind, exc: Exception, start: float) -> Prediction:
        return Prediction(
            case_id=case.id,
            output=None,
            error_kind=kind,
            error=f"{type(exc).__name__}: {exc}",
            latency_s=time.perf_counter() - start,
        )
