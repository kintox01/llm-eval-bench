from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model

from evalbench.runners.base import InvalidOutput, RawResult, Runner
from evalbench.schemas import TicketLabel


class PydanticAIRunner(Runner):
    def __init__(self, model: str | Model, instructions: str, retries: int = 1):
        super().__init__(model if isinstance(model, str) else model.model_name)
        self.agent = Agent(
            model,
            output_type=TicketLabel,
            instructions=instructions,
            retries=retries,
        )

    async def _call(self, text: str) -> RawResult:
        try:
            result = await self.agent.run(text)
        except UnexpectedModelBehavior as exc:
            raise InvalidOutput(str(exc)) from exc
        usage = result.usage
        return RawResult(
            output=result.output,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            extra={"requests": usage.requests},
        )
