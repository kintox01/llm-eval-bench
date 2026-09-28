from pathlib import Path

from evalbench.config import RunConfig
from evalbench.runners.base import Prediction, Runner
from evalbench.task import render_prompt

__all__ = ["Prediction", "Runner", "make_runner"]


def make_runner(config: RunConfig) -> Runner:
    if config.is_jev:
        from evalbench.runners.jev_runner import JevRunner

        return JevRunner(config.model)

    from evalbench.runners.pydantic_ai_runner import PydanticAIRunner

    instructions = render_prompt(Path(config.prompt).read_text(encoding="utf-8"))
    return PydanticAIRunner(config.model, instructions, retries=config.retries)
