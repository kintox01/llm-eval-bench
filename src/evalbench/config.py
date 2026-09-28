from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from evalbench.schemas import EvalCase


class RunConfig(BaseModel):
    name: str
    model: str
    prompt: str | None = None
    retries: int = 1

    @property
    def is_jev(self) -> bool:
        return self.model.startswith("typesafe:")

    @model_validator(mode="after")
    def _prompt_required_for_llms(self) -> "RunConfig":
        if not self.is_jev and self.prompt is None:
            raise ValueError(f"run '{self.name}': LLM runs need a `prompt`")
        return self


class BenchConfig(BaseModel):
    dataset: str
    concurrency: int = Field(default=5, ge=1)
    runs: list[RunConfig]

    @model_validator(mode="after")
    def _unique_names(self) -> "BenchConfig":
        names = [run.name for run in self.runs]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate run names: {sorted(duplicates)}")
        return self


def load_config(path: str | Path) -> BenchConfig:
    return BenchConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_cases(path: str | Path) -> list[EvalCase]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvalCase.model_validate_json(line) for line in lines if line.strip()]
