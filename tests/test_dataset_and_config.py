import pytest
from pydantic import ValidationError

from evalbench.config import RunConfig, load_cases, load_config
from evalbench.task import CATEGORIES, URGENCY_LEVELS, render_prompt


def test_dataset_is_valid_and_covers_every_label():
    cases = load_cases("data/tickets.jsonl")
    assert len(cases) == 30
    assert len({c.id for c in cases}) == len(cases)
    assert {c.expected.category for c in cases} == set(CATEGORIES)
    assert {c.expected.urgency for c in cases} == set(URGENCY_LEVELS)
    assert {c.expected.refund_requested for c in cases} == {True, False}


def test_config_loads():
    config = load_config("configs/tickets.yaml")
    assert any(run.is_jev for run in config.runs)
    assert all(run.prompt for run in config.runs if not run.is_jev)


def test_llm_run_requires_a_prompt():
    with pytest.raises(ValidationError):
        RunConfig(name="x", model="anthropic:claude-haiku-4-5")
    assert RunConfig(name="jev", model="typesafe:jev-latest").prompt is None


@pytest.mark.parametrize("path", ["prompts/tickets/v1_simple.md", "prompts/tickets/v2_few_shot.md"])
def test_prompts_render_every_definition(path):
    rendered = render_prompt(open(path, encoding="utf-8").read())
    assert "$" not in rendered
    for description in [*CATEGORIES.values(), *URGENCY_LEVELS.values()]:
        assert description in rendered
