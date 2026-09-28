import asyncio

from rich.progress import Progress

from evalbench.runners import Prediction, Runner
from evalbench.schemas import EvalCase


async def run_cases(
    runner: Runner,
    cases: list[EvalCase],
    concurrency: int,
    progress: Progress | None = None,
    label: str = "",
) -> list[Prediction]:
    semaphore = asyncio.Semaphore(concurrency)
    task_id = progress.add_task(label, total=len(cases)) if progress else None

    async def one(case: EvalCase) -> Prediction:
        async with semaphore:
            prediction = await runner.predict(case)
        if progress is not None:
            progress.advance(task_id)
        return prediction

    try:
        return await asyncio.gather(*(one(case) for case in cases))
    finally:
        await runner.aclose()
