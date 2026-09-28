import asyncio
import os
from pathlib import Path
from typing import Annotated

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from evalbench import scoring, storage
from evalbench.config import load_cases, load_config
from evalbench.engine import run_cases
from evalbench.runners import make_runner
from evalbench.schemas import FIELDS

app = typer.Typer(help="Benchmark LLMs and Jev on typed classification tasks.", no_args_is_help=True)
console = Console()

DbOption = Annotated[Path, typer.Option("--db", help="SQLite results file.")]


@app.callback()
def _setup() -> None:
    load_dotenv()


@app.command()
def run(
    config_path: Annotated[Path, typer.Argument(help="YAML config, e.g. configs/tickets.yaml")],
    only: Annotated[list[str] | None, typer.Option(help="Run only these run names.")] = None,
    limit: Annotated[int | None, typer.Option(help="Use only the first N cases (cheap smoke test).")] = None,
    concurrency: Annotated[int | None, typer.Option(min=1, help="Override the config's concurrency.")] = None,
    db: DbOption = storage.DEFAULT_DB,
) -> None:
    """Run every configuration of the YAML file on the dataset and store the results."""
    config = load_config(config_path)
    if concurrency:
        config.concurrency = concurrency
    cases = load_cases(config.dataset)[:limit]
    selected = [r for r in config.runs if not only or r.name in only]
    if not selected:
        raise typer.BadParameter(f"no run matches {only}")

    # create all clients first so a missing key fails before any paid call
    runners = {}
    for run_config in selected:
        try:
            runners[run_config.name] = make_runner(run_config)
        except Exception as exc:
            console.print(f"[red]Cannot start run '{run_config.name}':[/] {exc}")
            raise typer.Exit(1) from exc

    conn = storage.connect(db)

    async def run_all() -> list[str]:
        run_ids = []
        with Progress(console=console) as progress:
            for run_config in selected:
                runner = runners[run_config.name]
                predictions = await run_cases(runner, cases, config.concurrency, progress, label=run_config.name)
                run_ids.append(storage.save_run(conn, run_config, config.dataset, cases, predictions))
        return run_ids

    run_ids = asyncio.run(run_all())
    _print_report(conn, [storage.get_run(conn, run_id) for run_id in run_ids])


@app.command()
def report(
    run_ids: Annotated[list[str] | None, typer.Argument(help="Runs to show. Default: latest run of each name.")] = None,
    db: DbOption = storage.DEFAULT_DB,
) -> None:
    """Compare runs side by side."""
    conn = storage.connect(db)
    runs = [storage.get_run(conn, rid) for rid in run_ids] if run_ids else storage.latest_run_per_name(conn)
    if not runs:
        console.print("No runs yet. Start with: [bold]evalbench run configs/tickets.yaml[/]")
        raise typer.Exit(1)
    _print_report(conn, runs)


@app.command()
def compare(
    baseline: Annotated[str, typer.Argument(help="Reference run id.")],
    candidate: Annotated[str, typer.Argument(help="Run id to check against the reference.")],
    db: DbOption = storage.DEFAULT_DB,
    fail_on_regression: Annotated[bool, typer.Option(help="Exit with code 1 if any case regressed (for CI).")] = False,
) -> None:
    """List the cases a run fixed or broke compared to another one."""
    conn = storage.connect(db)
    diff = scoring.diff_runs(storage.load_records(conn, baseline), storage.load_records(conn, candidate))
    console.print(f"[green]Fixed ({len(diff['fixed'])})[/]: {', '.join(diff['fixed']) or '-'}")
    console.print(f"[red]Regressed ({len(diff['regressed'])})[/]: {', '.join(diff['regressed']) or '-'}")
    if fail_on_regression and diff["regressed"]:
        raise typer.Exit(1)


@app.command()
def errors(
    run_id: Annotated[str, typer.Argument(help="Run id.")],
    db: DbOption = storage.DEFAULT_DB,
) -> None:
    """Show every case a run got wrong, with the expected and predicted labels."""
    conn = storage.connect(db)
    table = Table("case", "ticket", "expected", "predicted", show_lines=True)
    for record in storage.load_records(conn, run_id):
        if scoring.is_exact_match(record):
            continue
        p = record.prediction
        predicted = f"[red]{p.error_kind}[/]\n{p.error}" if p.output is None else _label(p.output, record)
        table.add_row(record.case.id, record.case.text, _label(record.case.expected), predicted)
    console.print(table)


def _label(label, record: storage.Record | None = None) -> str:
    parts = []
    for field in FIELDS:
        value = str(getattr(label, field))
        if record is not None and not scoring.is_correct(record, field):
            value = f"[red]{value}[/]"
        parts.append(f"{field}={value}")
    return "\n".join(parts)


def _print_report(conn, runs: list[storage.RunInfo]) -> None:
    table = Table(title="Results")
    for column in ("run", "model", *FIELDS, "exact", "invalid", "api err", "p50", "p95", "$ / 1k", "ECE"):
        table.add_column(column, justify="left" if column in ("run", "model") else "right")
    for run_info in runs:
        m = scoring.compute_metrics(storage.load_records(conn, run_info.id))
        table.add_row(
            run_info.name,
            run_info.model,
            *(f"{m.accuracy[f]:.0%}" for f in FIELDS),
            f"{m.exact_match:.0%}",
            f"{m.invalid_rate:.0%}",
            f"{m.api_error_rate:.0%}",
            f"{m.latency_p50:.2f}s",
            f"{m.latency_p95:.2f}s",
            "?" if m.cost_per_1k is None else f"{m.cost_per_1k:.3f}",
            "-" if m.ece is None else f"{m.ece:.3f}",
        )
    console.print(table)
    console.print("[dim]Run ids:[/] " + ", ".join(r.id for r in runs))
