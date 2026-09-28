import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evalbench.config import RunConfig
from evalbench.runners.base import Prediction
from evalbench.schemas import EvalCase, TicketLabel

DEFAULT_DB = Path("results/evalbench.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_path TEXT,
    prompt_hash TEXT,
    dataset TEXT NOT NULL,
    n_cases INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT NOT NULL REFERENCES runs(id),
    case_id TEXT NOT NULL,
    text TEXT NOT NULL,
    expected TEXT NOT NULL,
    output TEXT,
    error_kind TEXT,
    error TEXT,
    latency_s REAL NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    confidence TEXT,
    extra TEXT,
    PRIMARY KEY (run_id, case_id)
);
"""


@dataclass
class RunInfo:
    id: str
    name: str
    model: str
    prompt_path: str | None
    prompt_hash: str | None
    dataset: str
    n_cases: int
    created_at: str


@dataclass
class Record:
    case: EvalCase
    prediction: Prediction


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_run(
    conn: sqlite3.Connection,
    config: RunConfig,
    dataset: str,
    cases: list[EvalCase],
    predictions: list[Prediction],
) -> str:
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{config.name}-{uuid.uuid4().hex[:6]}"
    prompt_hash = None
    if config.prompt:
        prompt_hash = hashlib.sha256(Path(config.prompt).read_bytes()).hexdigest()[:12]
    cases_by_id = {case.id: case for case in cases}
    with conn:
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                config.name,
                config.model,
                config.prompt,
                prompt_hash,
                dataset,
                len(predictions),
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        conn.executemany(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    p.case_id,
                    cases_by_id[p.case_id].text,
                    cases_by_id[p.case_id].expected.model_dump_json(),
                    p.output.model_dump_json() if p.output else None,
                    p.error_kind,
                    p.error,
                    p.latency_s,
                    p.input_tokens,
                    p.output_tokens,
                    p.cost_usd,
                    json.dumps(p.confidence) if p.confidence else None,
                    json.dumps(p.extra),
                )
                for p in predictions
            ],
        )
    return run_id


def list_runs(conn: sqlite3.Connection) -> list[RunInfo]:
    rows = conn.execute("SELECT * FROM runs ORDER BY rowid DESC").fetchall()
    return [RunInfo(**dict(row)) for row in rows]


def latest_run_per_name(conn: sqlite3.Connection) -> list[RunInfo]:
    latest: dict[str, RunInfo] = {}
    for run in list_runs(conn):
        latest.setdefault(run.name, run)
    return list(latest.values())


def get_run(conn: sqlite3.Connection, run_id: str) -> RunInfo:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown run: {run_id}")
    return RunInfo(**dict(row))


def load_records(conn: sqlite3.Connection, run_id: str) -> list[Record]:
    rows = conn.execute(
        "SELECT * FROM predictions WHERE run_id = ? ORDER BY case_id", (run_id,)
    ).fetchall()
    return [
        Record(
            case=EvalCase(
                id=row["case_id"],
                text=row["text"],
                expected=TicketLabel.model_validate_json(row["expected"]),
            ),
            prediction=Prediction(
                case_id=row["case_id"],
                output=TicketLabel.model_validate_json(row["output"]) if row["output"] else None,
                error_kind=row["error_kind"],
                error=row["error"],
                latency_s=row["latency_s"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cost_usd=row["cost_usd"],
                confidence=json.loads(row["confidence"]) if row["confidence"] else None,
                extra=json.loads(row["extra"]) if row["extra"] else {},
            ),
        )
        for row in rows
    ]
