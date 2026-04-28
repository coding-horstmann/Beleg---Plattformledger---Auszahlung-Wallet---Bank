from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def sqlite_ready(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for col in prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared[col]):
            prepared[col] = prepared[col].dt.strftime("%Y-%m-%d")
    return prepared


def save_run_to_sqlite(
    db_path: str | Path,
    docs: pd.DataFrame,
    bank: pd.DataFrame,
    matches: pd.DataFrame,
    links: pd.DataFrame,
    import_issues: Iterable[str],
    settings: dict,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> str:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                settings_json TEXT NOT NULL,
                import_issues_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO runs(run_id, created_at, settings_json, import_issues_json) VALUES (?, ?, ?, ?)",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(settings, ensure_ascii=False, default=str),
                json.dumps(list(import_issues), ensure_ascii=False),
            ),
        )
        for name, frame in {
            "accountable_docs": docs,
            "bank_transactions": bank,
            "matches": matches,
            "match_links": links,
            **(extra_tables or {}),
        }.items():
            prepared = sqlite_ready(frame)
            prepared.insert(0, "run_id", run_id)
            prepared.to_sql(name, conn, if_exists="append", index=False)
    return run_id
