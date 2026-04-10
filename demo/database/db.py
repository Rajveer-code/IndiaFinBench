"""
database/db.py
--------------
Purpose:  SQLite-backed leaderboard database for IndiaFinBench Spaces.
          Stores evaluation results, supports leaderboard retrieval.
Inputs:   baselines.json (pre-populated on first init)
Outputs:  Pandas DataFrame via get_leaderboard()
Usage:
    from database.db import init_db, save_result, get_leaderboard
    init_db()
    df = get_leaderboard()
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

DB_PATH        = Path(__file__).parent.parent / "leaderboard.db"
BASELINES_JSON = Path(__file__).parent.parent / "data/baselines.json"

TASK_SHORTS = ["REG", "NUM", "CON", "TMP"]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id     TEXT    NOT NULL,
    label        TEXT    NOT NULL,
    hf_id        TEXT    NOT NULL,
    params       TEXT    DEFAULT 'Unknown',
    model_type   TEXT    DEFAULT 'Open',
    overall      REAL    NOT NULL,
    score_REG    REAL    DEFAULT 0.0,
    score_NUM    REAL    DEFAULT 0.0,
    score_CON    REAL    DEFAULT 0.0,
    score_TMP    REAL    DEFAULT 0.0,
    n_items      INTEGER DEFAULT 150,
    submitted_at TEXT    NOT NULL,
    is_baseline  INTEGER DEFAULT 0,
    notes        TEXT    DEFAULT ''
)
"""


# ── Connection helper ──────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Open (or create) the leaderboard SQLite database.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── Initialisation ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the results table and pre-populate with baseline models.

    Safe to call multiple times — baselines are inserted only once (by hf_id).
    """
    conn = _connect()
    with conn:
        conn.execute(CREATE_TABLE_SQL)

    # Load baselines from JSON
    if not BASELINES_JSON.exists():
        print(f"  [WARN] baselines.json not found at {BASELINES_JSON}")
        conn.close()
        return

    with BASELINES_JSON.open(encoding="utf-8") as f:
        baselines = json.load(f)

    with conn:
        for b in baselines:
            # Only insert if this hf_id is not already present
            existing = conn.execute(
                "SELECT id FROM results WHERE hf_id = ?", (b["hf_id"],)
            ).fetchone()
            if existing:
                continue

            scores = b.get("scores", {})
            conn.execute(
                """INSERT INTO results
                   (model_id, label, hf_id, params, model_type,
                    overall, score_REG, score_NUM, score_CON, score_TMP,
                    n_items, submitted_at, is_baseline)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    b["model_id"], b["label"], b["hf_id"],
                    b.get("params", "N/A"), b.get("type", "API"),
                    b["overall"],
                    scores.get("REG", 0.0), scores.get("NUM", 0.0),
                    scores.get("CON", 0.0), scores.get("TMP", 0.0),
                    b.get("n_items", 150),
                    b.get("submitted", datetime.utcnow().strftime("%Y-%m-%d")),
                    1,
                ),
            )

    conn.close()
    print(f"  DB initialised: {DB_PATH}")


# ── Save result ────────────────────────────────────────────────────────────────

def save_result(
    hf_id:      str,
    label:      str,
    overall:    float,
    per_task:   dict[str, float],
    params:     str = "Unknown",
    model_type: str = "Open",
    n_items:    int = 150,
    notes:      str = "",
) -> int:
    """Save a new evaluation result to the database.

    Args:
        hf_id:      HuggingFace model ID.
        label:      Display name for the model.
        overall:    Overall accuracy (0–1).
        per_task:   Dict of task_short -> accuracy (0–1).
        params:     Parameter count string (e.g. "7B").
        model_type: "Open" or "API".
        n_items:    Number of items evaluated.
        notes:      Optional notes.

    Returns:
        Row id of the inserted record.
    """
    conn = _connect()
    with conn:
        cursor = conn.execute(
            """INSERT INTO results
               (model_id, label, hf_id, params, model_type,
                overall, score_REG, score_NUM, score_CON, score_TMP,
                n_items, submitted_at, is_baseline, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?)""",
            (
                hf_id.split("/")[-1], label, hf_id,
                params, model_type, overall,
                per_task.get("REG", 0.0), per_task.get("NUM", 0.0),
                per_task.get("CON", 0.0), per_task.get("TMP", 0.0),
                n_items,
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                notes,
            ),
        )
        row_id = cursor.lastrowid
    conn.close()
    return row_id


# ── Leaderboard retrieval ──────────────────────────────────────────────────────

def get_leaderboard(include_duplicates: bool = False) -> pd.DataFrame:
    """Retrieve the leaderboard as a pandas DataFrame.

    Args:
        include_duplicates: If False (default), keep only the best submission
                            per hf_id.

    Returns:
        DataFrame sorted by overall accuracy descending, with columns:
        Rank, Model, HF ID, Params, Type, Overall, REG, NUM, CON, TMP, Submitted.
    """
    conn  = _connect()
    query = "SELECT * FROM results ORDER BY overall DESC, submitted_at ASC"
    df    = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return df

    if not include_duplicates:
        df = df.sort_values("overall", ascending=False).drop_duplicates(
            subset="hf_id", keep="first"
        )

    df = df.sort_values("overall", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    display_cols = {
        "label":      "Model",
        "hf_id":      "HF Model ID",
        "params":     "Params",
        "model_type": "Type",
        "overall":    "Overall (%)",
        "score_REG":  "REG (%)",
        "score_NUM":  "NUM (%)",
        "score_CON":  "CON (%)",
        "score_TMP":  "TMP (%)",
        "submitted_at": "Submitted",
    }
    df = df.rename(columns=display_cols)

    # Convert 0–1 floats to percentages
    pct_cols = ["Overall (%)", "REG (%)", "NUM (%)", "CON (%)", "TMP (%)"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = (df[col] * 100).round(1)

    out_cols = ["Rank", "Model", "HF Model ID", "Params", "Type",
                "Overall (%)", "REG (%)", "NUM (%)", "CON (%)", "TMP (%)", "Submitted"]
    return df[[c for c in out_cols if c in df.columns]]
