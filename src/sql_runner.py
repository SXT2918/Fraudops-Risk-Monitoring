"""Run FraudOps SQL analytics files against the local SQLite database.

Examples:
    python -m src.sql_runner --list
    python -m src.sql_runner --file sql/01_fraud_kpis.sql
    python -m src.sql_runner --all
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import BASE_DIR, DATABASE_PATH, SQL_DIR


class SQLRunnerError(RuntimeError):
    """Raised when SQL files or the SQLite database cannot be used."""


@dataclass(frozen=True)
class SQLQueryResult:
    """Result from one SQL query inside a SQL analytics file."""

    file_path: Path
    statement_index: int
    title: str
    dataframe: pd.DataFrame


def list_sql_files(sql_dir: Path = SQL_DIR) -> list[Path]:
    """Return available SQL analytics files in run order."""
    sql_dir = Path(sql_dir)
    if not sql_dir.exists():
        return []
    return sorted(sql_dir.glob("*.sql"))


def resolve_sql_file(file_path: Path) -> Path:
    """Resolve a user-provided SQL file path from cwd or project root."""
    file_path = Path(file_path)
    if file_path.is_absolute():
        return file_path

    cwd_path = Path.cwd() / file_path
    if cwd_path.exists():
        return cwd_path
    return BASE_DIR / file_path


def split_sql_statements(sql_text: str) -> list[str]:
    """Split SQL text into complete executable statements.

    SQLite only runs one statement at a time through pandas.read_sql_query.
    This splitter keeps comments attached to the statement that follows them so
    labels like "-- Result: ..." can be used for terminal output.
    """
    statements: list[str] = []
    pending_lines: list[str] = []

    for line in sql_text.splitlines():
        pending_lines.append(line)
        candidate = "\n".join(pending_lines).strip()
        if sqlite3.complete_statement(candidate):
            if _statement_has_sql(candidate):
                statements.append(candidate)
            pending_lines = []

    remainder = "\n".join(pending_lines).strip()
    if remainder and _statement_has_sql(remainder):
        statements.append(remainder)

    return statements


def execute_sql_file(
    file_path: Path,
    db_path: Path = DATABASE_PATH,
) -> list[SQLQueryResult]:
    """Execute one SQL analytics file and return query result tables."""
    resolved_file = resolve_sql_file(file_path)
    if not resolved_file.exists():
        raise SQLRunnerError(
            f"Could not find SQL file: {resolved_file}. "
            "Run `python -m src.sql_runner --list` to see available files."
        )

    db_path = Path(db_path)
    if not db_path.exists():
        raise SQLRunnerError(
            f"Could not find SQLite database at {db_path}. "
            "Run `python -m src.ingestion` first to create it, then try again."
        )

    statements = split_sql_statements(resolved_file.read_text(encoding="utf-8"))
    if not statements:
        raise SQLRunnerError(f"No executable SQL statements found in {resolved_file}.")

    results: list[SQLQueryResult] = []
    with sqlite3.connect(db_path) as conn:
        for index, statement in enumerate(statements, start=1):
            query_text = _strip_sql_comments(statement).rstrip(";").strip()
            if not _is_read_query(query_text):
                conn.execute(query_text)
                conn.commit()
                continue

            try:
                dataframe = pd.read_sql_query(statement, conn)
            except Exception as exc:
                raise SQLRunnerError(
                    f"Could not run query {index} in {resolved_file.name}. "
                    f"SQLite said: {exc}"
                ) from exc

            results.append(
                SQLQueryResult(
                    file_path=resolved_file,
                    statement_index=index,
                    title=_extract_statement_title(statement, index),
                    dataframe=dataframe,
                )
            )

    return results


def execute_all_sql_files(
    sql_dir: Path = SQL_DIR,
    db_path: Path = DATABASE_PATH,
) -> list[SQLQueryResult]:
    """Execute all SQL analytics files in sorted file-name order."""
    sql_files = list_sql_files(sql_dir)
    if not sql_files:
        raise SQLRunnerError(f"No SQL files found in {Path(sql_dir)}.")

    all_results: list[SQLQueryResult] = []
    for sql_file in sql_files:
        all_results.extend(execute_sql_file(sql_file, db_path=db_path))
    return all_results


def print_available_files(sql_files: Iterable[Path]) -> None:
    """Print available SQL files for the CLI."""
    sql_files = list(sql_files)
    if not sql_files:
        print("No SQL files found.")
        return

    print("Available SQL files:")
    for sql_file in sql_files:
        print(f"- {_display_path(sql_file)}")


def print_query_results(results: Iterable[SQLQueryResult]) -> None:
    """Print SQL result tables in a readable terminal format."""
    current_file: Path | None = None

    for result in results:
        if current_file != result.file_path:
            current_file = result.file_path
            print("")
            print(f"=== {_display_path(result.file_path)} ===")

        print("")
        print(f"Query {result.statement_index}: {result.title}")
        if result.dataframe.empty:
            print("(no rows)")
        else:
            print(result.dataframe.to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse SQL runner command-line arguments."""
    parser = argparse.ArgumentParser(description="Run FraudOps SQL analytics files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List available SQL files")
    group.add_argument("--file", type=Path, help="Run one SQL file")
    group.add_argument("--all", action="store_true", help="Run all SQL files in order")
    return parser.parse_args()


def main() -> None:
    """Run the CLI."""
    args = parse_args()

    try:
        if args.list:
            print_available_files(list_sql_files())
            return
        if args.file:
            print_query_results(execute_sql_file(args.file))
            return
        if args.all:
            print_query_results(execute_all_sql_files())
            return
    except SQLRunnerError as exc:
        raise SystemExit(str(exc)) from exc


def _statement_has_sql(statement: str) -> bool:
    return bool(_strip_sql_comments(statement).rstrip(";").strip())


def _strip_sql_comments(statement: str) -> str:
    lines: list[str] = []
    for line in statement.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        if "--" in line:
            line = line.split("--", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _is_read_query(query_text: str) -> bool:
    first_word = query_text.lstrip().split(maxsplit=1)[0].lower()
    return first_word in {"select", "with", "pragma"}


def _extract_statement_title(statement: str, index: int) -> str:
    for line in statement.splitlines():
        match = re.match(r"\s*--\s*(?:Result|Query):\s*(.+)", line)
        if match:
            return match.group(1).strip().rstrip(".")
    return f"Statement {index}"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
