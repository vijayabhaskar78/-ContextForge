"""
Lightweight SQLite migration runner with schema version tracking.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .config import DB_PATH, console
from .observability import get_logger

logger = get_logger(__name__)


MigrationRunner = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True)
class SqliteMigration:
    version: int
    name: str
    statements: tuple[str, ...] = ()
    runner: MigrationRunner | None = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def apply_sqlite_migrations(
    conn: sqlite3.Connection,
    *,
    component: str,
    migrations: list[SqliteMigration],
):
    """Applies ordered migrations for a component and records applied versions."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            component TEXT NOT NULL,
            version INTEGER NOT NULL,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL,
            PRIMARY KEY(component, version)
        )
        """
    )

    rows = conn.execute(
        "SELECT version FROM schema_migrations WHERE component = ?",
        (component,),
    ).fetchall()
    applied_versions = {int(row[0]) for row in rows}

    for migration in sorted(migrations, key=lambda m: int(m.version)):
        version = int(migration.version)
        if version in applied_versions:
            continue

        if migration.statements:
            for statement in migration.statements:
                sql = str(statement or "").strip()
                if not sql:
                    continue
                conn.executescript(f"{sql.rstrip(';')};")
        if callable(migration.runner):
            migration.runner(conn)

        conn.execute(
            """
            INSERT INTO schema_migrations (component, version, name, applied_at)
            VALUES (?, ?, ?, ?)
            """,
            (component, version, migration.name, _utcnow_iso()),
        )
        logger.info(
            "db_migration_applied",
            component=component,
            version=version,
            name=migration.name,
        )
