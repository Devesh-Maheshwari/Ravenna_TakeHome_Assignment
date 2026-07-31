"""Apply `db/schema.sql`. Idempotent — safe to run on every boot.

This is the whole migration story, and that is a deliberate choice rather than an
omission: the schema is authored once and never evolves within this project, so a
migration tool would contribute one revision file and a fiddly async `env.py`.
See docs/DESIGN_DECISIONS.md ("Alembic") for what changes that calculus.

Run with `make init`.
"""

from pathlib import Path

from psycopg import connect

from support_agent.config import get_settings


def main() -> int:
    """Read the schema and execute it. Returns a process exit code.

    Every statement in `schema.sql` is `IF NOT EXISTS`-guarded, so a second run
    is a no-op rather than an error — which is what makes it safe to wire into
    startup.
    """
    schema_path = Path(__file__).resolve().parents[1] / "src/support_agent/db/schema.sql"
    with connect(get_settings().database_url, autocommit=True) as connection:
        connection.execute(schema_path.read_text())
    print("Schema applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
