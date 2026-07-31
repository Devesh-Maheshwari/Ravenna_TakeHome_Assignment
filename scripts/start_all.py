"""Prepare dependencies and run the API and Streamlit UI together.

This is the implementation behind ``make start``. It intentionally uses only
the standard library so it can explain a missing virtualenv before importing any
project dependency.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / ".venv" / "bin"
API_PORT = int(os.getenv("SUPPORT_AGENT_API_PORT", "8000"))
UI_PORT = int(os.getenv("SUPPORT_AGENT_UI_PORT", "8501"))
API_URL = f"http://localhost:{API_PORT}"
UI_URL = f"http://localhost:{UI_PORT}"


def log(message: str = "") -> None:
    """Print startup information immediately, even when output is redirected."""
    print(message, flush=True)


def run_step(label: str, command: list[str]) -> None:
    """Run one required preparation step and stop on failure."""
    log(f"\n[startup] {label}")
    subprocess.run(command, cwd=ROOT, check=True)


def url_is_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=1) as response:  # noqa: S310 - localhost only
            return 200 <= response.status < 300
    except (OSError, URLError):
        return False


def port_is_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for(url: str, process: subprocess.Popen | None, label: str, timeout: int = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if url_is_ready(url):
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{label} stopped during startup (exit {process.returncode}).")
        time.sleep(0.25)
    raise RuntimeError(f"{label} did not become ready at {url} within {timeout} seconds.")


def start_or_reuse(
    *,
    label: str,
    port: int,
    health_url: str,
    compatibility_url: str | None = None,
    command: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.Popen | None:
    """Start a service and fail clearly rather than silently reuse stale code."""
    if port_is_open(port):
        if url_is_ready(health_url) and (
            compatibility_url is None or url_is_ready(compatibility_url)
        ):
            raise RuntimeError(
                f"{label} is already running on port {port}. Refusing to reuse it "
                "because it may contain stale code. Stop the earlier `make start` "
                "with Ctrl+C, then run `make start` again."
            )
        raise RuntimeError(
            f"Port {port} is already in use, but it is not a compatible {label}. "
            "Stop the older process with Ctrl+C, then run `make start` again."
        )
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )
    wait_for(health_url, process, label)
    log(f"[startup] {label} is ready.")
    return process


def stop_process(process: subprocess.Popen | None, label: str) -> None:
    if process is None or process.poll() is not None:
        return
    log(f"[shutdown] Stopping {label}...")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def print_guide(api_reused: bool, ui_reused: bool) -> None:
    key_value = os.getenv("OPENAI_API_KEY", "").strip()
    key_configured = bool(key_value and key_value != "sk-replace-me")
    env_path = ROOT / ".env"
    if not key_configured and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                value = line.partition("=")[2].strip()
                key_configured = bool(value and value != "sk-replace-me")
                break
    llm_mode = (
        "OpenAI key configured; provider calls are enabled."
        if key_configured
        else "No usable OpenAI key detected; safe local fallback responses are enabled."
    )
    api_ownership = "reused" if api_reused else "started"
    ui_ownership = "reused" if ui_reused else "started"

    log(
        f"""

======================================================================
 Ravenna Support is ready
======================================================================

 Customer chat UI
   {UI_URL}
   Use this to create, switch, resume, and close customer conversations.
   Tool calls and the agent reasoning trace appear below each response.

 FastAPI Swagger documentation
   {API_URL}/docs
   Interactive API reference. You can expand endpoints and execute requests.

 FastAPI ReDoc documentation
   {API_URL}/redoc
   Read-only API reference with request and response schemas.

 Backend service
   {API_URL}
   Health:  {API_URL}/health
   Metrics: {API_URL}/metrics

 Database
   PostgreSQL is running through Docker on localhost:5434.
   The schema and local customer/knowledge-base seed data are ready.

 LLM mode
   {llm_mode}

 Logs
   FastAPI and Streamlit logs continue below.
   Press Ctrl+C once to stop processes started by this command.

 Service ownership: API={api_ownership}, UI={ui_ownership}
======================================================================
"""
    )


def main() -> int:
    api_process: subprocess.Popen | None = None
    ui_process: subprocess.Popen | None = None
    try:
        run_step("Starting PostgreSQL with Docker", ["docker", "compose", "up", "-d"])
        run_step(
            "Waiting for PostgreSQL",
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "db",
                "sh",
                "-c",
                "until pg_isready -U support -d support_agent; do sleep 1; done",
            ],
        )
        run_step("Applying the database schema", [str(BIN / "python"), "scripts/init_db.py"])
        run_step(
            "Loading local demo customers and knowledge-base articles",
            [str(BIN / "python"), "scripts/seed_db.py"],
        )

        api_process = start_or_reuse(
            label="FastAPI backend",
            port=API_PORT,
            health_url=f"{API_URL}/health",
            compatibility_url=f"{API_URL}/demo/customers",
            command=[
                str(BIN / "uvicorn"),
                "support_agent.main:create_app",
                "--factory",
                "--reload",
                "--host",
                "127.0.0.1",
                "--port",
                str(API_PORT),
            ],
        )
        ui_env = os.environ.copy()
        ui_env["SUPPORT_AGENT_API_URL"] = API_URL
        ui_process = start_or_reuse(
            label="Streamlit UI",
            port=UI_PORT,
            health_url=f"{UI_URL}/_stcore/health",
            command=[
                str(BIN / "streamlit"),
                "run",
                "ui/streamlit_app.py",
                "--server.headless",
                "true",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(UI_PORT),
            ],
            env=ui_env,
        )
        print_guide(api_process is None, ui_process is None)

        while True:
            if api_process is not None and api_process.poll() is not None:
                raise RuntimeError(f"FastAPI stopped unexpectedly (exit {api_process.returncode}).")
            if ui_process is not None and ui_process.poll() is not None:
                raise RuntimeError(
                    f"Streamlit stopped unexpectedly (exit {ui_process.returncode})."
                )
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("\n[startup] Ctrl+C received.")
        return 0
    except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError) as exc:
        log(f"\n[startup] ERROR: {exc}")
        return 1
    finally:
        stop_process(ui_process, "Streamlit UI")
        stop_process(api_process, "FastAPI backend")


if __name__ == "__main__":
    raise SystemExit(main())
