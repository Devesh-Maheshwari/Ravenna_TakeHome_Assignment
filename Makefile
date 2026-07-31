.PHONY: help bootstrap install up down init seed start run ui test lint fmt reset check-venv

# Call the venv's binaries by path rather than relying on an activated shell.
# A bare `uvicorn` resolves to whatever is first on PATH — on a machine with
# anaconda installed that is anaconda's copy, which cannot see this project and
# fails with a confusing `ModuleNotFoundError: No module named 'support_agent'`.
VENV := .venv
BIN  := $(VENV)/bin

help:
	@echo "make install - create .venv, .env, and install dependencies"
	@echo "make up      - start Postgres (needs Docker Desktop running)"
	@echo "make init    - apply db/schema.sql (idempotent)"
	@echo "make seed    - load the 25 KB articles + 15 customers"
	@echo "make start   - from a fresh clone, install and run DB + API + UI"
	@echo "make run     - run the API on http://localhost:8000"
	@echo "make ui      - run the Streamlit UI on http://localhost:8501"
	@echo "make test    - run the test suite (skips tests needing a live LLM)"
	@echo "make lint    - ruff check"
	@echo "make reset   - drop the database volume and rebuild from scratch"
	@echo ""
	@echo "No shell activation needed — every target uses $(BIN) directly."

# Fail with an actionable message instead of letting a stray PATH binary produce
# an unrelated-looking import error.
check-venv:
	@test -x $(BIN)/python || { \
		echo "No virtualenv at ./$(VENV)."; \
		echo "Run: make install"; \
		exit 1; }

bootstrap:
	@command -v uv >/dev/null 2>&1 || { \
		echo "The 'uv' command is required. Install it from https://docs.astral.sh/uv/"; \
		exit 1; \
	}
	@test -f .env || { cp .env.example .env; echo "Created .env from .env.example"; }
	@test -x $(BIN)/python || uv venv
	uv sync --extra dev

install: bootstrap
	@echo "Done. No need to activate — make targets use $(BIN) directly."

up:
	docker compose up -d
	@docker compose exec -T db sh -c 'until pg_isready -U support -d support_agent >/dev/null 2>&1; do sleep 1; done'
	@echo "Postgres healthy on localhost:5434"

down:
	docker compose down

# Applies the schema directly through psql in the container, so it works whether
# or not scripts/init_db.py has been implemented yet.
init:
	docker compose exec -T db psql -U support -d support_agent -v ON_ERROR_STOP=1 -q \
		< src/support_agent/db/schema.sql
	@echo "Schema applied."

seed: check-venv
	$(BIN)/python scripts/seed_db.py

start: bootstrap
	$(BIN)/python scripts/start_all.py

run: check-venv
	$(BIN)/uvicorn support_agent.main:create_app --factory --reload \
		--host 127.0.0.1 --port 8000

ui: check-venv
	$(BIN)/streamlit run ui/streamlit_app.py --server.address 127.0.0.1

test: check-venv
	$(BIN)/pytest -m "not llm"

lint: check-venv
	$(BIN)/ruff check .

fmt: check-venv
	$(BIN)/ruff format .

# Verifies the parts that work while the application is still stubbed.
verify: check-venv
	$(BIN)/python -m compileall -q src tests scripts ui && echo "compileall  OK"
	$(BIN)/ruff check . -q && echo "ruff        OK"
	@$(BIN)/pytest --collect-only -q 2>&1 | tail -1
	@docker compose exec -T db psql -U support -d support_agent -X -q -c \
		"SELECT (SELECT count(*) FROM customers) customers, (SELECT count(*) FROM kb_articles) kb_articles;"

reset: down
	docker volume rm ravenna_pgdata || true
	$(MAKE) up init seed
