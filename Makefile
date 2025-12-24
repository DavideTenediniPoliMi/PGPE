export SRC_DIR=src

format:
	uv run ruff format $(SRC_DIR)

lint:
	uv run ruff check --fix $(SRC_DIR)
	uv run pyrefly check $(SRC_DIR)

test:
	uv run pytest tests/