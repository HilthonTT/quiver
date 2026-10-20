.PHONY: help build build-rocksdb test test-rocksdb lint fmt bench run tui docker py-test py-lint check

PY := cd clients/python && uv run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Build the server and TUI in release mode
	cargo build --release -p quiver-server -p quiver-tui

build-rocksdb: ## Build the server with the RocksDB engine (needs clang)
	cargo build --release -p quiver-server --features rocksdb

test: ## Run all Rust tests
	cargo test --workspace --all-targets
	cargo test --workspace --doc

test-rocksdb: ## Run the storage tests against RocksDB too
	cargo test -p quiver-storage -p quiver-db -p quiver-server --features rocksdb

lint: ## Run rustfmt and clippy checks
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings

fmt: ## Format Rust and Python code
	cargo fmt --all
	$(PY) ruff format .

bench: ## Run the recall/throughput benchmark
	cargo run --release --example benchmark -p quiver-db -- 100000 128 1000

run: ## Run the server (REST :6900, gRPC :6901) with a local data directory
	cargo run --release -p quiver-server -- --data-dir ./quiver-data

tui: ## Open the terminal UI against a local server
	cargo run --release -p quiver-tui

docker: ## Build the Docker image
	docker build -t quiver:latest .

py-test: ## Run Python client unit tests
	$(PY) pytest

py-lint: ## Lint and type-check the Python client
	$(PY) ruff check . && $(PY) ruff format --check . && $(PY) mypy src tests

check: lint test py-lint py-test ## Run everything CI runs
