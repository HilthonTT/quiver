# Contributing to Quiver

Thanks for your interest! Bug reports, docs fixes, benchmarks and code are all welcome.

## Getting started

You'll need:

- Rust **1.89+** (`rustup update stable`)
- Optional: clang/libclang to build the RocksDB engine (`make test-rocksdb`)
- [uv](https://docs.astral.sh/uv/) for the Python client
- `make` (optional; the targets just wrap `cargo` and `uv`)

```bash
git clone https://github.com/HilthonTT/quiver && cd quiver
make check          # everything CI runs
```

## Workflow

1. For anything non-trivial, open an issue first so we can agree on the approach.
2. Branch from `main`, keep PRs focused, and write tests for behaviour changes.
3. Run `make check` before pushing.
4. Add a line under **Unreleased** in [`CHANGELOG.md`](CHANGELOG.md) for user-facing changes.
5. Use [Conventional Commits](https://www.conventionalcommits.org/) in titles where you can (`feat:`, `fix:`, `docs:`, `perf:`, `refactor:`, `test:`, `chore:`).

## Where things live

The workspace is layered. A crate may only depend on crates *below* it:

```
types → storage → index → snapshot → quiver → http / grpc → server
                                      types → client → tui
```

- A new metric goes in `crates/types/src/distance.rs`, and its k-d tree bound in `crates/index/src/kdtree.rs`.
- A new index type goes in a new module in `crates/index`, as an `Index` enum variant plus an `IndexConfig` variant in `types`.
- A new storage engine implements `StorageEngine` in `crates/storage` and must pass the shared `durable_engine_contract` test.
- A new endpoint needs a handler in `crates/http`, an RPC in `crates/grpc/proto/quiver.proto` and `service.rs`, a client method in `crates/client`, a Python client method, and docs in `docs/API.md`.

## Code guidelines

**Rust**

- `cargo fmt` and `cargo clippy -- -D warnings` must be clean.
- No `unsafe` (`#![forbid(unsafe_code)]` is enforced workspace-wide).
- Public items need doc comments (`#![warn(missing_docs)]`).
- Return `quiver_db::Error`. Don't panic on bad input, only on broken internal invariants.
- Performance changes need numbers: run `make bench` before and after and paste both into the PR.

**Python**

- `ruff check`, `ruff format` and `mypy --strict` must pass.
- `_client.py` and `_async_client.py` mirror each other. Request building and parsing live in `_protocol.py`, so a new endpoint normally needs one function there plus a thin method on each client.

## On-disk format

Snapshot and WAL formats are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). If you change them, bump `VERSION` in `snapshot.rs`, keep reading older versions where feasible, and call it out in the changelog.

## Running the integration tests

```bash
cargo run --release -p quiver-server -- --in-memory &
cd clients/python && QUIVER_URL=http://localhost:6900 uv run pytest
```

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating you agree to uphold it.
