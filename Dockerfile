# syntax=docker/dockerfile:1

# ---- build -------------------------------------------------------------------
FROM rust:1.99-slim-bookworm AS build
# clang/libclang + g++ build the bundled RocksDB engine.
RUN apt-get update \
    && apt-get install -y --no-install-recommends clang libclang-dev g++ make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock rustfmt.toml ./
COPY crates ./crates
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p quiver-server --features rocksdb \
    && cargo build --release --locked -p quiver-tui \
    && cp target/release/quiver target/release/quiver-tui /usr/local/bin/

# ---- runtime -----------------------------------------------------------------
FROM debian:bookworm-slim
RUN useradd --system --uid 10001 --home-dir /data quiver \
    && mkdir -p /data && chown quiver /data
COPY --from=build /usr/local/bin/quiver /usr/local/bin/quiver-tui /usr/local/bin/
USER quiver
ENV QUIVER_HOST=0.0.0.0 \
    QUIVER_HTTP_PORT=6900 \
    QUIVER_GRPC_PORT=6901 \
    QUIVER_DATA_DIR=/data
VOLUME ["/data"]
# 6900 = REST, 6901 = gRPC
EXPOSE 6900 6901
HEALTHCHECK --interval=30s --timeout=3s CMD ["/bin/sh", "-c", "exec 3<>/dev/tcp/127.0.0.1/6900"]
ENTRYPOINT ["/usr/local/bin/quiver"]
