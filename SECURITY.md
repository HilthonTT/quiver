# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Report them privately through
[GitHub Security Advisories](https://github.com/HilthonTT/vector-db/security/advisories/new).

Include what you found, how to reproduce it, the affected version (`quiver --version` or the commit),
and the impact you expect. You should get an acknowledgement within a few days; once a fix is ready
it will be released and the advisory published, crediting you unless you prefer otherwise.

## Supported versions

Quiver is pre-1.0. Only the latest release and `main` receive security fixes.

## Deploying safely

- **Set API keys before exposing the server.** With no keys configured, authentication is
  disabled and anyone who can reach the ports can read and write. The server logs a warning
  when it listens on a non-loopback address without keys. Use `--keys-file`, `--api-key` or
  `--readonly-api-key`, and give read-only keys to anything that only queries.
- **Terminate TLS in front of Quiver.** The REST (`6900`) and gRPC (`6901`) APIs speak plain
  HTTP; put a reverse proxy or load balancer with TLS in front of them, or keep them on a
  private network. The default bind address is `127.0.0.1`; the Docker image binds `0.0.0.0`.
- **Protect the keys file and data directory.** Keys are stored in plain text in the keys file;
  restrict its permissions. The data and snapshot directories hold all stored vectors and metadata.
- **Leave `--cors` off** unless a browser application on another origin needs direct access.
- **Size limits:** JSON bodies are capped at 64 MiB; lower it with `--max-body-mb` if your
  workload allows. Snapshot uploads are capped at 16 GiB and need a read-write key.

## What is in scope

The server (REST and gRPC APIs, authentication), the storage engines, snapshot archives,
and the Rust and Python clients. API keys are compared in constant time. Snapshot archives are
read in memory rather than extracted to disk, their data is checked against its SHA-256 before
use, and snapshot ids must be UUIDs, so they cannot be used as file paths.
