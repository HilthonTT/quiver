//! `quiver --healthcheck`: probes a running server, for container health checks.

use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::config::Config;

const TIMEOUT: Duration = Duration::from_secs(3);

/// Whether the server `config` describes is serving. Asks the REST API for
/// `/health`; with REST disabled, checks that the gRPC port accepts connections.
pub async fn check(config: &Config) -> Result<(), String> {
    let probe = async {
        if config.disable_http {
            let addr = config.probe_addr(config.grpc_port);
            TcpStream::connect(addr)
                .await
                .map(drop)
                .map_err(|e| format!("gRPC port {addr}: {e}"))
        } else {
            http_health(config.probe_addr(config.http_port)).await
        }
    };
    tokio::time::timeout(TIMEOUT, probe)
        .await
        .map_err(|_| format!("no answer within {}s", TIMEOUT.as_secs()))?
}

async fn http_health(addr: SocketAddr) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("{addr}: {e}");
    let mut stream = TcpStream::connect(addr).await.map_err(fail)?;
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .map_err(fail)?;
    // The status line is all we need.
    let mut head = [0u8; 12];
    stream.read_exact(&mut head).await.map_err(fail)?;
    match &head {
        b"HTTP/1.1 200" | b"HTTP/1.0 200" => Ok(()),
        _ => Err(format!(
            "{addr}: unexpected response `{}`",
            String::from_utf8_lossy(&head)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    async fn serve_once(response: &'static [u8]) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf).await;
            socket.write_all(response).await.unwrap();
        });
        port
    }

    fn config(args: &[&str]) -> Config {
        Config::try_parse_from(std::iter::once("quiver").chain(args.iter().copied())).unwrap()
    }

    #[tokio::test]
    async fn healthy_and_unhealthy_servers() {
        let port = serve_once(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n").await;
        let c = config(&["--host", "0.0.0.0", "--http-port", &port.to_string()]);
        assert_eq!(check(&c).await, Ok(()));

        let port = serve_once(b"HTTP/1.1 503 Service Unavailable\r\n\r\n").await;
        assert!(check(&config(&["--http-port", &port.to_string()])).await.is_err());

        // Nothing listening.
        let port = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        assert!(check(&config(&["--http-port", &port.to_string()])).await.is_err());
    }
}
