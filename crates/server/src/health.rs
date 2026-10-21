//! `quiver --healthcheck`: probes a running server, for container health checks.

use std::net::SocketAddr;
use std::time::Duration;

use std::sync::Arc;

use rustls::DigitallySignedStruct;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::config::Config;

const TIMEOUT: Duration = Duration::from_secs(3);

/// Whether the server `config` describes is serving. Asks the REST API for
/// `/health` (over TLS when the server uses it); with REST disabled, or when
/// clients need a certificate, checks that the port accepts connections.
pub async fn check(config: &Config) -> Result<(), String> {
    let probe = async {
        if !config.disable_http && config.tls_cert.is_some() && config.tls_client_ca.is_none() {
            https_health(config.probe_addr(config.http_port)).await
        } else if config.disable_http || config.tls_cert.is_some() {
            let port = if config.disable_http {
                config.grpc_port
            } else {
                config.http_port
            };
            let addr = config.probe_addr(port);
            TcpStream::connect(addr)
                .await
                .map(drop)
                .map_err(|e| format!("port {addr}: {e}"))
        } else {
            http_health(config.probe_addr(config.http_port)).await
        }
    };
    tokio::time::timeout(TIMEOUT, probe)
        .await
        .map_err(|_| format!("no answer within {}s", TIMEOUT.as_secs()))?
}

async fn http_health(addr: SocketAddr) -> Result<(), String> {
    let stream = TcpStream::connect(addr)
        .await
        .map_err(|e| format!("{addr}: {e}"))?;
    get_health(addr, stream).await
}

/// Like [`http_health`] over TLS. The probe runs next to the server and
/// checks that it answers, not who it is, so the certificate is not verified.
async fn https_health(addr: SocketAddr) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("{addr}: {e}");
    let config =
        rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .map_err(|e| e.to_string())?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AnyCertificate))
            .with_no_client_auth();
    let tcp = TcpStream::connect(addr).await.map_err(fail)?;
    let name = ServerName::IpAddress(addr.ip().into());
    let stream = tokio_rustls::TlsConnector::from(Arc::new(config))
        .connect(name, tcp)
        .await
        .map_err(fail)?;
    get_health(addr, stream).await
}

async fn get_health(addr: SocketAddr, mut stream: impl AsyncRead + AsyncWrite + Unpin) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("{addr}: {e}");
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

/// Accepts any server certificate, for the local health probe only.
#[derive(Debug)]
struct AnyCertificate;

impl ServerCertVerifier for AnyCertificate {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
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
        // Another test sets `QUIVER_*` variables that would break parsing.
        let _env = crate::config::tests::env_lock();
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

    #[tokio::test]
    async fn probes_https() {
        let dir = tempfile::tempdir().unwrap();
        let key = rcgen::KeyPair::generate().unwrap();
        let cert = rcgen::CertificateParams::new(vec!["quiver.example".into()])
            .unwrap()
            .self_signed(&key)
            .unwrap();
        let (cert_path, key_path) = (dir.path().join("cert.pem"), dir.path().join("key.pem"));
        std::fs::write(&cert_path, cert.pem()).unwrap();
        std::fs::write(&key_path, key.serialize_pem()).unwrap();

        let tls =
            quiver_http::tls_config(cert.pem().as_bytes(), key.serialize_pem().as_bytes(), None).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port().to_string();
        let app = quiver_http::router(
            Arc::new(quiver_db::Database::in_memory()),
            quiver_http::HttpOptions::default(),
        );
        tokio::spawn(quiver_http::serve_tls(listener, tls, app, std::future::pending()));

        let (cert_path, key_path) = (cert_path.to_str().unwrap(), key_path.to_str().unwrap());
        let tls_args = ["--tls-cert", cert_path, "--tls-key", key_path];
        let with_tls = config(&[&["--http-port", &port][..], &tls_args].concat());
        assert_eq!(check(&with_tls).await, Ok(()));
        // Speaking plain HTTP to a TLS port fails.
        assert!(check(&config(&["--http-port", &port])).await.is_err());
    }
}
