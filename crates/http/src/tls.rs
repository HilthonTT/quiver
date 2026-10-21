use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use rustls::RootCertStore;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::server::WebPkiClientVerifier;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::server::TlsStream;

/// A handshake that has not finished by then is dropped, so idle
/// connections cannot pile up before they are ever served.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Connections whose handshake is done but that `axum` has not taken yet.
const READY_QUEUE: usize = 128;

/// Builds a server configuration from a PEM certificate chain and private
/// key. With `client_ca`, clients must present a certificate signed by one of
/// its certificates (mutual TLS).
pub fn server_config(
    cert_pem: &[u8],
    key_pem: &[u8],
    client_ca_pem: Option<&[u8]>,
) -> io::Result<Arc<rustls::ServerConfig>> {
    let invalid = |what: &str, e: &dyn std::fmt::Display| {
        io::Error::new(io::ErrorKind::InvalidData, format!("{what}: {e}"))
    };
    let certs = CertificateDer::pem_slice_iter(cert_pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| invalid("TLS certificate", &e))?;
    if certs.is_empty() {
        return Err(invalid("TLS certificate", &"no certificate found"));
    }
    let key = PrivateKeyDer::from_pem_slice(key_pem).map_err(|e| invalid("TLS private key", &e))?;

    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let builder = rustls::ServerConfig::builder_with_provider(Arc::clone(&provider))
        .with_safe_default_protocol_versions()
        .map_err(|e| invalid("TLS", &e))?;
    let builder = match client_ca_pem {
        None => builder.with_no_client_auth(),
        Some(pem) => {
            let mut roots = RootCertStore::empty();
            for cert in CertificateDer::pem_slice_iter(pem) {
                roots
                    .add(cert.map_err(|e| invalid("TLS client CA", &e))?)
                    .map_err(|e| invalid("TLS client CA", &e))?;
            }
            if roots.is_empty() {
                return Err(invalid("TLS client CA", &"no certificate found"));
            }
            let verifier = WebPkiClientVerifier::builder_with_provider(Arc::new(roots), provider)
                .build()
                .map_err(|e| invalid("TLS client CA", &e))?;
            builder.with_client_cert_verifier(verifier)
        }
    };
    let mut config = builder
        .with_single_cert(certs, key)
        .map_err(|e| invalid("TLS certificate and key", &e))?;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(Arc::new(config))
}

/// A listener that hands `axum::serve` connections whose TLS handshake has
/// completed. Handshakes run concurrently in their own tasks, so a slow or
/// silent client never holds up anyone else's.
#[derive(Debug)]
pub struct TlsListener {
    ready: mpsc::Receiver<(TlsStream<TcpStream>, SocketAddr)>,
    local_addr: SocketAddr,
    acceptor: tokio::task::JoinHandle<()>,
}

impl TlsListener {
    /// Starts accepting TLS connections on `listener`.
    pub fn new(listener: TcpListener, config: Arc<rustls::ServerConfig>) -> io::Result<Self> {
        let local_addr = listener.local_addr()?;
        let (tx, ready) = mpsc::channel(READY_QUEUE);
        let tls = TlsAcceptor::from(config);
        let acceptor = tokio::spawn(async move {
            loop {
                let (tcp, addr) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(e) => {
                        // Typically out of file descriptors; back off instead of spinning.
                        tracing::warn!(error = %e, "accepting a connection failed");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        continue;
                    }
                };
                if tx.is_closed() {
                    return;
                }
                let (tls, tx) = (tls.clone(), tx.clone());
                tokio::spawn(async move {
                    match tokio::time::timeout(HANDSHAKE_TIMEOUT, tls.accept(tcp)).await {
                        Ok(Ok(stream)) => {
                            let _ = tx.send((stream, addr)).await;
                        }
                        Ok(Err(e)) => tracing::debug!(%addr, error = %e, "TLS handshake failed"),
                        Err(_) => tracing::debug!(%addr, "TLS handshake timed out"),
                    }
                });
            }
        });
        Ok(Self {
            ready,
            local_addr,
            acceptor,
        })
    }
}

impl Drop for TlsListener {
    fn drop(&mut self) {
        self.acceptor.abort();
    }
}

impl axum::serve::Listener for TlsListener {
    type Io = TlsStream<TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.ready.recv().await {
            Some(conn) => conn,
            // The accept task only stops once this listener is dropped.
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        Ok(self.local_addr)
    }
}

/// Serves `router` over HTTPS on `listener` until `shutdown` resolves.
pub async fn serve_tls(
    listener: TcpListener,
    config: Arc<rustls::ServerConfig>,
    router: Router,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> io::Result<()> {
    axum::serve(TlsListener::new(listener, config)?, router)
        .with_graceful_shutdown(shutdown)
        .await
}
