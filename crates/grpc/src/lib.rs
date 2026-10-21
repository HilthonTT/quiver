mod auth;
mod convert;
mod metrics;
mod service;
use std::sync::Arc;

use quiver_db::Database;
use quiver_db::auth::KeyStore;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Certificate, Identity, ServerTlsConfig};

pub use auth::Authenticator;
pub use metrics::MetricsLayer;
pub use service::QuiverService;

#[allow(missing_docs, unused_qualifications, clippy::all, clippy::pedantic)]
pub mod pb {
    tonic::include_proto!("quiver.v1");
}

/// Largest message accepted or sent (64 MiB), e.g. for large upsert batches.
pub const MAX_MESSAGE_BYTES: usize = 64 << 20;

/// Builds the gRPC service with authentication applied.
pub fn service(
    db: Arc<Database>,
    keys: Arc<KeyStore>,
) -> tonic::service::interceptor::InterceptedService<
    pb::quiver_server::QuiverServer<QuiverService>,
    Authenticator,
> {
    let server = pb::quiver_server::QuiverServer::new(QuiverService::new(db))
        .max_decoding_message_size(MAX_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_MESSAGE_BYTES);
    tonic::service::interceptor::InterceptedService::new(server, Authenticator::new(keys))
}

/// TLS settings for [`serve`] from a PEM certificate chain and private key.
/// With `client_ca_pem`, clients must present a certificate it signed.
pub fn tls_config(cert_pem: &[u8], key_pem: &[u8], client_ca_pem: Option<&[u8]>) -> ServerTlsConfig {
    let config = ServerTlsConfig::new().identity(Identity::from_pem(cert_pem, key_pem));
    match client_ca_pem {
        Some(ca) => config.client_ca_root(Certificate::from_pem(ca)),
        None => config,
    }
}

/// Serves the gRPC API on `listener` until `shutdown` resolves, over TLS
/// when `tls` is given.
pub async fn serve(
    listener: tokio::net::TcpListener,
    db: Arc<Database>,
    keys: Arc<KeyStore>,
    tls: Option<ServerTlsConfig>,
    shutdown: impl Future<Output = ()> + Send,
) -> Result<(), tonic::transport::Error> {
    let mut builder = tonic::transport::Server::builder();
    if let Some(tls) = tls {
        builder = builder.tls_config(tls)?;
    }
    builder
        .layer(MetricsLayer::new(Arc::clone(&db)))
        .add_service(service(db, keys))
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
        .await
}
