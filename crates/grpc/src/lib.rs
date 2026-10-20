mod auth;
mod convert;
mod service;
use std::sync::Arc;

use quiver_db::Database;
use quiver_db::auth::KeyStore;
use tokio_stream::wrappers::TcpListenerStream;

pub use auth::Authenticator;
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

/// Serves the gRPC API on `listener` until `shutdown` resolves.
pub async fn serve(
    listener: tokio::net::TcpListener,
    db: Arc<Database>,
    keys: Arc<KeyStore>,
    shutdown: impl Future<Output = ()> + Send,
) -> Result<(), tonic::transport::Error> {
    tonic::transport::Server::builder()
        .add_service(service(db, keys))
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
        .await
}
