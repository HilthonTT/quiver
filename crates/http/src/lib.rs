pub mod error;
pub mod handlers;

mod auth;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use quiver_db::Database;
use quiver_db::auth::{KeyStore, Role};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::error::ApiError;

/// Settings that shape the HTTP layer.
#[derive(Debug, Clone, Default)]
pub struct HttpOptions {
    /// API keys. Empty disables authentication.
    pub keys: Arc<KeyStore>,
    /// Maximum JSON request body size in bytes (`None` = 64 MiB).
    pub max_body_bytes: Option<usize>,
    /// Maximum snapshot upload size in bytes (`None` = 16 GiB).
    pub max_upload_bytes: Option<usize>,
    /// Allow cross-origin requests from any origin.
    pub permissive_cors: bool,
}

/// Builds the application router.
pub fn router(db: Arc<Database>, options: HttpOptions) -> Router {
    use handlers::*;

    let guard = |role| middleware::from_fn_with_state((Arc::clone(&options.keys), role), auth::require);

    let read = Router::new()
        .route("/info", get(info))
        .route("/collections", get(list_collections))
        .route("/collections/{name}", get(get_collection))
        .route("/collections/{name}/points", get(scroll_points))
        .route("/collections/{name}/points/lookup", post(lookup_points))
        .route("/collections/{name}/points/count", post(count_points))
        .route("/collections/{name}/points/{id}", get(get_point))
        .route("/collections/{name}/search", post(search))
        .route("/collections/{name}/search/batch", post(search_batch))
        .route("/collections/{name}/recommend", post(recommend))
        .route("/collections/{name}/snapshots", get(list_collection_snapshots))
        .route("/snapshots", get(list_snapshots))
        .route("/snapshots/{id}", get(get_snapshot))
        .route("/snapshots/{id}/download", get(download_snapshot))
        .route_layer(guard(Role::ReadOnly));

    let upload_limit = options.max_upload_bytes.unwrap_or(16 << 30);
    let write = Router::new()
        .route("/collections", post(create_collection))
        .route("/collections/{name}", axum::routing::delete(drop_collection))
        .route("/collections/{name}/flush", post(flush_collection))
        .route("/collections/{name}/compact", post(compact_collection))
        .route("/collections/{name}/points", post(upsert_points))
        .route("/collections/{name}/points/delete", post(delete_points))
        .route("/collections/{name}/points/metadata", post(update_metadata))
        .route(
            "/collections/{name}/points/{id}",
            axum::routing::delete(delete_point),
        )
        .route("/collections/{name}/snapshots", post(create_snapshot))
        .route(
            "/snapshots",
            post(upload_snapshot).layer(axum::Extension(UploadLimit(upload_limit))),
        )
        .route("/snapshots/{id}", axum::routing::delete(delete_snapshot))
        .route("/snapshots/{id}/restore", post(restore_snapshot))
        .route_layer(guard(Role::ReadWrite));

    let mut app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .merge(read)
        .merge(write)
        .fallback(|| async { ApiError::not_found("no such route") })
        .with_state(db)
        .layer(DefaultBodyLimit::max(options.max_body_bytes.unwrap_or(64 << 20)))
        .layer(TraceLayer::new_for_http());

    if options.permissive_cors {
        app = app.layer(CorsLayer::permissive());
    }

    app
}

/// Serves `router` on `listener` until `shutdown` resolves.
pub async fn serve(
    listener: tokio::net::TcpListener,
    router: Router,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await
}
