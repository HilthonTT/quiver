use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use quiver_db::auth::{KeyStore, Role};

use crate::error::ApiError;

pub(crate) type Guard = (Arc<KeyStore>, Role);

pub(crate) async fn require(
    State((keys, required)): State<Guard>,
    req: Request,
    next: Next,
) -> Response {
    if keys.is_empty() {
        return next.run(req).await;
    }

    let headers = req.headers();
    let presented = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));

    match presented.and_then(|k| keys.find(k)) {
        Some(key) if key.role.allows(required) => next.run(req).await,
        Some(key) => {
            tracing::debug!(key = %key.name, "API key lacks the required role");
            ApiError::forbidden().into_response()
        }
        None => ApiError::unauthorized().into_response(),
    }
}
