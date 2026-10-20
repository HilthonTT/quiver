use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use quiver_db::auth::{KeyStore, Role};

use crate::error::ApiError;

pub(crate) type Guard = (Arc<KeyStore>, Role);

pub(crate) async fn require(State((keys, required)): State<Guard>, req: Request, next: Next) -> Response {
    if keys.is_empty() {
        return next.run(req).await;
    }

    let headers = req.headers();
    let presented = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(bearer_token)
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

/// The token of an `Authorization: Bearer <token>` value. The scheme is
/// case-insensitive (RFC 7235) and whitespace around the token is ignored.
pub(crate) fn bearer_token(value: &str) -> Option<&str> {
    let (scheme, token) = value.trim().split_once(|c: char| c.is_ascii_whitespace())?;
    let token = token.trim();
    (scheme.eq_ignore_ascii_case("bearer") && !token.is_empty()).then_some(token)
}

#[cfg(test)]
mod tests {
    use super::bearer_token;

    #[test]
    fn bearer_scheme_is_case_insensitive_and_trimmed() {
        assert_eq!(bearer_token("Bearer abc"), Some("abc"));
        assert_eq!(bearer_token("bearer abc"), Some("abc"));
        assert_eq!(bearer_token("BEARER   abc  "), Some("abc"));
        assert_eq!(bearer_token("  Bearer\tabc"), Some("abc"));
        assert_eq!(bearer_token("Basic abc"), None);
        assert_eq!(bearer_token("Bearer"), None);
        assert_eq!(bearer_token("Bearer   "), None);
        assert_eq!(bearer_token("Bearerabc"), None);
    }
}
