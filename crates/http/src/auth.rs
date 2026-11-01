use std::sync::Arc;

use axum::extract::{FromRequestParts, MatchedPath, RawPathParams, Request, State};
use axum::http::header::AUTHORIZATION;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use quiver_db::Database;
use quiver_db::auth::{Access, KeyStore, Role};

use crate::error::ApiError;

/// What the authentication middleware of one group of routes checks.
#[derive(Clone)]
pub(crate) struct Guard {
    pub keys: Arc<KeyStore>,
    pub role: Role,
    pub db: Arc<Database>,
}

/// Resolves the caller's key, checks its role, and attaches its [`Access`]
/// to the request for the handlers.
///
/// The collection scope is checked here for every route with a `{name}`
/// (collection or alias) or `{alias}` parameter, and for every
/// `/snapshots/{id}` route, so that routes added later are covered too.
/// Handlers check the names that arrive in bodies or queries and filter
/// listings.
pub(crate) async fn require(State(guard): State<Guard>, req: Request, next: Next) -> Response {
    let access = if guard.keys.is_empty() {
        Access::full()
    } else {
        let headers = req.headers();
        let presented = headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(bearer_token)
            .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));
        match presented.and_then(|k| guard.keys.find(k)) {
            Some(key) if key.role.allows(guard.role) => key.access(),
            Some(key) => {
                tracing::debug!(key = %key.name, "API key lacks the required role");
                return ApiError::forbidden().into_response();
            }
            None => return ApiError::unauthorized().into_response(),
        }
    };

    let (mut parts, body) = req.into_parts();
    if access.is_scoped()
        && let Err(e) = check_path(&guard.db, &access, &mut parts).await
    {
        return e.into_response();
    }
    parts.extensions.insert(access);
    next.run(Request::from_parts(parts, body)).await
}

async fn check_path(
    db: &Arc<Database>,
    access: &Access,
    parts: &mut axum::http::request::Parts,
) -> Result<(), ApiError> {
    let route = parts
        .extensions
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_owned())
        .unwrap_or_default();
    // Routes without parameters have none to check.
    let Ok(params) = RawPathParams::from_request_parts(parts, &()).await else {
        return Ok(());
    };
    for (key, value) in &params {
        match key {
            "name" => db.authorize(access, value)?,
            "alias" => db.authorize_alias(access, value)?,
            "id" if route.starts_with("/snapshots/") => {
                let (db, access, id) = (Arc::clone(db), access.clone(), value.to_owned());
                tokio::task::spawn_blocking(move || db.authorize_snapshot(&access, &id))
                    .await
                    .map_err(|e| ApiError::internal(format!("task failed: {e}")))??;
            }
            _ => {}
        }
    }
    Ok(())
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
