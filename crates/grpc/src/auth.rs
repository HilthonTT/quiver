use std::sync::Arc;

use quiver_db::auth::{Access, KeyStore, Role};
use tonic::{Request, Status};

/// Resolves the caller's API key and attaches its [`Access`] (role and
/// collection scope) to the request.
///
/// Accepts `authorization: Bearer <key>` or `x-api-key: <key>` metadata.
/// With no keys configured every caller gets [`Role::ReadWrite`]. A caller
/// that presents no key gets no role, which only public RPCs (`Health`)
/// accept; a wrong key is rejected outright. Each RPC then checks the role
/// it needs and the collections it names.
#[derive(Debug, Clone)]
pub struct Authenticator {
    keys: Arc<KeyStore>,
}

impl Authenticator {
    /// An interceptor checking against `keys`.
    pub fn new(keys: Arc<KeyStore>) -> Self {
        Self { keys }
    }
}

impl tonic::service::Interceptor for Authenticator {
    fn call(&mut self, mut req: Request<()>) -> Result<Request<()>, Status> {
        let access = if self.keys.is_empty() {
            Access::full()
        } else {
            let md = req.metadata();
            let presented = md
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(bearer_token)
                .or_else(|| md.get("x-api-key").and_then(|v| v.to_str().ok()));
            let Some(presented) = presented else {
                // Left to `require`, so that `Health` stays public.
                return Ok(req);
            };
            self.keys
                .find(presented)
                .map(|k| k.access())
                .ok_or_else(|| Status::unauthenticated("missing or invalid API key"))?
        };
        req.extensions_mut().insert(access);
        Ok(req)
    }
}

/// The token of an `authorization: Bearer <token>` value. The scheme is
/// case-insensitive (RFC 7235) and whitespace around the token is ignored.
fn bearer_token(value: &str) -> Option<&str> {
    let (scheme, token) = value.trim().split_once(|c: char| c.is_ascii_whitespace())?;
    let token = token.trim();
    (scheme.eq_ignore_ascii_case("bearer") && !token.is_empty()).then_some(token)
}

/// Fails unless the request's key grants `needed`; returns its access.
pub(crate) fn require<T>(req: &Request<T>, needed: Role) -> Result<Access, Status> {
    match req.extensions().get::<Access>() {
        Some(access) if access.role.allows(needed) => Ok(access.clone()),
        Some(_) => Err(Status::permission_denied("this API key is read-only")),
        None => Err(Status::unauthenticated("missing or invalid API key")),
    }
}

#[cfg(test)]
mod tests {
    use super::bearer_token;

    #[test]
    fn bearer_scheme_is_case_insensitive_and_trimmed() {
        assert_eq!(bearer_token("Bearer abc"), Some("abc"));
        assert_eq!(bearer_token("bearer abc"), Some("abc"));
        assert_eq!(bearer_token("BEARER   abc  "), Some("abc"));
        assert_eq!(bearer_token("Basic abc"), None);
        assert_eq!(bearer_token("Bearer "), None);
        assert_eq!(bearer_token("Bearerabc"), None);
    }
}
