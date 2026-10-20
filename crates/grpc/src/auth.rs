use std::sync::Arc;

use quiver_db::auth::{KeyStore, Role};
use tonic::{Request, Status};

/// Resolves the caller's API key and attaches its [`Role`] to the request.
///
/// Accepts `authorization: Bearer <key>` or `x-api-key: <key>` metadata.
/// With no keys configured every caller gets [`Role::ReadWrite`]. Each RPC
/// then checks the role it needs.
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
        let role = if self.keys.is_empty() {
            Role::ReadWrite
        } else {
            let md = req.metadata();
            let presented = md
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .or_else(|| md.get("x-api-key").and_then(|v| v.to_str().ok()));
            presented
                .and_then(|k| self.keys.find(k))
                .map(|k| k.role)
                .ok_or_else(|| Status::unauthenticated("missing or invalid API key"))?
        };
        req.extensions_mut().insert(role);
        Ok(req)
    }
}

/// Fails unless the request's key grants `needed`.
pub(crate) fn require<T>(req: &Request<T>, needed: Role) -> Result<(), Status> {
    match req.extensions().get::<Role>() {
        Some(role) if role.allows(needed) => Ok(()),
        Some(_) => Err(Status::permission_denied("this API key is read-only")),
        None => Err(Status::unauthenticated("missing or invalid API key")),
    }
}
