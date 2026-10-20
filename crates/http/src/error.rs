use axum::Json;
use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_argument", message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found", message)
    }

    pub fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing or invalid API key",
        )
    }

    pub fn forbidden() -> Self {
        Self::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            "this API key is read-only",
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", message)
    }
}

impl From<quiver_db::Error> for ApiError {
    fn from(err: quiver_db::Error) -> Self {
        use quiver_db::Error as E;

        let (status, code) = match &err {
            E::CollectionNotFound(_) | E::SnapshotNotFound(_) => {
                (StatusCode::NOT_FOUND, "not_found")
            }
            E::CollectionExists(_) => (StatusCode::CONFLICT, "already_exists"),
            E::DimensionMismatch { .. } => (StatusCode::BAD_REQUEST, "dimension_mismatch"),
            E::InvalidArgument(_) => (StatusCode::BAD_REQUEST, "invalid_argument"),
            E::InvalidFilter(_) => (StatusCode::BAD_REQUEST, "invalid_filter"),
            E::Unsupported(_) => (StatusCode::NOT_IMPLEMENTED, "unsupported"),
            _ => {
                tracing::error!(error = %err, "internal error");
                (StatusCode::INTERNAL_SERVER_ERROR, "internal")
            }
        };
        Self::new(status, code, err.to_string())
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        Self::new(rejection.status(), "invalid_body", rejection.body_text())
    }
}

impl From<QueryRejection> for ApiError {
    fn from(rejection: QueryRejection) -> Self {
        Self::bad_request(rejection.body_text())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({ "error": { "code": self.code, "message": self.message } });
        (self.status, Json(body)).into_response()
    }
}
