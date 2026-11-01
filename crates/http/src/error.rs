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

    /// A missing point in an existing collection. Its own code lets clients
    /// tell it apart from a missing collection, which is also a 404.
    pub fn point_not_found(id: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "point_not_found",
            format!("point `{id}` not found"),
        )
    }

    pub fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing or invalid API key",
        )
    }

    pub fn forbidden() -> Self {
        Self::new(StatusCode::FORBIDDEN, "forbidden", "this API key is read-only")
    }

    /// An endpoint that spans every collection, refused to a key limited
    /// to some of them.
    pub fn server_wide() -> Self {
        Self::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            "this API key is limited to some collections and cannot use server-wide endpoints",
        )
    }

    pub fn payload_too_large(limit: usize) -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            format!("request body exceeds the limit of {limit} bytes"),
        )
    }

    /// An uploaded snapshot archive that cannot be read.
    pub fn invalid_snapshot(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_snapshot", message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", message)
    }
}

impl From<quiver_db::Error> for ApiError {
    fn from(err: quiver_db::Error) -> Self {
        use quiver_db::Error as E;

        let (status, code) = match &err {
            E::CollectionNotFound(_) | E::SnapshotNotFound(_) | E::AliasNotFound(_) => {
                (StatusCode::NOT_FOUND, "not_found")
            }
            E::PointNotFound(_) => (StatusCode::NOT_FOUND, "point_not_found"),
            E::CollectionExists(_) | E::AliasExists(_) => (StatusCode::CONFLICT, "already_exists"),
            E::DimensionMismatch { .. } => (StatusCode::BAD_REQUEST, "dimension_mismatch"),
            E::InvalidArgument(_) => (StatusCode::BAD_REQUEST, "invalid_argument"),
            E::InvalidFilter(_) => (StatusCode::BAD_REQUEST, "invalid_filter"),
            E::Unsupported(_) => (StatusCode::NOT_IMPLEMENTED, "unsupported"),
            E::Forbidden(_) => (StatusCode::FORBIDDEN, "forbidden"),
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
        // Malformed JSON and bodies of the wrong shape (bad types, unknown
        // fields) are plain invalid input: 400, like every other validation
        // error. Size and content-type rejections keep their own status.
        let status = match rejection {
            JsonRejection::JsonDataError(_) | JsonRejection::JsonSyntaxError(_) => StatusCode::BAD_REQUEST,
            _ => rejection.status(),
        };
        Self::new(status, "invalid_body", rejection.body_text())
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
