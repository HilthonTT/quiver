use std::io;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("collection `{0}` not found")]
    CollectionNotFound(String),

    #[error("collection `{0}` already exists")]
    CollectionExists(String),

    #[error("alias `{0}` not found")]
    AliasNotFound(String),

    #[error("`{0}` is already an alias")]
    AliasExists(String),

    #[error("snapshot `{0}` not found")]
    SnapshotNotFound(String),

    #[error("point `{0}` not found")]
    PointNotFound(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("invalid filter: {0}")]
    InvalidFilter(String),

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("corrupt data: {0}")]
    Corrupt(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("i/o error: {0}")]
    Io(#[from] io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl Error {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }
}
