//! Records each RPC in the database's [`Metrics`](quiver_db::metrics::Metrics).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use quiver_db::Database;
use tonic::codegen::http;

/// Every RPC of `quiver.v1.Quiver`. Only these become metric labels.
const METHODS: [&str; 32] = [
    "Health",
    "Info",
    "ListCollections",
    "CreateCollection",
    "GetCollection",
    "DropCollection",
    "Flush",
    "Compact",
    "CreateFieldIndex",
    "DropFieldIndex",
    "UpdateCollection",
    "RenameCollection",
    "ListAliases",
    "SetAlias",
    "DeleteAlias",
    "Upsert",
    "GetPoints",
    "DeletePoints",
    "Scroll",
    "Count",
    "UpdateMetadata",
    "Search",
    "SearchBatch",
    "Recommend",
    "Query",
    "CreateSnapshot",
    "ListSnapshots",
    "GetSnapshot",
    "DeleteSnapshot",
    "RestoreSnapshot",
    "DownloadSnapshot",
    "UploadSnapshot",
];

/// A tower layer recording every RPC's latency and status code.
///
/// The status is read from the response headers, where tonic puts it for
/// RPCs that fail before sending a message. Anything else is counted as
/// `0` (OK), including a stream that fails after its first message.
#[derive(Debug, Clone)]
pub struct MetricsLayer {
    db: Arc<Database>,
}

impl MetricsLayer {
    /// Records into `db`'s metrics.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }
}

impl<S> tower::Layer<S> for MetricsLayer {
    type Service = MetricsService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        MetricsService {
            inner,
            db: Arc::clone(&self.db),
        }
    }
}

/// The service built by [`MetricsLayer`].
#[derive(Debug, Clone)]
pub struct MetricsService<S> {
    inner: S,
    db: Arc<Database>,
}

impl<S, B, R> tower::Service<http::Request<B>> for MetricsService<S>
where
    S: tower::Service<http::Request<B>, Response = http::Response<R>> + Clone + Send + 'static,
    S::Future: Send,
    B: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: http::Request<B>) -> Self::Future {
        // The service the caller readied, leaving a fresh clone in its place.
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let db = Arc::clone(&self.db);
        // Clients can send any path: collapse unknown ones so they cannot
        // create unbounded series.
        let path = req.uri().path();
        let route = match path.strip_prefix("/quiver.v1.Quiver/") {
            Some(method) if METHODS.contains(&method) => path.to_owned(),
            _ => "unmatched".to_owned(),
        };
        Box::pin(async move {
            let start = Instant::now();
            let res = inner.call(req).await;
            let status = match &res {
                Ok(r) => r
                    .headers()
                    .get("grpc-status")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("0")
                    .to_owned(),
                Err(_) => "transport_error".to_owned(),
            };
            db.metrics().record("grpc", &route, &status, start.elapsed());
            res
        })
    }
}

#[cfg(test)]
mod tests {
    use super::METHODS;

    #[test]
    fn method_list_matches_the_proto() {
        let proto = include_str!("../proto/quiver.proto");
        let declared: Vec<&str> = proto
            .lines()
            .filter_map(|l| l.trim().strip_prefix("rpc "))
            .filter_map(|l| l.split('(').next())
            .collect();
        assert_eq!(declared, METHODS);
    }
}
