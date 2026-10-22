mod config;
mod health;

use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use quiver_db::{Database, ExpirySweeper, SnapshotScheduler};
use quiver_http::HttpOptions;
use tokio::sync::watch;
use tracing_subscriber::EnvFilter;

use crate::config::Config;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// How long in-flight requests get to finish once shutdown starts.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

#[tokio::main]
async fn main() {
    // A missing .env file is fine; real environment variables win over it.
    let dotenv = dotenvy::dotenv();
    let config = Config::parse();
    if config.healthcheck {
        match health::check(&config).await {
            Ok(()) => return,
            Err(e) => {
                eprintln!("unhealthy: {e}");
                std::process::exit(1);
            }
        }
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();
    if let Ok(path) = dotenv {
        tracing::info!(path = %path.display(), "loaded environment file");
    }

    if let Err(e) = run(config).await {
        tracing::error!("{e}");
        std::process::exit(1);
    }
}

async fn run(config: Config) -> Result<(), BoxError> {
    config.validate()?;
    let keys = Arc::new(config.key_store()?);
    // Load the certificates before anything else, so bad ones fail fast.
    let tls = config.tls_pem()?;
    let http_tls = match &tls {
        Some(pem) => Some(quiver_http::tls_config(
            &pem.cert,
            &pem.key,
            pem.client_ca.as_deref(),
        )?),
        None => None,
    };

    let db = {
        let (root, opts) = (config.root(), config.db_options());
        Arc::new(tokio::task::spawn_blocking(move || Database::new(root, opts)).await??)
    };
    if config.in_memory {
        tracing::warn!("running in memory: data is lost on exit");
    }
    if keys.is_empty() && !config.host.is_loopback() {
        tracing::warn!("listening on a non-loopback address without API keys; anyone can read and write");
    }
    if !keys.is_empty() && tls.is_none() && !config.host.is_loopback() {
        tracing::warn!(
            "API keys are sent in plain text; set --tls-cert and --tls-key or terminate TLS in a proxy"
        );
    }

    let scheduler = config.snapshot_every().map(|every| {
        tracing::info!(
            every_secs = every.as_secs(),
            keep = config.snapshot_keep,
            "periodic snapshots enabled"
        );
        SnapshotScheduler::start(Arc::clone(&db), every)
    });
    let sweeper = config
        .expiry_every()
        .map(|every| ExpirySweeper::start(Arc::clone(&db), every));

    let (stop_tx, stop_rx) = watch::channel(false);
    let stopped = move || {
        let mut rx = stop_rx.clone();
        async move {
            let _ = rx.wait_for(|stop| *stop).await;
        }
    };

    let mut tasks = tokio::task::JoinSet::<Result<(), BoxError>>::new();
    if !config.disable_http {
        let listener = tokio::net::TcpListener::bind(config.http_addr()).await?;
        let router = quiver_http::router(
            Arc::clone(&db),
            HttpOptions {
                keys: Arc::clone(&keys),
                max_body_bytes: Some(config.max_body_mb << 20),
                max_upload_bytes: None,
                permissive_cors: config.cors,
            },
        );
        tracing::info!(addr = %config.http_addr(), tls = http_tls.is_some(), "REST API listening");
        let shutdown = stopped();
        match http_tls {
            Some(tls) => {
                tasks.spawn(async move { Ok(quiver_http::serve_tls(listener, tls, router, shutdown).await?) })
            }
            None => tasks.spawn(async move { Ok(quiver_http::serve(listener, router, shutdown).await?) }),
        };
    }
    if !config.disable_grpc {
        let listener = tokio::net::TcpListener::bind(config.grpc_addr()).await?;
        tracing::info!(addr = %config.grpc_addr(), tls = tls.is_some(), "gRPC API listening");
        let grpc_tls = tls
            .as_ref()
            .map(|pem| quiver_grpc::tls_config(&pem.cert, &pem.key, pem.client_ca.as_deref()));
        let (db, keys, shutdown) = (Arc::clone(&db), Arc::clone(&keys), stopped());
        tasks.spawn(async move { Ok(quiver_grpc::serve(listener, db, keys, grpc_tls, shutdown).await?) });
    }
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        storage = %db.options().storage,
        collections = db.list_collections().len(),
        auth = if keys.is_empty() { "disabled" } else { "enabled" },
        "Quiver ready"
    );

    // Run until a signal arrives or a server fails.
    let result = tokio::select! {
        () = shutdown_signal() => Ok(()),
        Some(res) = tasks.join_next() => res.map_err(BoxError::from).and_then(|r| r),
    };
    tracing::info!("shutting down");
    let _ = stop_tx.send(true);
    let drain = async {
        while let Some(res) = tasks.join_next().await {
            if let Err(e) = res.map_err(BoxError::from).and_then(|r| r) {
                tracing::error!(error = %e, "server task failed during shutdown");
            }
        }
    };
    // A stalled client must not keep the server (and the final flush) waiting.
    if tokio::time::timeout(SHUTDOWN_GRACE, drain).await.is_err() {
        tracing::warn!(
            grace_secs = SHUTDOWN_GRACE.as_secs(),
            "requests still in flight after the grace period; closing them"
        );
        tasks.shutdown().await;
    }
    drop(scheduler);
    drop(sweeper);

    tracing::info!("flushing collections");
    tokio::task::spawn_blocking(move || db.flush()).await??;
    result
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
