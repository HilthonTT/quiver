mod actions;
mod app;
mod embeddings;
mod input;
mod ui;

use std::time::Duration;

use clap::Parser;
use quiver_client::Client;
use ratatui::crossterm::event::{self, Event, KeyEventKind};

use crate::actions::Executor;
use crate::app::{Action, App};
use crate::embeddings::{Embedder, EmbeddingConfig};

/// Browse, search and manage a Quiver server from the terminal.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Server URL.
    #[arg(long, env = "QUIVER_URL", default_value = "http://localhost:6900")]
    url: String,

    /// API key (sent as `Authorization: Bearer <key>`).
    #[arg(long, env = "QUIVER_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// OpenAI-compatible embeddings endpoint, e.g.
    /// http://localhost:11434/v1/embeddings (Ollama). Enables text input.
    #[arg(long, env = "QUIVER_EMBEDDING_URL")]
    embedding_url: Option<String>,

    /// Embedding model name, e.g. `nomic-embed-text` or `text-embedding-3-small`.
    #[arg(long, env = "QUIVER_EMBEDDING_MODEL")]
    embedding_model: Option<String>,

    /// API key for the embeddings endpoint.
    #[arg(long, env = "QUIVER_EMBEDDING_API_KEY", hide_env_values = true)]
    embedding_api_key: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let client = Client::new(args.url.clone())?.with_api_key(args.api_key.unwrap_or_default());
    let embedder = args
        .embedding_url
        .map(|url| {
            Embedder::new(EmbeddingConfig {
                url,
                model: args.embedding_model,
                api_key: args.embedding_api_key,
            })
        })
        .transpose()?;

    let mut app = App::new(&args.url, embedder.is_some());
    let executor = Executor::new(client, embedder)?;
    executor.run(&mut app, Action::Refresh);

    // `ratatui::init` enters the alternate screen and raw mode, and installs
    // a panic hook that restores the terminal before the panic is printed.
    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app, &executor);
    ratatui::restore();
    result.map_err(Into::into)
}

fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App, executor: &Executor) -> std::io::Result<()> {
    while !app.should_quit {
        terminal.draw(|frame| ui::draw(frame, app))?;
        if !event::poll(Duration::from_millis(250))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if let Some(action) = app.handle_key(key) {
                    executor.run(app, action);
                }
            }
            // The next draw picks up the new size.
            Event::Resize(..) => terminal.autoresize()?,
            _ => {}
        }
    }
    Ok(())
}
