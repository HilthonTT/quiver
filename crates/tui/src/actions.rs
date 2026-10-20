use quiver_client::{Client, ClientError, SearchRequest};
use quiver_types::PointInput;
use tokio::runtime::Runtime;

use crate::app::{Action, App, ListMode, Modal, PAGE_SIZE, Query, Screen, ServerState, short_id};
use crate::embeddings::Embedder;

/// Runs actions on a single-threaded tokio runtime.
#[derive(Debug)]
pub struct Executor {
    rt: Runtime,
    client: Client,
    embedder: Option<Embedder>,
}

/// A user-facing description of a client error.
fn describe(e: &ClientError) -> String {
    match e {
        ClientError::Api { message, .. } => message.clone(),
        ClientError::Http(err) if err.is_connect() || err.is_timeout() => {
            format!("cannot reach server: {err}")
        }
        other => other.to_string(),
    }
}

fn is_transport(e: &ClientError) -> bool {
    matches!(e, ClientError::Http(_))
}

impl Executor {
    /// Creates an executor for `client`, with optional text embedding.
    pub fn new(client: Client, embedder: Option<Embedder>) -> std::io::Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(Self { rt, client, embedder })
    }

    fn vector(&self, query: Query) -> Result<Vec<f32>, String> {
        match query {
            Query::Vector(v) => Ok(v),
            Query::Text(text) => {
                let embedder = self.embedder.as_ref().ok_or("text input needs --embedding-url")?;
                self.rt.block_on(embedder.embed(&text))
            }
        }
    }

    /// Performs `action`, then any follow-up it implies.
    pub fn run(&self, app: &mut App, action: Action) {
        let mut next = Some(action);
        while let Some(action) = next.take() {
            next = self.step(app, action);
        }
    }

    /// Records a client error; transport failures also mark the server unreachable.
    fn fail(app: &mut App, e: &ClientError) {
        if is_transport(e) {
            app.server = ServerState::Unreachable(describe(e));
        }
        app.error(describe(e));
    }

    fn step(&self, app: &mut App, action: Action) -> Option<Action> {
        let c = &self.client;
        match action {
            Action::Refresh => {
                match self.rt.block_on(c.info()) {
                    Ok(info) => app.server = ServerState::Online(info),
                    Err(e) => {
                        Self::fail(app, &e);
                        return None;
                    }
                }
                match self.rt.block_on(c.list_collections()) {
                    Ok(list) => {
                        app.collections = list;
                        app.selected = app.selected.min(app.collections.len().saturating_sub(1));
                    }
                    Err(e) => Self::fail(app, &e),
                }
                return match (app.screen, app.current.clone()) {
                    (Screen::Collection, Some(collection)) if app.list == ListMode::Points => {
                        Some(Action::LoadPoints {
                            collection,
                            offset: app.offset,
                        })
                    }
                    (Screen::Snapshots, _) => Some(Action::LoadSnapshots),
                    _ => None,
                };
            }
            Action::OpenCollection(name) => {
                app.current = Some(name.clone());
                app.screen = Screen::Collection;
                app.list = ListMode::Points;
                return Some(Action::LoadPoints {
                    collection: name,
                    offset: 0,
                });
            }
            Action::LoadPoints { collection, offset } => {
                match self.rt.block_on(c.scroll(&collection, offset, PAGE_SIZE, false)) {
                    Ok(page) => {
                        app.points = page.points;
                        app.next_offset = page.next_offset;
                        app.offset = offset;
                        app.row = 0;
                    }
                    Err(e) => Self::fail(app, &e),
                }
            }
            Action::ShowPoint {
                collection,
                id,
                distance,
            } => match self.rt.block_on(c.get_point(&collection, &id)) {
                Ok(Some(p)) => {
                    let mut body = format!("id: {}\n", p.id);
                    if let Some(d) = distance {
                        body.push_str(&format!("distance: {d}\n"));
                    }
                    let head: Vec<String> = p.vector.iter().take(16).map(|x| format!("{x:.4}")).collect();
                    let more = if p.vector.len() > 16 { ", …" } else { "" };
                    body.push_str(&format!(
                        "vector ({} dims): [{}{more}]\n\nmetadata:\n",
                        p.vector.len(),
                        head.join(", ")
                    ));
                    body.push_str(&match &p.metadata {
                        Some(m) => serde_json::to_string_pretty(m).unwrap_or_default(),
                        None => "(none)".into(),
                    });
                    app.modal = Some(Modal::Detail {
                        title: format!("Point {}", p.id),
                        body,
                        scroll: 0,
                    });
                }
                Ok(None) => app.error(format!("point `{id}` no longer exists")),
                Err(e) => Self::fail(app, &e),
            },
            Action::CreateCollection { name, config } => {
                match self.rt.block_on(c.create_collection(&name, &config)) {
                    Ok(_) => {
                        app.modal = None;
                        app.info(format!("created collection `{name}`"));
                        return Some(Action::Refresh);
                    }
                    Err(e) => Self::fail(app, &e),
                }
            }
            Action::DropCollection(name) => match self.rt.block_on(c.drop_collection(&name)) {
                Ok(()) => {
                    app.info(format!("dropped collection `{name}`"));
                    if app.current.as_deref() == Some(&name) {
                        app.current = None;
                        app.screen = Screen::Dashboard;
                    }
                    return Some(Action::Refresh);
                }
                Err(e) => Self::fail(app, &e),
            },
            Action::Insert {
                collection,
                id,
                query,
                metadata,
            } => {
                let vector = match self.vector(query) {
                    Ok(v) => v,
                    Err(e) => {
                        app.error(e);
                        return None;
                    }
                };
                let point = PointInput { id, vector, metadata };
                match self.rt.block_on(c.upsert(&collection, vec![point])) {
                    Ok(res) => {
                        app.modal = None;
                        let id = res.ids.first().cloned().unwrap_or_default();
                        app.info(format!("inserted point `{id}`"));
                        return Some(Action::Refresh);
                    }
                    Err(e) => Self::fail(app, &e),
                }
            }
            Action::Search {
                collection,
                query,
                k,
                filter,
            } => {
                let vector = match self.vector(query) {
                    Ok(v) => v,
                    Err(e) => {
                        app.error(e);
                        return None;
                    }
                };
                let mut req = SearchRequest::new(vector, k);
                req.filter = filter;
                match self.rt.block_on(c.search(&collection, &req)) {
                    Ok(hits) => {
                        app.modal = None;
                        app.info(format!("{} results · Esc returns to points", hits.len()));
                        app.list = ListMode::Results(hits);
                        app.row = 0;
                    }
                    Err(e) => Self::fail(app, &e),
                }
            }
            Action::DeletePoint { collection, id } => {
                match self
                    .rt
                    .block_on(c.delete_points(&collection, std::slice::from_ref(&id)))
                {
                    Ok(n) => {
                        app.info(if n == 0 {
                            format!("point `{id}` was already gone")
                        } else {
                            format!("deleted point `{id}`")
                        });
                        if let ListMode::Results(hits) = &mut app.list {
                            hits.retain(|h| h.id != id);
                        }
                        return Some(Action::Refresh);
                    }
                    Err(e) => Self::fail(app, &e),
                }
            }
            Action::CreateSnapshot(collection) => match self.rt.block_on(c.create_snapshot(&collection)) {
                Ok(info) => {
                    app.info(format!(
                        "snapshot {} of `{collection}` created ({} points)",
                        short_id(&info.manifest.id),
                        info.manifest.points
                    ));
                    if app.screen == Screen::Snapshots {
                        return Some(Action::LoadSnapshots);
                    }
                }
                Err(e) => Self::fail(app, &e),
            },
            Action::LoadSnapshots => match self.rt.block_on(c.list_snapshots(None)) {
                Ok(mut list) => {
                    // Newest first.
                    list.reverse();
                    app.snapshots = list;
                    app.snapshot_row = app.snapshot_row.min(app.snapshots.len().saturating_sub(1));
                }
                Err(e) => Self::fail(app, &e),
            },
            Action::RestoreSnapshot { id, name, replace } => {
                match self
                    .rt
                    .block_on(c.restore_snapshot(&id, name.as_deref(), replace))
                {
                    Ok(info) => {
                        app.modal = None;
                        app.info(format!(
                            "restored snapshot {} as `{}` ({} points)",
                            short_id(&id),
                            info.name,
                            info.count
                        ));
                        return Some(Action::Refresh);
                    }
                    Err(e) => Self::fail(app, &e),
                }
            }
            Action::DeleteSnapshot(id) => match self.rt.block_on(c.delete_snapshot(&id)) {
                Ok(()) => {
                    app.info(format!("deleted snapshot {}", short_id(&id)));
                    return Some(Action::LoadSnapshots);
                }
                Err(e) => Self::fail(app, &e),
            },
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreachable_server_is_reported_not_fatal() {
        // Port 9 (discard) is essentially never listening on localhost.
        let client = Client::new("http://127.0.0.1:9").unwrap();
        let exec = Executor::new(client, None).unwrap();
        let mut app = App::new("http://127.0.0.1:9", false);
        exec.run(&mut app, Action::Refresh);
        let ServerState::Unreachable(msg) = &app.server else {
            panic!("expected unreachable, got {:?}", app.server)
        };
        assert!(msg.contains("cannot reach server"), "{msg}");
        assert!(app.status.as_ref().unwrap().error);
    }

    #[test]
    fn text_queries_need_an_embedder() {
        let exec = Executor::new(Client::new("http://127.0.0.1:9").unwrap(), None).unwrap();
        assert!(exec.vector(Query::Text("hi".into())).is_err());
        assert_eq!(exec.vector(Query::Vector(vec![1.0])).unwrap(), vec![1.0]);
    }
}
