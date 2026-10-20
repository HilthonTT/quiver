use quiver_client::{Client, ClientError, SearchRequest};
use quiver_types::PointInput;
use tokio::runtime::Runtime;

use crate::app::{Action, App, ListMode, Modal, PAGE_SIZE, Page, Query, Screen, ServerState, short_id};
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
                    // Reload points even while showing results, so they are
                    // current when the user goes back to them.
                    (Screen::Collection, Some(collection)) => Some(Action::LoadPoints {
                        collection,
                        page: Page::Current,
                    }),
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
                    page: Page::First,
                });
            }
            Action::LoadPoints { collection, page } => {
                // The page stack as it will be if the load succeeds.
                let mut cursors = app.cursors.clone();
                match page {
                    Page::First => cursors = vec![None],
                    Page::Current => {}
                    Page::Next => cursors.push(Some(app.next_cursor.clone()?)),
                    Page::Previous => {
                        if cursors.len() > 1 {
                            cursors.pop();
                        }
                    }
                }
                if cursors.is_empty() {
                    cursors.push(None);
                }
                let cursor = cursors.last().cloned().flatten();
                match self
                    .rt
                    .block_on(c.scroll(&collection, cursor.as_deref(), PAGE_SIZE, false))
                {
                    // Deleting the last points of the last page leaves it empty:
                    // step back until there is something to show.
                    Ok(page) if page.points.is_empty() && cursors.len() > 1 => {
                        app.cursors = cursors;
                        return Some(Action::LoadPoints {
                            collection,
                            page: Page::Previous,
                        });
                    }
                    Ok(page) => {
                        app.points = page.points;
                        app.next_cursor = page.next_cursor;
                        app.cursors = cursors;
                        if app.list == ListMode::Points {
                            app.row = 0;
                        }
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
                            app.row = app.row.min(hits.len().saturating_sub(1));
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
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    use quiver_types::SearchHit;
    use serde_json::{Value, json};

    /// A minimal stand-in for the server: one collection `docs` (dimension 2)
    /// holding the given point ids, enough for paging, insert and delete.
    fn fake_server(ids: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let ids = Arc::new(Mutex::new(ids));
        let state = Arc::clone(&ids);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let mut parts = line.split_whitespace();
                let (method, target) = (parts.next().unwrap().to_owned(), parts.next().unwrap().to_owned());
                let mut length = 0;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    if header.trim().is_empty() {
                        break;
                    }
                    if let Some((k, v)) = header.split_once(':')
                        && k.eq_ignore_ascii_case("content-length")
                    {
                        length = v.trim().parse().unwrap();
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                let (path, query) = target.split_once('?').unwrap_or((&target, ""));
                let param = |name: &str| -> Option<String> {
                    query
                        .split('&')
                        .find_map(|kv| kv.strip_prefix(name)?.strip_prefix('='))
                        .map(str::to_owned)
                };
                let mut ids = state.lock().unwrap();
                let reply = match (method.as_str(), path) {
                    ("GET", "/info") => json!({
                        "version": "0", "storage": "memory", "snapshots_enabled": false,
                        "collections": 1, "points": ids.len(),
                    }),
                    ("GET", "/collections") => json!({ "collections": [{
                        "name": "docs", "dimension": 2, "metric": "cosine",
                        "index": { "type": "flat" }, "count": ids.len(), "tombstones": 0,
                    }]}),
                    ("GET", "/collections/docs/points") => {
                        assert!(param("offset").is_none(), "offset paging is gone");
                        let limit: usize = param("limit").unwrap().parse().unwrap();
                        let cursor = param("cursor").unwrap_or_default();
                        let mut sorted: Vec<&String> = ids.iter().filter(|id| **id > cursor).collect();
                        sorted.sort();
                        let page: Vec<Value> =
                            sorted.iter().take(limit).map(|id| json!({ "id": id })).collect();
                        let next = (sorted.len() > limit && limit > 0).then(|| sorted[limit - 1].clone());
                        json!({ "points": page, "next_cursor": next })
                    }
                    ("POST", "/collections/docs/points") => {
                        let id = body["points"][0]["id"].as_str().unwrap().to_owned();
                        ids.push(id.clone());
                        json!({ "upserted": 1, "ids": [id] })
                    }
                    ("POST", "/collections/docs/points/delete") => {
                        let gone: Vec<&str> = body["ids"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter_map(Value::as_str)
                            .collect();
                        let before = ids.len();
                        ids.retain(|id| !gone.contains(&id.as_str()));
                        json!({ "deleted": before - ids.len() })
                    }
                    other => panic!("unexpected request {other:?}"),
                };
                drop(ids);
                let reply = reply.to_string();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                    reply.len()
                )
                .unwrap();
            }
        });
        (url, ids)
    }

    fn open_docs(ids: &[&str]) -> (Executor, App, Arc<Mutex<Vec<String>>>) {
        let (url, state) = fake_server(ids.iter().map(|s| (*s).to_owned()).collect());
        let exec = Executor::new(Client::new(&url).unwrap(), None).unwrap();
        let mut app = App::new(&url, false);
        exec.run(&mut app, Action::Refresh);
        exec.run(&mut app, Action::OpenCollection("docs".into()));
        (exec, app, state)
    }

    fn hits(ids: &[&str]) -> ListMode {
        ListMode::Results(
            ids.iter()
                .map(|id| SearchHit {
                    id: (*id).into(),
                    distance: 0.0,
                    metadata: None,
                    vector: None,
                })
                .collect(),
        )
    }

    fn point_ids(app: &App) -> Vec<&str> {
        app.points.iter().map(|p| p.id.as_str()).collect()
    }

    #[test]
    fn points_are_reloaded_after_changes_made_from_search_results() {
        let (exec, mut app, _) = open_docs(&["a", "b", "c"]);
        app.list = hits(&["a", "b"]);
        exec.run(
            &mut app,
            Action::DeletePoint {
                collection: "docs".into(),
                id: "a".into(),
            },
        );
        exec.run(
            &mut app,
            Action::Insert {
                collection: "docs".into(),
                id: Some("d".into()),
                query: Query::Vector(vec![1.0, 0.0]),
                metadata: None,
            },
        );
        assert!(matches!(app.list, ListMode::Results(_)), "still showing results");
        let back = crate::app::tests::press(&mut app, ratatui::crossterm::event::KeyCode::Esc);
        assert!(
            matches!(back, Some(Action::LoadPoints { .. })),
            "Esc reloads points"
        );
        exec.run(&mut app, back.unwrap());
        assert_eq!(app.list, ListMode::Points);
        assert_eq!(point_ids(&app), ["b", "c", "d"]);
    }

    #[test]
    fn deleting_the_last_result_keeps_the_selection_in_range() {
        let (exec, mut app, _) = open_docs(&["a", "b", "c"]);
        app.list = hits(&["a", "b", "c"]);
        app.row = 2;
        exec.run(
            &mut app,
            Action::DeletePoint {
                collection: "docs".into(),
                id: "c".into(),
            },
        );
        assert_eq!(app.row, 1);
        assert_eq!(app.selected_row().map(|(id, _)| id).as_deref(), Some("b"));
    }

    fn first_id(app: &App) -> &str {
        &app.points[0].id
    }

    #[test]
    fn pages_forward_and_back_by_cursor() {
        let ids: Vec<String> = (0..2 * PAGE_SIZE + 5).map(|i| format!("p{i:03}")).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let (exec, mut app, _) = open_docs(&refs);
        use ratatui::crossterm::event::KeyCode;
        let press = |app: &mut App, code| {
            let action = crate::app::tests::press(app, code);
            if let Some(a) = action.clone() {
                exec.run(app, a);
            }
            action.is_some()
        };

        assert_eq!((app.page_index(), first_id(&app)), (0, "p000"));
        assert!(!press(&mut app, KeyCode::PageUp), "no page before the first");
        assert!(press(&mut app, KeyCode::PageDown));
        assert_eq!((app.page_index(), first_id(&app)), (1, "p050"));
        assert!(press(&mut app, KeyCode::Right));
        assert_eq!((app.page_index(), app.points.len()), (2, 5));
        assert_eq!(app.next_cursor, None);
        assert!(!press(&mut app, KeyCode::PageDown), "no page after the last");
        assert!(press(&mut app, KeyCode::PageUp));
        assert_eq!((app.page_index(), first_id(&app)), (1, "p050"));
        assert!(press(&mut app, KeyCode::Left));
        assert_eq!((app.page_index(), first_id(&app)), (0, "p000"));
    }

    #[test]
    fn emptied_last_page_steps_back() {
        let ids: Vec<String> = (0..=PAGE_SIZE).map(|i| format!("p{i:03}")).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let (exec, mut app, _) = open_docs(&refs);
        exec.run(
            &mut app,
            Action::LoadPoints {
                collection: "docs".into(),
                page: Page::Next,
            },
        );
        assert_eq!((app.page_index(), app.points.len()), (1, 1));
        let last = ids[PAGE_SIZE].clone();
        exec.run(
            &mut app,
            Action::DeletePoint {
                collection: "docs".into(),
                id: last,
            },
        );
        assert_eq!((app.page_index(), app.points.len()), (0, PAGE_SIZE));
        assert_eq!(app.next_cursor, None);
    }

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
