use quiver_client::ServerInfo;
use quiver_types::snapshot::SnapshotInfo;
use quiver_types::{
    CollectionConfig, CollectionInfo, HnswParams, IndexConfig, KdTreeParams, Metric, Point, SearchHit,
};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde_json::Value;

use crate::input::{TextField, parse_json_object, parse_vector};

/// Points shown per page in the collection view.
pub const PAGE_SIZE: usize = 50;

/// Index types offered when creating a collection.
pub const INDEX_TYPES: [&str; 3] = ["hnsw", "flat", "kd_tree"];

/// Something to do against the server.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Reload server info and whatever the current screen shows.
    Refresh,
    /// Open a collection and load its first page of points.
    OpenCollection(String),
    /// Load a page of points.
    LoadPoints {
        /// Collection name.
        collection: String,
        /// Offset of the page.
        offset: usize,
    },
    /// Fetch a point (with its vector) and show it.
    ShowPoint {
        /// Collection name.
        collection: String,
        /// Point id.
        id: String,
        /// Distance to the last query, if it came from search results.
        distance: Option<f32>,
    },
    /// Create a collection.
    CreateCollection {
        /// Name.
        name: String,
        /// Configuration.
        config: CollectionConfig,
    },
    /// Drop a collection.
    DropCollection(String),
    /// Insert one point.
    Insert {
        /// Collection name.
        collection: String,
        /// Id, or `None` for a server-generated one.
        id: Option<String>,
        /// The vector, or text to embed.
        query: Query,
        /// Metadata object.
        metadata: Option<Value>,
    },
    /// Search a collection.
    Search {
        /// Collection name.
        collection: String,
        /// The query vector, or text to embed.
        query: Query,
        /// Number of results.
        k: usize,
        /// Metadata filter.
        filter: Option<Value>,
    },
    /// Delete one point.
    DeletePoint {
        /// Collection name.
        collection: String,
        /// Point id.
        id: String,
    },
    /// Snapshot a collection.
    CreateSnapshot(String),
    /// Reload the snapshot list.
    LoadSnapshots,
    /// Restore a snapshot.
    RestoreSnapshot {
        /// Snapshot id.
        id: String,
        /// New collection name, if any.
        name: Option<String>,
        /// Replace an existing collection.
        replace: bool,
    },
    /// Delete a snapshot.
    DeleteSnapshot(String),
}

/// A query given either as a vector or as text to embed.
#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// A literal vector.
    Vector(Vec<f32>),
    /// Text to send to the embedding endpoint.
    Text(String),
}

/// The main screens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    /// Server overview and collection list.
    Dashboard,
    /// One collection's points or search results.
    Collection,
    /// Snapshot list.
    Snapshots,
}

/// What the server last told us about itself.
#[derive(Debug, Clone, PartialEq)]
pub enum ServerState {
    /// Not contacted yet.
    Unknown,
    /// Connected.
    Online(ServerInfo),
    /// The last request failed at the transport level.
    Unreachable(String),
}

/// A message in the status bar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    /// Text to show.
    pub text: String,
    /// Render as an error.
    pub error: bool,
}

/// One input in a form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldKind {
    /// Free text.
    Text(TextField),
    /// One of several options, cycled with ←/→.
    Choice {
        /// The options.
        options: Vec<String>,
        /// Selected option.
        index: usize,
    },
    /// On/off, flipped with Space.
    Toggle(bool),
}

/// A labelled form input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// Label shown before the input.
    pub label: &'static str,
    /// The input.
    pub kind: FieldKind,
    /// Help text shown under the form when focused.
    pub hint: &'static str,
}

impl Field {
    fn text(label: &'static str, value: &str, hint: &'static str) -> Self {
        Self {
            label,
            kind: FieldKind::Text(TextField::new(value)),
            hint,
        }
    }

    fn choice(label: &'static str, options: Vec<String>, hint: &'static str) -> Self {
        Self {
            label,
            kind: FieldKind::Choice { options, index: 0 },
            hint,
        }
    }
}

/// Which form is open, with the context it needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormKind {
    /// Create a collection.
    NewCollection,
    /// Insert a point into a collection.
    Insert(String),
    /// Search a collection.
    Search(String),
    /// Restore a snapshot.
    Restore(String),
}

/// A modal form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Form {
    /// What the form does.
    pub kind: FormKind,
    /// Its inputs.
    pub fields: Vec<Field>,
    /// Focused input.
    pub focus: usize,
    /// Validation or server error to show.
    pub error: Option<String>,
}

// Field positions, per form.
const NC_NAME: usize = 0;
const NC_DIM: usize = 1;
const NC_METRIC: usize = 2;
const NC_INDEX: usize = 3;
const IN_ID: usize = 0;
const IN_MODE: usize = 1;
const IN_QUERY: usize = 2;
const IN_META: usize = 3;
const SE_MODE: usize = 0;
const SE_QUERY: usize = 1;
const SE_K: usize = 2;
const SE_FILTER: usize = 3;
const RE_NAME: usize = 0;
const RE_REPLACE: usize = 1;

fn mode_options(embeddings: bool) -> Vec<String> {
    if embeddings {
        vec!["vector".into(), "text".into()]
    } else {
        vec!["vector".into()]
    }
}

impl Form {
    /// The "new collection" form.
    pub fn new_collection() -> Self {
        Self::build(
            FormKind::NewCollection,
            vec![
                Field::text("Name", "", "letters, digits, _ and -"),
                Field::text("Dimension", "", "vector length, e.g. 384"),
                Field::choice(
                    "Metric",
                    Metric::ALL.iter().map(|m| m.as_str().to_owned()).collect(),
                    "←/→ to change",
                ),
                Field::choice(
                    "Index",
                    INDEX_TYPES.iter().map(|s| (*s).to_owned()).collect(),
                    "hnsw: fast approximate · flat: exact · kd_tree: exact, low dimensions",
                ),
            ],
        )
    }

    /// The insert form for `collection`.
    pub fn insert(collection: &str, embeddings: bool) -> Self {
        Self::build(
            FormKind::Insert(collection.to_owned()),
            vec![
                Field::text("Id", "", "leave blank for a generated UUID"),
                Field::choice("Input", mode_options(embeddings), mode_hint(embeddings)),
                Field::text("Vector/Text", "", "numbers: 0.1, 0.2, … · or text to embed"),
                Field::text("Metadata", "", r#"JSON object, e.g. {"lang": "en"}"#),
            ],
        )
    }

    /// The search form for `collection`. Opens on the query field, since
    /// that is almost always what gets typed first.
    pub fn search(collection: &str, embeddings: bool) -> Self {
        let mut form = Self::build(
            FormKind::Search(collection.to_owned()),
            vec![
                Field::choice("Input", mode_options(embeddings), mode_hint(embeddings)),
                Field::text("Query", "", "numbers: 0.1, 0.2, … · or text to embed"),
                Field::text("k", "10", "number of results"),
                Field::text("Filter", "", r#"JSON, e.g. {"year": {"$gte": 2000}}"#),
            ],
        );
        form.focus = SE_QUERY;
        form
    }

    /// The restore form for snapshot `id`.
    pub fn restore(id: &str) -> Self {
        Self::build(
            FormKind::Restore(id.to_owned()),
            vec![
                Field::text("New name", "", "blank = original collection name"),
                Field {
                    label: "Replace",
                    kind: FieldKind::Toggle(false),
                    hint: "Space: overwrite an existing collection",
                },
            ],
        )
    }

    fn build(kind: FormKind, fields: Vec<Field>) -> Self {
        Self {
            kind,
            fields,
            focus: 0,
            error: None,
        }
    }

    /// Title shown on the modal.
    pub fn title(&self) -> String {
        match &self.kind {
            FormKind::NewCollection => "New collection".into(),
            FormKind::Insert(c) => format!("Insert into {c}"),
            FormKind::Search(c) => format!("Search {c}"),
            FormKind::Restore(id) => format!("Restore snapshot {}", short_id(id)),
        }
    }

    fn text(&self, i: usize) -> &str {
        match &self.fields[i].kind {
            FieldKind::Text(t) => t.value(),
            _ => "",
        }
    }

    fn choice(&self, i: usize) -> &str {
        match &self.fields[i].kind {
            FieldKind::Choice { options, index } => &options[*index],
            _ => "",
        }
    }

    fn toggle(&self, i: usize) -> bool {
        matches!(self.fields[i].kind, FieldKind::Toggle(true))
    }

    /// The "Input" mode field, for forms that have one.
    fn mode_field(&self) -> Option<usize> {
        match self.kind {
            FormKind::Insert(_) => Some(IN_MODE),
            FormKind::Search(_) => Some(SE_MODE),
            _ => None,
        }
    }

    fn cycle(&mut self, i: usize, forward: bool) {
        if let FieldKind::Choice { options, index } = &mut self.fields[i].kind {
            let n = options.len();
            *index = if forward {
                (*index + 1) % n
            } else {
                (*index + n - 1) % n
            };
        }
    }

    fn query(&self, mode: usize, field: usize) -> Result<Query, String> {
        let raw = self.text(field);
        if self.choice(mode) == "text" {
            if raw.trim().is_empty() {
                return Err("enter some text to embed".into());
            }
            Ok(Query::Text(raw.to_owned()))
        } else {
            parse_vector(raw).map(Query::Vector)
        }
    }

    /// Validates the form and turns it into an action.
    pub fn submit(&self) -> Result<Action, String> {
        match &self.kind {
            FormKind::NewCollection => {
                let name = self.text(NC_NAME).trim().to_owned();
                quiver_types::validate_name(&name).map_err(|e| e.to_string())?;
                let dimension: usize = self
                    .text(NC_DIM)
                    .trim()
                    .parse()
                    .map_err(|_| "dimension must be a positive whole number".to_owned())?;
                let metric = Metric::ALL
                    .into_iter()
                    .find(|m| m.as_str() == self.choice(NC_METRIC))
                    .unwrap_or_default();
                let index = match self.choice(NC_INDEX) {
                    "flat" => IndexConfig::Flat,
                    "kd_tree" => IndexConfig::KdTree(KdTreeParams::default()),
                    _ => IndexConfig::Hnsw(HnswParams::default()),
                };
                let config = CollectionConfig::new(dimension).metric(metric).index(index);
                config.validate().map_err(|e| e.to_string())?;
                Ok(Action::CreateCollection { name, config })
            }
            FormKind::Insert(collection) => {
                let id = self.text(IN_ID).trim();
                Ok(Action::Insert {
                    collection: collection.clone(),
                    id: (!id.is_empty()).then(|| id.to_owned()),
                    query: self.query(IN_MODE, IN_QUERY)?,
                    metadata: parse_json_object(self.text(IN_META)).map_err(|e| format!("metadata {e}"))?,
                })
            }
            FormKind::Search(collection) => {
                let k: usize = self
                    .text(SE_K)
                    .trim()
                    .parse()
                    .ok()
                    .filter(|k| (1..=10_000).contains(k))
                    .ok_or("k must be between 1 and 10000")?;
                Ok(Action::Search {
                    collection: collection.clone(),
                    query: self.query(SE_MODE, SE_QUERY)?,
                    k,
                    filter: parse_json_object(self.text(SE_FILTER)).map_err(|e| format!("filter {e}"))?,
                })
            }
            FormKind::Restore(id) => {
                let name = self.text(RE_NAME).trim();
                if !name.is_empty() {
                    quiver_types::validate_name(name).map_err(|e| e.to_string())?;
                }
                Ok(Action::RestoreSnapshot {
                    id: id.clone(),
                    name: (!name.is_empty()).then(|| name.to_owned()),
                    replace: self.toggle(RE_REPLACE),
                })
            }
        }
    }
}

fn mode_hint(embeddings: bool) -> &'static str {
    if embeddings {
        "←/→ or Ctrl-T: type numbers or text to embed"
    } else {
        "text input needs --embedding-url (QUIVER_EMBEDDING_URL)"
    }
}

/// First 8 characters of an id, for compact tables.
pub fn short_id(id: &str) -> &str {
    id.char_indices().nth(8).map_or(id, |(i, _)| &id[..i])
}

/// An overlay on top of the current screen.
#[derive(Debug, Clone, PartialEq)]
pub enum Modal {
    /// Key reference.
    Help,
    /// Yes/no question guarding a destructive action.
    Confirm {
        /// The question.
        message: String,
        /// What `y` does.
        action: Action,
    },
    /// An input form.
    Form(Form),
    /// Read-only text (point details).
    Detail {
        /// Title.
        title: String,
        /// Body.
        body: String,
        /// Vertical scroll offset.
        scroll: u16,
    },
}

/// What the collection screen lists.
#[derive(Debug, Clone, PartialEq)]
pub enum ListMode {
    /// Paged points.
    Points,
    /// Results of the last search.
    Results(Vec<SearchHit>),
}

/// The whole UI state.
#[derive(Debug, Clone)]
pub struct App {
    /// Server base URL, for display.
    pub url: String,
    /// Whether text can be embedded.
    pub embeddings: bool,
    /// Current screen.
    pub screen: Screen,
    /// Connection state.
    pub server: ServerState,
    /// Collections on the dashboard.
    pub collections: Vec<CollectionInfo>,
    /// Selected dashboard row.
    pub selected: usize,
    /// Open collection.
    pub current: Option<String>,
    /// Current page of points.
    pub points: Vec<Point>,
    /// Offset of the current page.
    pub offset: usize,
    /// Offset of the next page, if any.
    pub next_offset: Option<usize>,
    /// Points or search results.
    pub list: ListMode,
    /// Selected row in the collection view.
    pub row: usize,
    /// Snapshots.
    pub snapshots: Vec<SnapshotInfo>,
    /// Selected snapshot.
    pub snapshot_row: usize,
    /// Open overlay.
    pub modal: Option<Modal>,
    /// Status bar message.
    pub status: Option<Status>,
    /// Set when the user asked to quit.
    pub should_quit: bool,
}

fn step(index: &mut usize, len: usize, down: bool) {
    if len == 0 {
        *index = 0;
    } else if down {
        *index = (*index + 1).min(len - 1);
    } else {
        *index = index.saturating_sub(1);
    }
}

impl App {
    /// Fresh state for the server at `url`.
    pub fn new(url: &str, embeddings: bool) -> Self {
        Self {
            url: url.to_owned(),
            embeddings,
            screen: Screen::Dashboard,
            server: ServerState::Unknown,
            collections: Vec::new(),
            selected: 0,
            current: None,
            points: Vec::new(),
            offset: 0,
            next_offset: None,
            list: ListMode::Points,
            row: 0,
            snapshots: Vec::new(),
            snapshot_row: 0,
            modal: None,
            status: None,
            should_quit: false,
        }
    }

    /// Shows an informational status message.
    pub fn info(&mut self, text: impl Into<String>) {
        self.status = Some(Status {
            text: text.into(),
            error: false,
        });
    }

    /// Shows an error. If a form is open, the error appears in the form.
    pub fn error(&mut self, text: impl Into<String>) {
        let text = text.into();
        if let Some(Modal::Form(form)) = &mut self.modal {
            form.error = Some(text);
        } else {
            self.status = Some(Status { text, error: true });
        }
    }

    /// The collection selected on the dashboard.
    pub fn selected_collection(&self) -> Option<&CollectionInfo> {
        self.collections.get(self.selected)
    }

    /// Number of rows in the collection view's list.
    fn list_len(&self) -> usize {
        match &self.list {
            ListMode::Points => self.points.len(),
            ListMode::Results(hits) => hits.len(),
        }
    }

    /// Id (and distance, for results) of the selected collection-view row.
    pub fn selected_row(&self) -> Option<(String, Option<f32>)> {
        match &self.list {
            ListMode::Points => self.points.get(self.row).map(|p| (p.id.clone(), None)),
            ListMode::Results(hits) => hits.get(self.row).map(|h| (h.id.clone(), Some(h.distance))),
        }
    }

    /// Handles a key press; returns an action to perform, if any.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<Action> {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.should_quit = true;
            return None;
        }
        if self.modal.is_some() {
            return self.handle_modal_key(key);
        }
        match key.code {
            KeyCode::Char('q') => {
                self.should_quit = true;
                return None;
            }
            KeyCode::Char('?') => {
                self.modal = Some(Modal::Help);
                return None;
            }
            KeyCode::Char('r') => return Some(Action::Refresh),
            _ => {}
        }
        match self.screen {
            Screen::Dashboard => self.dashboard_key(key),
            Screen::Collection => self.collection_key(key),
            Screen::Snapshots => self.snapshots_key(key),
        }
    }

    fn dashboard_key(&mut self, key: KeyEvent) -> Option<Action> {
        match key.code {
            KeyCode::Esc => self.should_quit = true,
            KeyCode::Up | KeyCode::Char('k') => step(&mut self.selected, self.collections.len(), false),
            KeyCode::Down | KeyCode::Char('j') => step(&mut self.selected, self.collections.len(), true),
            KeyCode::Enter => {
                return self
                    .selected_collection()
                    .map(|c| Action::OpenCollection(c.name.clone()));
            }
            KeyCode::Char('n') => self.modal = Some(Modal::Form(Form::new_collection())),
            KeyCode::Char('d') => {
                if let Some(c) = self.selected_collection() {
                    self.modal = Some(Modal::Confirm {
                        message: format!("Drop collection `{}` and its {} points?", c.name, c.count),
                        action: Action::DropCollection(c.name.clone()),
                    });
                }
            }
            KeyCode::Char('s') => {
                self.screen = Screen::Snapshots;
                return Some(Action::LoadSnapshots);
            }
            _ => {}
        }
        None
    }

    fn collection_key(&mut self, key: KeyEvent) -> Option<Action> {
        let collection = self.current.clone()?;
        let len = self.list_len();
        match key.code {
            KeyCode::Esc => {
                if matches!(self.list, ListMode::Results(_)) {
                    self.list = ListMode::Points;
                    self.row = 0;
                } else {
                    self.screen = Screen::Dashboard;
                    self.current = None;
                }
            }
            KeyCode::Up | KeyCode::Char('k') => step(&mut self.row, len, false),
            KeyCode::Down | KeyCode::Char('j') => step(&mut self.row, len, true),
            KeyCode::PageDown | KeyCode::Right if self.list == ListMode::Points => {
                return self
                    .next_offset
                    .map(|offset| Action::LoadPoints { collection, offset });
            }
            KeyCode::PageUp | KeyCode::Left if self.list == ListMode::Points && self.offset > 0 => {
                let offset = self.offset.saturating_sub(PAGE_SIZE);
                return Some(Action::LoadPoints { collection, offset });
            }
            KeyCode::Enter => {
                return self.selected_row().map(|(id, distance)| Action::ShowPoint {
                    collection,
                    id,
                    distance,
                });
            }
            KeyCode::Char('i') => {
                self.modal = Some(Modal::Form(Form::insert(&collection, self.embeddings)));
            }
            KeyCode::Char('/') => {
                self.modal = Some(Modal::Form(Form::search(&collection, self.embeddings)));
            }
            KeyCode::Char('x') => {
                if let Some((id, _)) = self.selected_row() {
                    self.modal = Some(Modal::Confirm {
                        message: format!("Delete point `{id}`?"),
                        action: Action::DeletePoint { collection, id },
                    });
                }
            }
            KeyCode::Char('S') => return Some(Action::CreateSnapshot(collection)),
            _ => {}
        }
        None
    }

    fn snapshots_key(&mut self, key: KeyEvent) -> Option<Action> {
        let selected = self
            .snapshots
            .get(self.snapshot_row)
            .map(|s| s.manifest.id.clone());
        match key.code {
            KeyCode::Esc => self.screen = Screen::Dashboard,
            KeyCode::Up | KeyCode::Char('k') => step(&mut self.snapshot_row, self.snapshots.len(), false),
            KeyCode::Down | KeyCode::Char('j') => step(&mut self.snapshot_row, self.snapshots.len(), true),
            KeyCode::Char('c') => match self.selected_collection() {
                Some(c) => return Some(Action::CreateSnapshot(c.name.clone())),
                None => self.error("select a collection on the dashboard first"),
            },
            KeyCode::Char('R') => {
                if let Some(id) = selected {
                    self.modal = Some(Modal::Form(Form::restore(&id)));
                }
            }
            KeyCode::Char('D') => {
                if let Some(id) = selected {
                    self.modal = Some(Modal::Confirm {
                        message: format!("Delete snapshot {}?", short_id(&id)),
                        action: Action::DeleteSnapshot(id),
                    });
                }
            }
            _ => {}
        }
        None
    }

    fn handle_modal_key(&mut self, key: KeyEvent) -> Option<Action> {
        let modal = self.modal.as_mut()?;
        match modal {
            Modal::Help => {
                self.modal = None;
                None
            }
            Modal::Confirm { action, .. } => {
                let action = action.clone();
                match key.code {
                    KeyCode::Char('y') | KeyCode::Char('Y') => {
                        self.modal = None;
                        Some(action)
                    }
                    KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                        self.modal = None;
                        None
                    }
                    _ => None,
                }
            }
            Modal::Detail { scroll, .. } => {
                match key.code {
                    KeyCode::Up | KeyCode::Char('k') => *scroll = scroll.saturating_sub(1),
                    KeyCode::Down | KeyCode::Char('j') => *scroll = scroll.saturating_add(1),
                    KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => self.modal = None,
                    _ => {}
                }
                None
            }
            Modal::Form(form) => {
                let embeddings = self.embeddings;
                match form_key(form, key, embeddings) {
                    FormOutcome::Continue => None,
                    FormOutcome::Cancel => {
                        self.modal = None;
                        None
                    }
                    FormOutcome::Submit => match form.submit() {
                        Ok(action) => {
                            form.error = None;
                            Some(action)
                        }
                        Err(e) => {
                            form.error = Some(e);
                            None
                        }
                    },
                }
            }
        }
    }
}

enum FormOutcome {
    Continue,
    Cancel,
    Submit,
}

fn form_key(form: &mut Form, key: KeyEvent, embeddings: bool) -> FormOutcome {
    let n = form.fields.len();
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('t') {
        match form.mode_field() {
            Some(i) if embeddings => form.cycle(i, true),
            Some(_) => form.error = Some(mode_hint(false).into()),
            None => {}
        }
        return FormOutcome::Continue;
    }
    match key.code {
        KeyCode::Esc => return FormOutcome::Cancel,
        KeyCode::Enter => return FormOutcome::Submit,
        KeyCode::Tab | KeyCode::Down => form.focus = (form.focus + 1) % n,
        KeyCode::BackTab | KeyCode::Up => form.focus = (form.focus + n - 1) % n,
        _ => {
            let focus = form.focus;
            match &mut form.fields[focus].kind {
                FieldKind::Text(t) => match key.code {
                    KeyCode::Char(c) => t.insert(c),
                    KeyCode::Backspace => t.backspace(),
                    KeyCode::Delete => t.delete(),
                    KeyCode::Left => t.left(),
                    KeyCode::Right => t.right(),
                    KeyCode::Home => t.home(),
                    KeyCode::End => t.end(),
                    _ => {}
                },
                FieldKind::Choice { .. } => match key.code {
                    KeyCode::Left => form.cycle(focus, false),
                    KeyCode::Right | KeyCode::Char(' ') => form.cycle(focus, true),
                    _ => {}
                },
                FieldKind::Toggle(on) => {
                    if key.code == KeyCode::Char(' ') {
                        *on = !*on;
                    }
                }
            }
        }
    }
    FormOutcome::Continue
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn press(app: &mut App, code: KeyCode) -> Option<Action> {
        app.handle_key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    pub(crate) fn type_str(app: &mut App, s: &str) {
        for c in s.chars() {
            press(app, KeyCode::Char(c));
        }
    }

    pub(crate) fn collection(name: &str, count: usize) -> CollectionInfo {
        CollectionInfo {
            name: name.into(),
            config: CollectionConfig::new(3),
            count,
            tombstones: 0,
        }
    }

    fn form(app: &App) -> &Form {
        match &app.modal {
            Some(Modal::Form(f)) => f,
            other => panic!("expected a form, got {other:?}"),
        }
    }

    #[test]
    fn new_collection_form_flow() {
        let mut app = App::new("http://x", false);
        press(&mut app, KeyCode::Char('n'));
        assert_eq!(form(&app).kind, FormKind::NewCollection);

        type_str(&mut app, "docs");
        press(&mut app, KeyCode::Tab);
        type_str(&mut app, "4");
        press(&mut app, KeyCode::Tab);
        press(&mut app, KeyCode::Right); // cosine -> euclidean
        press(&mut app, KeyCode::Tab);
        press(&mut app, KeyCode::Left); // hnsw -> kd_tree (wraps)
        assert_eq!(form(&app).focus, NC_INDEX);

        let action = press(&mut app, KeyCode::Enter).unwrap();
        let Action::CreateCollection { name, config } = action else {
            panic!("unexpected action")
        };
        assert_eq!(name, "docs");
        assert_eq!(config.dimension, 4);
        assert_eq!(config.metric, Metric::Euclidean);
        assert!(matches!(config.index, IndexConfig::KdTree(_)));
    }

    #[test]
    fn invalid_forms_show_errors_and_esc_cancels() {
        let mut app = App::new("http://x", false);
        press(&mut app, KeyCode::Char('n'));
        type_str(&mut app, "bad name");
        assert!(press(&mut app, KeyCode::Enter).is_none());
        assert!(form(&app).error.as_deref().unwrap().contains("invalid name"));
        press(&mut app, KeyCode::BackTab);
        assert_eq!(form(&app).focus, NC_INDEX, "shift-tab wraps");
        press(&mut app, KeyCode::Esc);
        assert!(app.modal.is_none());
    }

    #[test]
    fn insert_and_search_forms() {
        let mut app = App::new("http://x", false);
        app.screen = Screen::Collection;
        app.current = Some("docs".into());

        press(&mut app, KeyCode::Char('i'));
        press(&mut app, KeyCode::Tab);
        press(&mut app, KeyCode::Tab);
        type_str(&mut app, "1, 2, 3");
        press(&mut app, KeyCode::Tab);
        type_str(&mut app, r#"{"a":1}"#);
        let action = press(&mut app, KeyCode::Enter).unwrap();
        assert_eq!(
            action,
            Action::Insert {
                collection: "docs".into(),
                id: None,
                query: Query::Vector(vec![1.0, 2.0, 3.0]),
                metadata: Some(serde_json::json!({"a": 1})),
            }
        );

        app.modal = None;
        press(&mut app, KeyCode::Char('/'));
        assert_eq!(form(&app).focus, SE_QUERY, "search opens on the query field");
        type_str(&mut app, "0 1 0");
        let Some(Action::Search { k, query, .. }) = press(&mut app, KeyCode::Enter) else {
            panic!("expected search")
        };
        assert_eq!((k, query), (10, Query::Vector(vec![0.0, 1.0, 0.0])));
    }

    #[test]
    fn ctrl_t_switches_to_text_only_with_embeddings() {
        let ctrl_t = KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL);

        let mut app = App::new("http://x", false);
        app.screen = Screen::Collection;
        app.current = Some("docs".into());
        press(&mut app, KeyCode::Char('/'));
        app.handle_key(ctrl_t);
        assert!(form(&app).error.as_deref().unwrap().contains("--embedding-url"));

        let mut app = App::new("http://x", true);
        app.screen = Screen::Collection;
        app.current = Some("docs".into());
        press(&mut app, KeyCode::Char('/'));
        app.handle_key(ctrl_t);
        type_str(&mut app, "space opera");
        let Some(Action::Search { query, .. }) = press(&mut app, KeyCode::Enter) else {
            panic!("expected search")
        };
        assert_eq!(query, Query::Text("space opera".into()));
    }

    #[test]
    fn navigation_and_confirmation() {
        let mut app = App::new("http://x", false);
        app.collections = vec![collection("a", 1), collection("b", 2)];
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Down);
        assert_eq!(app.selected, 1, "selection is clamped");
        assert_eq!(
            press(&mut app, KeyCode::Enter),
            Some(Action::OpenCollection("b".into()))
        );

        press(&mut app, KeyCode::Char('d'));
        assert!(matches!(app.modal, Some(Modal::Confirm { .. })));
        assert_eq!(press(&mut app, KeyCode::Char('n')), None);
        assert!(app.modal.is_none());
        press(&mut app, KeyCode::Char('d'));
        assert_eq!(
            press(&mut app, KeyCode::Char('y')),
            Some(Action::DropCollection("b".into()))
        );

        assert_eq!(press(&mut app, KeyCode::Char('s')), Some(Action::LoadSnapshots));
        assert_eq!(app.screen, Screen::Snapshots);
        assert_eq!(
            press(&mut app, KeyCode::Char('c')),
            Some(Action::CreateSnapshot("b".into()))
        );
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.screen, Screen::Dashboard);
        press(&mut app, KeyCode::Char('?'));
        assert_eq!(app.modal, Some(Modal::Help));
        press(&mut app, KeyCode::Char('x'));
        assert!(app.modal.is_none(), "any key closes help");
        press(&mut app, KeyCode::Char('q'));
        assert!(app.should_quit);
    }

    #[test]
    fn restore_form() {
        let mut f = Form::restore("0123456789abcdef");
        assert_eq!(f.title(), "Restore snapshot 01234567");
        if let FieldKind::Toggle(on) = &mut f.fields[RE_REPLACE].kind {
            *on = true;
        }
        assert_eq!(
            f.submit().unwrap(),
            Action::RestoreSnapshot {
                id: "0123456789abcdef".into(),
                name: None,
                replace: true
            }
        );
    }
}
