use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Cell, Clear, Paragraph, Row, Table, TableState, Wrap};

use crate::app::{App, FieldKind, Form, ListMode, Modal, PAGE_SIZE, Screen, ServerState, short_id};
use crate::input::{format_bytes, format_timestamp, preview_json, truncate};

const ACCENT: Color = Color::Rgb(249, 115, 22);

fn highlight() -> Style {
    Style::new()
        .bg(Color::Rgb(49, 46, 129))
        .add_modifier(Modifier::BOLD)
}

fn block(title: String) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .title(Span::styled(format!(" {title} "), Style::new().fg(ACCENT).bold()))
}

/// Draws the whole UI.
pub fn draw(frame: &mut Frame, app: &App) {
    let [header, body, status, keys] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(3),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    draw_header(frame, header, app);
    match app.screen {
        Screen::Dashboard => draw_dashboard(frame, body, app),
        Screen::Collection => draw_collection(frame, body, app),
        Screen::Snapshots => draw_snapshots(frame, body, app),
    }
    if let Some(s) = &app.status {
        let style = if s.error {
            Style::new().fg(Color::Red)
        } else {
            Style::new().fg(Color::Green)
        };
        frame.render_widget(
            Paragraph::new(Span::styled(format!(" {}", s.text), style)),
            status,
        );
    }
    frame.render_widget(Paragraph::new(key_hints(app)).dark_gray(), keys);

    match &app.modal {
        None => {}
        Some(Modal::Help) => draw_help(frame),
        Some(Modal::Confirm { message, .. }) => {
            let area = centered(frame.area(), 60, 5);
            frame.render_widget(Clear, area);
            let text = vec![
                Line::from(message.as_str()),
                Line::from(""),
                Line::from("y = yes · n = no").dark_gray(),
            ];
            frame.render_widget(
                Paragraph::new(text)
                    .wrap(Wrap { trim: true })
                    .block(block("Confirm".into())),
                area,
            );
        }
        Some(Modal::Detail { title, body, scroll }) => {
            let r = frame.area();
            let area = centered(r, r.width.saturating_sub(8).min(100), r.height.saturating_sub(4));
            frame.render_widget(Clear, area);
            frame.render_widget(
                Paragraph::new(body.as_str())
                    .wrap(Wrap { trim: false })
                    .scroll((*scroll, 0))
                    .block(block(title.clone()).title_bottom(" ↑/↓ scroll · Esc close ")),
                area,
            );
        }
        Some(Modal::Form(form)) => draw_form(frame, form),
    }
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App) {
    let line = match &app.server {
        ServerState::Unknown => Line::from(format!("connecting to {} …", app.url)),
        ServerState::Unreachable(e) => Line::from(vec![
            Span::styled("cannot reach server ", Style::new().fg(Color::Red).bold()),
            Span::raw(format!("at {} — press r to retry", app.url)),
            Span::raw(format!(" ({})", truncate(e, 60))).dark_gray(),
        ]),
        ServerState::Online(info) => Line::from(vec![
            Span::styled("● ", Style::new().fg(Color::Green)),
            Span::raw(format!("{}  ", app.url)),
            Span::raw(format!("v{}  ", info.version)).dark_gray(),
            Span::raw(format!("storage: {}  ", info.storage)),
            Span::raw(format!(
                "snapshots: {}  ",
                if info.snapshots_enabled { "on" } else { "off" }
            )),
            Span::raw(format!(
                "{} collections · {} points",
                info.collections, info.points
            )),
        ]),
    };
    frame.render_widget(Paragraph::new(line).block(block("Quiver".into())), area);
}

/// The named vectors besides the main one, sparse ones marked.
fn extra_vectors(config: &quiver_types::CollectionConfig) -> String {
    let dense = config.vectors.keys().map(String::from);
    let sparse = config.sparse_vectors.keys().map(|n| format!("{n} (sparse)"));
    dense.chain(sparse).collect::<Vec<_>>().join(", ")
}

fn draw_dashboard(frame: &mut Frame, area: Rect, app: &App) {
    let rows = app.collections.iter().map(|c| {
        Row::new(vec![
            Cell::from(c.name.clone()),
            Cell::from(c.config.dimension.to_string()),
            Cell::from(c.config.metric.as_str()),
            Cell::from(c.config.index.kind()),
            Cell::from(c.count.to_string()),
            Cell::from(c.tombstones.to_string()),
            Cell::from(extra_vectors(&c.config)),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Min(16),
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Min(12),
        ],
    )
    .header(
        Row::new([
            "name",
            "dimension",
            "metric",
            "index",
            "points",
            "tombstones",
            "other vectors",
        ])
        .bold(),
    )
    .block(block(format!("Collections ({})", app.collections.len())))
    .row_highlight_style(highlight())
    .highlight_symbol("▶ ");
    if app.collections.is_empty() {
        let hint = match app.server {
            ServerState::Unreachable(_) => "No connection. Start the server (`quiver`) and press r.",
            _ => "No collections yet. Press n to create one.",
        };
        frame.render_widget(
            Paragraph::new(hint)
                .dark_gray()
                .block(block("Collections".into())),
            area,
        );
        return;
    }
    let mut state = TableState::default().with_selected(Some(app.selected));
    frame.render_stateful_widget(table, area, &mut state);
}

fn draw_collection(frame: &mut Frame, area: Rect, app: &App) {
    let name = app.current.as_deref().unwrap_or("?");
    let (title, rows, widths, header): (String, Vec<Row>, Vec<Constraint>, Row) = match &app.list {
        ListMode::Points => {
            let start = app.page_index() * PAGE_SIZE;
            let end = start + app.points.len();
            let more = if app.next_cursor.is_some() { "+" } else { "" };
            (
                format!(
                    "{name} · points {}–{end}{more}",
                    if end == start { start } else { start + 1 }
                ),
                app.points
                    .iter()
                    .map(|p| Row::new(vec![p.id.clone(), preview_json(p.metadata.as_ref(), 120)]))
                    .collect(),
                vec![Constraint::Length(38), Constraint::Min(10)],
                Row::new(["id", "metadata"]).bold(),
            )
        }
        ListMode::Results(hits) => (
            format!("{name} · {} search results", hits.len()),
            hits.iter()
                .enumerate()
                .map(|(i, h)| {
                    Row::new(vec![
                        (i + 1).to_string(),
                        h.id.clone(),
                        format!("{:.5}", h.distance),
                        preview_json(h.metadata.as_ref(), 100),
                    ])
                })
                .collect(),
            vec![
                Constraint::Length(5),
                Constraint::Length(38),
                Constraint::Length(10),
                Constraint::Min(10),
            ],
            Row::new(["#", "id", "distance", "metadata"]).bold(),
        ),
    };
    if rows.is_empty() {
        let hint = match app.list {
            ListMode::Points => "No points. Press i to insert one.",
            ListMode::Results(_) => "No matches. Press / to search again, Esc for points.",
        };
        frame.render_widget(Paragraph::new(hint).dark_gray().block(block(title)), area);
        return;
    }
    let table = Table::new(rows, widths)
        .header(header)
        .block(block(title))
        .row_highlight_style(highlight())
        .highlight_symbol("▶ ");
    let mut state = TableState::default().with_selected(Some(app.row));
    frame.render_stateful_widget(table, area, &mut state);
}

fn draw_snapshots(frame: &mut Frame, area: Rect, app: &App) {
    let title = format!("Snapshots ({})", app.snapshots.len());
    if app.snapshots.is_empty() {
        let hint = "No snapshots. Press c to snapshot the collection selected on the dashboard.";
        frame.render_widget(Paragraph::new(hint).dark_gray().block(block(title)), area);
        return;
    }
    let rows = app.snapshots.iter().map(|s| {
        Row::new(vec![
            short_id(&s.manifest.id).to_owned(),
            s.manifest.collection.clone(),
            s.manifest.points.to_string(),
            format_timestamp(s.manifest.created_at_ms),
            format_bytes(s.archive_bytes),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Min(14),
            Constraint::Length(10),
            Constraint::Length(24),
            Constraint::Length(11),
        ],
    )
    .header(Row::new(["id", "collection", "points", "created", "size"]).bold())
    .block(block(title))
    .row_highlight_style(highlight())
    .highlight_symbol("▶ ");
    let mut state = TableState::default().with_selected(Some(app.snapshot_row));
    frame.render_stateful_widget(table, area, &mut state);
}

fn draw_form(frame: &mut Frame, form: &Form) {
    let height = form.fields.len() as u16 * 2 + 6;
    let area = centered(frame.area(), 76, height);
    frame.render_widget(Clear, area);
    let outer = block(form.title()).title_bottom(" Tab/↑↓ move · Enter submit · Esc cancel ");
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    let label_width = 13u16;
    let mut lines = Vec::new();
    let mut cursor = None;
    for (i, field) in form.fields.iter().enumerate() {
        let focused = i == form.focus;
        let label = Span::styled(
            format!("{:>12} ", field.label),
            if focused {
                Style::new().fg(ACCENT).bold()
            } else {
                Style::new()
            },
        );
        let value = match &field.kind {
            FieldKind::Text(t) => {
                if focused {
                    // Columns, not characters: wide characters take two cells.
                    let before: String = t.value().chars().take(t.cursor()).collect();
                    let width = u16::try_from(Span::raw(before).width()).unwrap_or(u16::MAX);
                    cursor = Some((
                        inner.x.saturating_add(label_width).saturating_add(width),
                        inner.y + lines.len() as u16,
                    ));
                }
                Span::raw(t.value().to_owned())
            }
            FieldKind::Choice { options, index } => Span::raw(format!("◀ {} ▶", options[*index])),
            FieldKind::Toggle(on) => Span::raw(if *on { "[x]" } else { "[ ]" }),
        };
        let value = if focused { value.underlined() } else { value };
        lines.push(Line::from(vec![label, value]));
        lines.push(Line::from(""));
    }
    let hint = form.fields.get(form.focus).map_or("", |f| f.hint);
    lines.push(Line::from(hint).dark_gray());
    if let Some(e) = &form.error {
        lines.push(Line::from(Span::styled(e.clone(), Style::new().fg(Color::Red))));
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    if let Some((x, y)) = cursor {
        frame.set_cursor_position((x.min(inner.right().saturating_sub(1)), y));
    }
}

const HELP: &[(&str, &str)] = &[
    ("Anywhere", ""),
    ("?", "this help"),
    ("r", "refresh"),
    ("q / Ctrl-C", "quit"),
    ("Esc", "back / close"),
    ("Dashboard", ""),
    ("↑↓ / j k", "select collection"),
    ("Enter", "open collection"),
    ("n", "new collection"),
    ("d", "drop collection"),
    ("s", "snapshots"),
    ("Collection", ""),
    ("PgUp/PgDn ←→", "previous / next page"),
    ("Enter", "point details"),
    ("i", "insert point"),
    ("/", "search"),
    ("x", "delete point"),
    ("S", "snapshot this collection"),
    ("Snapshots", ""),
    ("c", "snapshot selected collection"),
    ("R", "restore"),
    ("D", "delete"),
    ("Forms", ""),
    ("Ctrl-T", "toggle vector / text input"),
];

fn draw_help(frame: &mut Frame) {
    let area = centered(frame.area(), 56, HELP.len() as u16 + 2);
    frame.render_widget(Clear, area);
    let lines: Vec<Line> = HELP
        .iter()
        .map(|(k, v)| {
            if v.is_empty() {
                Line::from(Span::styled(*k, Style::new().fg(ACCENT).bold()))
            } else {
                Line::from(vec![Span::raw(format!("  {k:<16}")).bold(), Span::raw(*v)])
            }
        })
        .collect();
    frame.render_widget(Paragraph::new(lines).block(block("Keys".into())), area);
}

fn key_hints(app: &App) -> &'static str {
    match app.screen {
        Screen::Dashboard => " Enter open · n new · d drop · s snapshots · r refresh · ? help · q quit",
        Screen::Collection => {
            " Enter details · i insert · / search · x delete · S snapshot · ←→ page · Esc back"
        }
        Screen::Snapshots => " c create · R restore · D delete · r refresh · Esc back",
    }
}

/// A `width × height` rectangle centred in `area` (clamped to fit).
fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let w = width.min(area.width);
    let h = height.min(area.height);
    Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests::{collection, press};
    use quiver_client::ServerInfo;
    use quiver_types::Point;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::KeyCode;

    fn render(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(110, 34)).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
        let buf = terminal.backend().buffer();
        (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn dashboard_lists_collections() {
        let mut app = App::new("http://localhost:6900", false);
        app.server = ServerState::Online(ServerInfo {
            version: "0.2.0".into(),
            storage: "file".into(),
            snapshots_enabled: true,
            collections: 2,
            points: 30,
        });
        app.collections = vec![collection("movies", 10), collection("books", 20)];
        let screen = render(&app);
        assert!(screen.contains("movies"));
        assert!(screen.contains("books"));
        assert!(screen.contains("storage: file"));
        assert!(screen.contains("2 collections · 30 points"));
    }

    #[test]
    fn unreachable_server_is_shown() {
        let mut app = App::new("http://127.0.0.1:9", false);
        app.server = ServerState::Unreachable("connection refused".into());
        let screen = render(&app);
        assert!(screen.contains("cannot reach server"));
        assert!(screen.contains("press r to retry"));
    }

    #[test]
    fn help_and_forms_render() {
        let mut app = App::new("http://x", false);
        press(&mut app, KeyCode::Char('?'));
        assert!(render(&app).contains("snapshot this collection"));
        app.modal = None;
        press(&mut app, KeyCode::Char('n'));
        let screen = render(&app);
        assert!(screen.contains("New collection"));
        assert!(screen.contains("◀ cosine ▶"));
    }

    #[test]
    fn form_cursor_follows_wide_characters() {
        let mut app = App::new("http://x", false);
        app.screen = Screen::Collection;
        app.current = Some("docs".into());
        press(&mut app, KeyCode::Char('i'));
        for c in "日本".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        let mut terminal = Terminal::new(TestBackend::new(110, 34)).unwrap();
        terminal.draw(|f| draw(f, &app)).unwrap();
        let buf = terminal.backend().buffer();
        let last = (0..buf.area.height)
            .flat_map(|y| (0..buf.area.width).map(move |x| (x, y)))
            .find(|&(x, y)| buf[(x, y)].symbol() == "本")
            .unwrap();
        let cursor = terminal.get_cursor_position().unwrap();
        assert_eq!(
            (cursor.x, cursor.y),
            (last.0 + 2, last.1),
            "cursor sits after the text"
        );
    }

    #[test]
    fn points_title_counts_pages() {
        let mut app = App::new("http://x", false);
        app.screen = Screen::Collection;
        app.current = Some("docs".into());
        app.cursors = vec![None, Some("p049".into()), Some("p099".into())];
        app.points = (100..105).map(|i| Point::new(format!("p{i}"), vec![])).collect();
        assert!(render(&app).contains("docs · points 101–105 "));
        app.next_cursor = Some("p104".into());
        assert!(render(&app).contains("docs · points 101–105+"));
        app.points.clear();
        app.cursors.truncate(1);
        app.next_cursor = None;
        assert!(render(&app).contains("docs · points 0–0 "));
    }

    #[test]
    fn tiny_terminals_do_not_panic() {
        let mut app = App::new("http://x", false);
        app.collections = vec![collection("a", 1)];
        press(&mut app, KeyCode::Char('n'));
        let mut terminal = Terminal::new(TestBackend::new(20, 6)).unwrap();
        terminal.draw(|f| draw(f, &app)).unwrap();
    }
}
