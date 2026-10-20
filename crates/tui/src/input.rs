use serde_json::Value;

/// A single-line text field with a cursor (counted in characters).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextField {
    value: String,
    cursor: usize,
}

impl TextField {
    /// A field pre-filled with `value`, cursor at the end.
    pub fn new(value: &str) -> Self {
        Self {
            value: value.to_owned(),
            cursor: value.chars().count(),
        }
    }

    /// The current text.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Cursor position in characters.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    fn byte_index(&self, chars: usize) -> usize {
        self.value
            .char_indices()
            .nth(chars)
            .map_or(self.value.len(), |(i, _)| i)
    }

    /// Inserts a character at the cursor.
    pub fn insert(&mut self, c: char) {
        let at = self.byte_index(self.cursor);
        self.value.insert(at, c);
        self.cursor += 1;
    }

    /// Deletes the character before the cursor.
    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            let at = self.byte_index(self.cursor);
            self.value.remove(at);
        }
    }

    /// Deletes the character under the cursor.
    pub fn delete(&mut self) {
        if self.cursor < self.value.chars().count() {
            let at = self.byte_index(self.cursor);
            self.value.remove(at);
        }
    }

    /// Moves the cursor one character left.
    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    /// Moves the cursor one character right.
    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.value.chars().count());
    }

    /// Moves the cursor to the start.
    pub fn home(&mut self) {
        self.cursor = 0;
    }

    /// Moves the cursor to the end.
    pub fn end(&mut self) {
        self.cursor = self.value.chars().count();
    }
}

/// Parses a vector typed as numbers separated by commas and/or whitespace,
/// optionally wrapped in brackets: `0.1, 0.2 0.3` or `[1, 2, 3]`.
pub fn parse_vector(input: &str) -> Result<Vec<f32>, String> {
    let trimmed = input.trim();
    let inner = trimmed
        .strip_prefix('[')
        .map(|s| s.strip_suffix(']').unwrap_or(s))
        .unwrap_or(trimmed);
    let values = inner
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| match s.parse::<f32>() {
            Ok(x) if x.is_finite() => Ok(x),
            _ => Err(format!("`{s}` is not a finite number")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("enter at least one number".into());
    }
    Ok(values)
}

/// Parses optional JSON that must be an object (metadata, filters).
/// Blank input means "none".
pub fn parse_json_object(input: &str) -> Result<Option<Value>, String> {
    if input.trim().is_empty() {
        return Ok(None);
    }
    match serde_json::from_str::<Value>(input) {
        Ok(v @ Value::Object(_)) => Ok(Some(v)),
        Ok(_) => Err("must be a JSON object, e.g. {\"lang\": \"en\"}".into()),
        Err(e) => Err(format!("invalid JSON: {e}")),
    }
}

/// Compact single-line JSON, cut to `max` characters.
pub fn preview_json(value: Option<&Value>, max: usize) -> String {
    match value {
        None => String::new(),
        Some(v) => truncate(&v.to_string(), max),
    }
}

/// Cuts `s` to at most `max` characters, ending with `…` if shortened.
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Formats milliseconds since the Unix epoch as `YYYY-MM-DD HH:MM:SS UTC`.
pub fn format_timestamp(ms: u64) -> String {
    let secs = ms / 1000;
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Days since 1970-01-01 → (year, month, day) in the proleptic Gregorian
/// calendar (Howard Hinnant's `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// Human-readable byte size.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn vectors_parse_in_many_shapes() {
        assert_eq!(parse_vector("1, 2, 3").unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(parse_vector(" 0.5 -1e2\t4 ").unwrap(), vec![0.5, -100.0, 4.0]);
        assert_eq!(parse_vector("[1,2]").unwrap(), vec![1.0, 2.0]);
        assert_eq!(parse_vector("[ 1 , 2 , ]").unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn bad_vectors_are_rejected() {
        assert!(parse_vector("").is_err());
        assert!(parse_vector("[]").is_err());
        assert!(parse_vector("1, two").unwrap_err().contains("two"));
        assert!(parse_vector("NaN").is_err());
        assert!(parse_vector("inf").is_err());
    }

    #[test]
    fn json_objects() {
        assert_eq!(parse_json_object("  ").unwrap(), None);
        assert_eq!(parse_json_object(r#"{"a": 1}"#).unwrap(), Some(json!({"a": 1})));
        assert!(parse_json_object("[1]").is_err());
        assert!(
            parse_json_object("{oops")
                .unwrap_err()
                .starts_with("invalid JSON")
        );
    }

    #[test]
    fn text_field_editing_handles_unicode() {
        let mut f = TextField::new("hé");
        f.insert('!');
        assert_eq!(f.value(), "hé!");
        f.left();
        f.left();
        f.backspace();
        assert_eq!(f.value(), "é!");
        f.delete();
        assert_eq!(f.value(), "!");
        f.end();
        f.insert('x');
        f.home();
        f.insert('>');
        assert_eq!(f.value(), ">!x");
        assert_eq!(f.cursor(), 1);
    }

    #[test]
    fn timestamps() {
        assert_eq!(format_timestamp(0), "1970-01-01 00:00:00 UTC");
        assert_eq!(format_timestamp(1_792_173_547_000), "2026-10-16 17:59:07 UTC");
        // Leap day.
        assert_eq!(format_timestamp(951_782_400_000), "2000-02-29 00:00:00 UTC");
    }

    #[test]
    fn truncation_and_sizes() {
        assert_eq!(truncate("abcdef", 4), "abc…");
        assert_eq!(truncate("abc", 4), "abc");
        assert_eq!(preview_json(Some(&json!({"k": "v"})), 50), r#"{"k":"v"}"#);
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KiB");
    }
}
