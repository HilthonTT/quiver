use std::cmp::Ordering;
use std::collections::HashMap;

use quiver_types::{FacetCount, filter};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Key {
    Bool(bool),
    Int(i128),
    Float(u64),
    Str(String),
}

impl Key {
    fn of(v: &Value) -> Option<Self> {
        Some(match v {
            Value::Bool(b) => Key::Bool(*b),
            Value::String(s) => Key::Str(s.clone()),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Key::Int(i128::from(i))
                } else if let Some(u) = n.as_u64() {
                    Key::Int(i128::from(u))
                } else {
                    let x = n.as_f64()?;
                    if x.fract() == 0.0 && x.abs() < i128::MAX as f64 {
                        Key::Int(x as i128)
                    } else {
                        Key::Float(x.to_bits())
                    }
                }
            }
            Value::Null | Value::Array(_) | Value::Object(_) => return None,
        })
    }
}

pub(crate) fn lookup<'a>(metadata: Option<&'a Value>, path: &[String]) -> Option<&'a Value> {
    let mut cur = metadata?;
    for key in path {
        cur = cur.as_object()?.get(key)?;
    }
    Some(cur)
}

pub(crate) fn split_path(field: &str) -> Vec<String> {
    field.split('.').map(str::to_owned).collect()
}

pub(crate) fn keys<'a>(metadata: Option<&'a Value>, path: &[String]) -> Vec<(Key, &'a Value)> {
    let Some(v) = lookup(metadata, path) else {
        return Vec::new();
    };
    let mut out: Vec<(Key, &Value)> = match v {
        Value::Array(items) => items.iter().filter_map(|e| Key::of(e).map(|k| (k, e))).collect(),
        other => Key::of(other).map(|k| (k, other)).into_iter().collect(),
    };
    let mut seen = Vec::with_capacity(out.len());
    out.retain(|(k, _)| {
        if seen.contains(k) {
            false
        } else {
            seen.push(k.clone());
            true
        }
    });
    out
}

#[derive(Debug, Default)]
pub(crate) struct Facets {
    counts: HashMap<Key, (Value, usize)>,
}

impl Facets {
    pub fn add(&mut self, metadata: Option<&Value>, path: &[String]) {
        for (key, value) in keys(metadata, path) {
            self.counts.entry(key).or_insert_with(|| (value.clone(), 0)).1 += 1;
        }
    }

    pub fn top(self, limit: usize) -> Vec<FacetCount> {
        let mut values: Vec<FacetCount> = self
            .counts
            .into_values()
            .map(|(value, count)| FacetCount { value, count })
            .collect();
        values.sort_by(|a, b| {
            b.count
                .cmp(&a.count)
                .then_with(|| value_order(&a.value, &b.value))
        });
        values.truncate(limit);
        values
    }
}

pub(crate) fn value_order(a: &Value, b: &Value) -> Ordering {
    let rank = |v: &Value| match v {
        Value::Bool(_) => 0,
        Value::Number(_) => 1,
        Value::String(_) => 2,
        _ => 3,
    };
    match (a, b) {
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        _ => filter::compare(a, b).unwrap_or_else(|| rank(a).cmp(&rank(b))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keys_merge_equal_numbers_and_skip_compound_values() {
        let path = split_path("k");
        let md = json!({"k": [1, 1.0, "1", true, null, {"a": 1}, [2]]});
        let got: Vec<Key> = keys(Some(&md), &path).into_iter().map(|(k, _)| k).collect();
        assert_eq!(got, [Key::Int(1), Key::Str("1".into()), Key::Bool(true)]);
        assert!(keys(Some(&json!({"k": {"a": 1}})), &path).is_empty());
        assert!(keys(None, &path).is_empty());
        let big = json!({"k": [1.0e16, 10_000_000_000_000_000u64]});
        assert_eq!(keys(Some(&big), &path).len(), 1);
    }

    #[test]
    fn facets_count_points_and_sort_by_count_then_value() {
        let path = split_path("tags");
        let mut f = Facets::default();
        for md in [
            json!({"tags": ["b", "a", "a"]}),
            json!({"tags": "b"}),
            json!({"tags": ["c", 2]}),
            json!({"other": 1}),
        ] {
            f.add(Some(&md), &path);
        }
        let top = f.top(10);
        let pairs: Vec<(Value, usize)> = top.into_iter().map(|c| (c.value, c.count)).collect();
        assert_eq!(
            pairs,
            [(json!("b"), 2), (json!(2), 1), (json!("a"), 1), (json!("c"), 1)]
        );
    }
}
