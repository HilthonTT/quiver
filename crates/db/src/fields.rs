//! Metadata field indexes.
//!
//! An index narrows a filter to *candidate* slots: every slot that matches is
//! among them, but some may not match (deleted points, superseded metadata,
//! numbers that round to the same `f64`). Callers always re-check the full
//! filter on each candidate, so the index never changes a result, only how
//! many points are looked at. That lets updates simply append entries and
//! leave stale ones until the next rebuild.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::ops::Bound;

use quiver_storage::VectorStore;
use quiver_types::filter::Condition;
use quiver_types::{Filter, MAX_FIELD_INDEXES, Result, validate_field_path};
use serde_json::Value;

use quiver_types::Error;

/// Rebuild once stale entries outnumber this many plus the live points.
const REBUILD_SLACK: usize = 4_096;

/// A JSON number as an ordered map key. JSON has no NaN, and `-0.0` is
/// normalised to `0.0` so the two land in one entry, as they compare equal.
#[derive(Debug, Clone, Copy)]
struct Num(f64);

impl Num {
    fn new(x: f64) -> Self {
        Self(if x == 0.0 { 0.0 } else { x })
    }
}

impl PartialEq for Num {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Num {}
impl PartialOrd for Num {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Num {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Slots by value for one field. Slot lists are unsorted and may repeat.
#[derive(Debug, Clone)]
struct FieldIndex {
    name: String,
    path: Vec<String>,
    nums: BTreeMap<Num, Vec<u32>>,
    strs: BTreeMap<String, Vec<u32>>,
    bools: [Vec<u32>; 2],
    nulls: Vec<u32>,
    /// Slots where the field exists, whatever its value.
    present: Vec<u32>,
}

impl FieldIndex {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            path: name.split('.').map(str::to_owned).collect(),
            nums: BTreeMap::new(),
            strs: BTreeMap::new(),
            bools: [Vec::new(), Vec::new()],
            nulls: Vec::new(),
            present: Vec::new(),
        }
    }

    fn add(&mut self, slot: u32, metadata: Option<&Value>) {
        let Some(value) = lookup(metadata, &self.path) else {
            return;
        };
        self.present.push(slot);
        self.add_scalar(slot, value);
        // Filters match an array when any element does.
        if let Value::Array(items) = value {
            for item in items {
                self.add_scalar(slot, item);
            }
        }
    }

    fn add_scalar(&mut self, slot: u32, value: &Value) {
        match value {
            Value::Null => self.nulls.push(slot),
            Value::Bool(b) => self.bools[usize::from(*b)].push(slot),
            Value::Number(n) => {
                if let Some(x) = n.as_f64() {
                    self.nums.entry(Num::new(x)).or_default().push(slot);
                }
            }
            Value::String(s) => self.strs.entry(s.clone()).or_default().push(slot),
            Value::Array(_) | Value::Object(_) => {}
        }
    }

    /// Slots that may equal `value`, or `None` if it is not indexable.
    fn equal(&self, value: &Value, out: &mut Vec<u32>) -> Option<()> {
        match value {
            Value::Null => out.extend(&self.nulls),
            Value::Bool(b) => out.extend(&self.bools[usize::from(*b)]),
            Value::Number(n) => {
                if let Some(slots) = self.nums.get(&Num::new(n.as_f64()?)) {
                    out.extend(slots);
                }
            }
            Value::String(s) => {
                if let Some(slots) = self.strs.get(s) {
                    out.extend(slots);
                }
            }
            Value::Array(_) | Value::Object(_) => return None,
        }
        Some(())
    }

    /// Slots that may satisfy `cond`, or `None` if the index cannot tell.
    fn candidates(&self, cond: &Condition) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        match cond {
            Condition::Eq(v) => self.equal(v, &mut out)?,
            Condition::In(values) => {
                for v in values {
                    self.equal(v, &mut out)?;
                }
            }
            Condition::Exists(true) => out.extend(&self.present),
            Condition::Gt(t) | Condition::Gte(t) | Condition::Lt(t) | Condition::Lte(t) => {
                // Bounds are inclusive even for `$gt`/`$lt`: values that round
                // to the bound must stay candidates.
                let lower = matches!(cond, Condition::Gt(_) | Condition::Gte(_));
                match t {
                    Value::Number(n) => {
                        let bound = Bound::Included(Num::new(n.as_f64()?));
                        let range = if lower {
                            self.nums.range((bound, Bound::Unbounded))
                        } else {
                            self.nums.range((Bound::Unbounded, bound))
                        };
                        range.for_each(|(_, slots)| out.extend(slots));
                    }
                    Value::String(s) => {
                        let bound = Bound::Included(s.as_str());
                        let range = if lower {
                            self.strs.range::<str, _>((bound, Bound::Unbounded))
                        } else {
                            self.strs.range::<str, _>((Bound::Unbounded, bound))
                        };
                        range.for_each(|(_, slots)| out.extend(slots));
                    }
                    _ => return None,
                }
            }
            // Negations match points the index has no entry for.
            Condition::Ne(_) | Condition::Nin(_) | Condition::Exists(false) => return None,
        }
        Some(out)
    }
}

fn lookup<'a>(metadata: Option<&'a Value>, path: &[String]) -> Option<&'a Value> {
    let mut cur = metadata?;
    for key in path {
        cur = cur.as_object()?.get(key)?;
    }
    Some(cur)
}

/// A collection's field indexes.
#[derive(Debug, Clone, Default)]
pub(crate) struct FieldIndexes {
    fields: Vec<FieldIndex>,
    /// Entries for slots that have since been deleted or updated.
    stale: usize,
}

impl FieldIndexes {
    /// Indexes `names` over the live points of `store`.
    pub fn build(names: &[String], store: &VectorStore) -> Self {
        let mut fields: Vec<FieldIndex> = names.iter().map(|n| FieldIndex::new(n)).collect();
        for slot in store.live_slots() {
            for f in &mut fields {
                f.add(slot, store.metadata(slot));
            }
        }
        Self { fields, stale: 0 }
    }

    /// Checks that `names` can be indexed together.
    pub fn validate(names: &[String]) -> Result<()> {
        if names.len() > MAX_FIELD_INDEXES {
            return Err(Error::invalid(format!(
                "a collection can index at most {MAX_FIELD_INDEXES} fields"
            )));
        }
        for (i, name) in names.iter().enumerate() {
            validate_field_path(name)?;
            if names[..i].contains(name) {
                return Err(Error::invalid(format!("field `{name}` is listed twice")));
            }
        }
        Ok(())
    }

    pub fn names(&self) -> Vec<String> {
        self.fields.iter().map(|f| f.name.clone()).collect()
    }

    /// Indexes newly written slots.
    pub fn add(&mut self, slots: std::ops::Range<u32>, store: &VectorStore) {
        for slot in slots {
            for f in &mut self.fields {
                f.add(slot, store.metadata(slot));
            }
        }
    }

    /// Re-indexes a slot whose metadata changed in place.
    pub fn update(&mut self, slot: u32, store: &VectorStore) {
        if !self.fields.is_empty() {
            self.add(slot..slot + 1, store);
            self.stale += 1;
        }
    }

    /// Notes that `n` indexed slots were deleted or superseded.
    pub fn removed(&mut self, n: usize) {
        if !self.fields.is_empty() {
            self.stale += n;
        }
    }

    /// Rebuilds if stale entries have piled up.
    pub fn maintain(&mut self, store: &VectorStore) {
        if self.stale > store.live() + REBUILD_SLACK {
            *self = Self::build(&self.names(), store);
        }
    }

    /// Sorted, deduplicated candidate slots for `filter`, or `None` when no
    /// index applies and every point must be checked. Candidates may include
    /// dead slots and non-matching ones; re-check them.
    pub fn candidates(&self, filter: &Filter) -> Option<Vec<u32>> {
        if self.fields.is_empty() {
            return None;
        }
        let mut slots = self.plan(filter)?;
        slots.sort_unstable();
        slots.dedup();
        Some(slots)
    }

    fn plan(&self, filter: &Filter) -> Option<Vec<u32>> {
        match filter {
            Filter::Field { path, cond } => {
                let field = self.fields.iter().find(|f| f.path == *path)?;
                field.candidates(cond)
            }
            // Any one indexed conjunct bounds the result; intersect them all.
            Filter::And(parts) => {
                let mut best: Option<Vec<u32>> = None;
                for part in parts {
                    let Some(mut slots) = self.plan(part) else {
                        continue;
                    };
                    slots.sort_unstable();
                    slots.dedup();
                    best = Some(match best {
                        None => slots,
                        Some(acc) => intersect(&acc, &slots),
                    });
                }
                best
            }
            // Every alternative must be indexed, or a match could be missed.
            Filter::Or(parts) => {
                let mut out = Vec::new();
                for part in parts {
                    out.extend(self.plan(part)?);
                }
                Some(out)
            }
            Filter::Not(_) => None,
        }
    }
}

/// Intersection of two sorted, deduplicated lists.
fn intersect(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store(metadata: &[Value]) -> VectorStore {
        let mut s = VectorStore::new(1);
        for (i, m) in metadata.iter().enumerate() {
            s.push(i.to_string(), &[0.0], Some(m.clone()));
        }
        s
    }

    /// The index's candidates always include the true matches, which are
    /// what a full scan finds.
    fn check(idx: &FieldIndexes, store: &VectorStore, filter: Value) -> Option<Vec<u32>> {
        let filter = Filter::parse(&filter).unwrap();
        let truth: Vec<u32> = store
            .live_slots()
            .filter(|&s| filter.matches(store.metadata(s)))
            .collect();
        let got = idx.candidates(&filter)?;
        for t in &truth {
            assert!(got.contains(t), "{filter:?}: index lost slot {t}");
        }
        Some(got)
    }

    #[test]
    fn candidates_cover_every_match() {
        let s = store(&[
            json!({"genre": "scifi", "year": 1965, "tags": ["a", "b"], "stats": {"views": 10}}),
            json!({"genre": "fantasy", "year": 1937.0, "tags": "b"}),
            json!({"genre": "scifi", "year": -0.0, "stats": {"views": 3}}),
            json!({"year": null, "flag": true}),
            json!({"genre": ["scifi", "horror"], "year": 9_007_199_254_740_993u64}),
        ]);
        let idx = FieldIndexes::build(
            &["genre".into(), "year".into(), "tags".into(), "stats.views".into()],
            &s,
        );
        assert_eq!(check(&idx, &s, json!({"genre": "scifi"})), Some(vec![0, 2, 4]));
        assert_eq!(check(&idx, &s, json!({"tags": "b"})), Some(vec![0, 1]));
        assert_eq!(check(&idx, &s, json!({"year": 0})), Some(vec![2]));
        assert_eq!(check(&idx, &s, json!({"year": null})), Some(vec![3]));
        assert_eq!(
            check(&idx, &s, json!({"year": {"$gt": 1937}})),
            Some(vec![0, 1, 4])
        );
        assert_eq!(check(&idx, &s, json!({"year": {"$lte": 1937}})), Some(vec![1, 2]));
        assert_eq!(
            check(&idx, &s, json!({"year": {"$gt": 9_007_199_254_740_992u64}})),
            Some(vec![4])
        );
        assert_eq!(
            check(&idx, &s, json!({"genre": {"$in": ["fantasy", "horror"]}})),
            Some(vec![1, 4])
        );
        assert_eq!(
            check(&idx, &s, json!({"genre": {"$gte": "g"}})),
            Some(vec![0, 2, 4])
        );
        assert_eq!(check(&idx, &s, json!({"stats.views": {"$lt": 5}})), Some(vec![2]));
        assert_eq!(
            check(&idx, &s, json!({"year": {"$exists": true}})),
            Some(vec![0, 1, 2, 3, 4])
        );
        assert_eq!(
            check(
                &idx,
                &s,
                json!({"genre": "scifi", "flag": true, "year": {"$lt": 2000}})
            ),
            Some(vec![0, 2])
        );
        assert_eq!(
            check(&idx, &s, json!({"$or": [{"genre": "fantasy"}, {"tags": "a"}]})),
            Some(vec![0, 1])
        );

        // Unindexed fields, negations and partly-indexed `$or` need a scan.
        for unplannable in [
            json!({"flag": true}),
            json!({"genre": {"$ne": "scifi"}}),
            json!({"$not": {"genre": "scifi"}}),
            json!({"year": {"$exists": false}}),
            json!({"$or": [{"genre": "fantasy"}, {"flag": true}]}),
            json!({"stats": {"views": 10}}),
        ] {
            assert_eq!(check(&idx, &s, unplannable.clone()), None, "{unplannable}");
        }
    }

    #[test]
    fn updates_and_rebuilds_keep_candidates_complete() {
        let mut s = store(&[json!({"g": "a"}), json!({"g": "b"})]);
        let mut idx = FieldIndexes::build(&["g".into()], &s);
        s.set_metadata(0, Some(json!({"g": "b"})));
        idx.update(0, &s);
        assert_eq!(check(&idx, &s, json!({"g": "b"})), Some(vec![0, 1]));
        // The stale entry for slot 0 under "a" is still a candidate, harmlessly.
        assert_eq!(check(&idx, &s, json!({"g": "a"})), Some(vec![0]));

        let first = s.slots() as u32;
        s.push("2".into(), &[0.0], Some(json!({"g": "a"})));
        idx.add(first..s.slots() as u32, &s);
        assert_eq!(check(&idx, &s, json!({"g": "a"})), Some(vec![0, 2]));

        idx.stale = usize::MAX / 2;
        idx.maintain(&s);
        assert_eq!(idx.stale, 0);
        assert_eq!(check(&idx, &s, json!({"g": "a"})), Some(vec![2]));
    }

    #[test]
    fn validation() {
        assert!(FieldIndexes::validate(&["a".into(), "b.c".into()]).is_ok());
        assert!(FieldIndexes::validate(&["a".into(), "a".into()]).is_err());
        assert!(FieldIndexes::validate(&["$a".into()]).is_err());
        let many: Vec<String> = (0..=MAX_FIELD_INDEXES).map(|i| format!("f{i}")).collect();
        assert!(FieldIndexes::validate(&many).is_err());
    }
}
