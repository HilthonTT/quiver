use std::cmp::Ordering;

use serde_json::{Map, Number, Value};

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// All sub-filters must match.
    And(Vec<Filter>),
    /// At least one sub-filter must match.
    Or(Vec<Filter>),
    /// The sub-filter must not match.
    Not(Box<Filter>),
    /// A condition on a single field.
    Field {
        /// Dot-separated path into the metadata object, split into segments.
        path: Vec<String>,
        /// Condition the value at `path` must satisfy.
        cond: Condition,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    /// Equal to (`$eq`, or a bare value).
    Eq(Value),
    /// Not equal to (`$ne`). Also matches when the field is missing.
    Ne(Value),
    /// Greater than (`$gt`).
    Gt(Value),
    /// Greater than or equal to (`$gte`).
    Gte(Value),
    /// Less than (`$lt`).
    Lt(Value),
    /// Less than or equal to (`$lte`).
    Lte(Value),
    /// Equal to any of the values (`$in`).
    In(Vec<Value>),
    /// Equal to none of the values (`$nin`). Also matches when the field is missing.
    Nin(Vec<Value>),
    /// Field is present (`true`) or absent (`false`) (`$exists`).
    Exists(bool),
    Text(Vec<String>),
    Prefix(String),
    Regex(Pattern),
    All(Vec<Value>),
    Size(usize),
    ElemMatch(Box<Filter>),
    GeoRadius {
        center: GeoPoint,
        meters: f64,
    },
    GeoBox {
        top_left: GeoPoint,
        bottom_right: GeoPoint,
    },
}

pub const MAX_REGEX_LEN: usize = 1_024;
const REGEX_SIZE_LIMIT: usize = 1 << 20;
const EARTH_RADIUS_METERS: f64 = 6_371_008.8;

#[derive(Debug, Clone)]
pub struct Pattern(regex::Regex);

impl Pattern {
    pub fn new(pattern: &str) -> Result<Self> {
        if pattern.len() > MAX_REGEX_LEN {
            return Err(Error::InvalidFilter(format!(
                "`$regex` patterns can be at most {MAX_REGEX_LEN} bytes"
            )));
        }
        regex::RegexBuilder::new(pattern)
            .size_limit(REGEX_SIZE_LIMIT)
            .build()
            .map(Self)
            .map_err(|e| Error::InvalidFilter(format!("invalid `$regex`: {e}")))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn is_match(&self, s: &str) -> bool {
        self.0.is_match(s)
    }
}

impl PartialEq for Pattern {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
}

impl GeoPoint {
    pub fn from_value(v: &Value) -> Option<Self> {
        let obj = v.as_object()?;
        let lat = obj.get("lat")?.as_f64()?;
        let lon = obj.get("lon")?.as_f64()?;
        ((-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon)).then_some(Self { lat, lon })
    }

    pub fn meters_to(&self, other: &GeoPoint) -> f64 {
        let (lat1, lat2) = (self.lat.to_radians(), other.lat.to_radians());
        let dlat = lat2 - lat1;
        let dlon = (other.lon - self.lon).to_radians();
        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        2.0 * EARTH_RADIUS_METERS * a.sqrt().min(1.0).asin()
    }

    fn within_box(&self, top_left: &GeoPoint, bottom_right: &GeoPoint) -> bool {
        let lat_ok = self.lat <= top_left.lat && self.lat >= bottom_right.lat;
        let lon_ok = if top_left.lon <= bottom_right.lon {
            self.lon >= top_left.lon && self.lon <= bottom_right.lon
        } else {
            self.lon >= top_left.lon || self.lon <= bottom_right.lon
        };
        lat_ok && lon_ok
    }
}

pub fn tokenize(s: &str) -> Vec<String> {
    let mut tokens: Vec<String> = s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect();
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

impl Filter {
    pub fn parse(value: &Value) -> Result<Self> {
        let obj = value
            .as_object()
            .ok_or_else(|| Error::InvalidFilter("a filter must be a JSON object".into()))?;

        let mut parts = Vec::with_capacity(obj.len());
        for (key, val) in obj {
            parts.push(parse_entry(key, val)?);
        }
        Ok(if parts.len() == 1 {
            parts.pop().expect("len checked")
        } else {
            Filter::And(parts)
        })
    }

    pub fn matches(&self, metadata: Option<&Value>) -> bool {
        match self {
            Filter::And(fs) => fs.iter().all(|f| f.matches(metadata)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(metadata)),
            Filter::Not(f) => !f.matches(metadata),
            Filter::Field { path, cond } => cond.eval(lookup(metadata, path)),
        }
    }
}

impl TryFrom<&Value> for Filter {
    type Error = Error;

    fn try_from(value: &Value) -> Result<Self> {
        Filter::parse(value)
    }
}

fn parse_entry(key: &str, val: &Value) -> Result<Filter> {
    match key {
        "$and" | "$or" => {
            let arr = val
                .as_array()
                .ok_or_else(|| Error::InvalidFilter(format!("`{key}` expects an array")))?;
            if arr.is_empty() {
                return Err(Error::InvalidFilter(format!("`{key}` must not be empty")));
            }
            let subs = arr.iter().map(Filter::parse).collect::<Result<Vec<_>>>()?;
            Ok(if key == "$and" {
                Filter::And(subs)
            } else {
                Filter::Or(subs)
            })
        }
        "$not" => Ok(Filter::Not(Box::new(Filter::parse(val)?))),
        k if k.starts_with('$') => Err(Error::InvalidFilter(format!("unknown operator `{k}`"))),
        "" => Err(Error::InvalidFilter("field names must not be empty".into())),
        field => {
            let path: Vec<String> = field.split('.').map(str::to_owned).collect();
            match val {
                Value::Object(ops) if is_operator_object(ops) => {
                    let mut conds = ops
                        .iter()
                        .map(|(op, arg)| parse_condition(op, arg))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(if conds.len() == 1 {
                        Filter::Field {
                            path,
                            cond: conds.pop().expect("len checked"),
                        }
                    } else {
                        Filter::And(
                            conds
                                .into_iter()
                                .map(|cond| Filter::Field {
                                    path: path.clone(),
                                    cond,
                                })
                                .collect(),
                        )
                    })
                }
                other => Ok(Filter::Field {
                    path,
                    cond: Condition::Eq(other.clone()),
                }),
            }
        }
    }
}

fn is_operator_object(obj: &Map<String, Value>) -> bool {
    !obj.is_empty() && obj.keys().all(|k| k.starts_with('$'))
}

fn is_logical(key: &str) -> bool {
    matches!(key, "$and" | "$or" | "$not")
}

fn parse_elem_match(arg: &Value) -> Result<Filter> {
    let obj = arg
        .as_object()
        .ok_or_else(|| Error::InvalidFilter("`$elemMatch` expects an object".into()))?;
    if obj.is_empty() {
        return Err(Error::InvalidFilter("`$elemMatch` must not be empty".into()));
    }
    if is_operator_object(obj) && !obj.keys().any(|k| is_logical(k)) {
        let mut parts = obj
            .iter()
            .map(|(op, a)| {
                Ok(Filter::Field {
                    path: Vec::new(),
                    cond: parse_condition(op, a)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(if parts.len() == 1 {
            parts.pop().expect("len checked")
        } else {
            Filter::And(parts)
        });
    }
    Filter::parse(arg)
}

fn geo_point(op: &str, v: Option<&Value>, what: &str) -> Result<GeoPoint> {
    v.and_then(GeoPoint::from_value).ok_or_else(|| {
        Error::InvalidFilter(format!(
            "`{op}` expects `{what}` as {{\"lat\": -90..=90, \"lon\": -180..=180}}"
        ))
    })
}

fn parse_geo_radius(arg: &Value) -> Result<Condition> {
    let op = "$geoRadius";
    let obj = arg
        .as_object()
        .ok_or_else(|| Error::InvalidFilter(format!("`{op}` expects an object")))?;
    if let Some(k) = obj.keys().find(|k| !matches!(k.as_str(), "center" | "meters")) {
        return Err(Error::InvalidFilter(format!("`{op}` has no `{k}`")));
    }
    let center = geo_point(op, obj.get("center"), "center")?;
    let meters = obj
        .get("meters")
        .and_then(Value::as_f64)
        .filter(|m| m.is_finite() && *m >= 0.0)
        .ok_or_else(|| Error::InvalidFilter(format!("`{op}` expects `meters` as a number >= 0")))?;
    Ok(Condition::GeoRadius { center, meters })
}

fn parse_geo_box(arg: &Value) -> Result<Condition> {
    let op = "$geoBox";
    let obj = arg
        .as_object()
        .ok_or_else(|| Error::InvalidFilter(format!("`{op}` expects an object")))?;
    if let Some(k) = obj
        .keys()
        .find(|k| !matches!(k.as_str(), "top_left" | "bottom_right"))
    {
        return Err(Error::InvalidFilter(format!("`{op}` has no `{k}`")));
    }
    let top_left = geo_point(op, obj.get("top_left"), "top_left")?;
    let bottom_right = geo_point(op, obj.get("bottom_right"), "bottom_right")?;
    if top_left.lat < bottom_right.lat {
        return Err(Error::InvalidFilter(format!(
            "`{op}`: `top_left` must not be south of `bottom_right`"
        )));
    }
    Ok(Condition::GeoBox {
        top_left,
        bottom_right,
    })
}

fn parse_condition(op: &str, arg: &Value) -> Result<Condition> {
    let list = |arg: &Value| -> Result<Vec<Value>> {
        arg.as_array()
            .cloned()
            .ok_or_else(|| Error::InvalidFilter(format!("`{op}` expects an array")))
    };

    let string = |arg: &Value| -> Result<String> {
        arg.as_str()
            .map(str::to_owned)
            .ok_or_else(|| Error::InvalidFilter(format!("`{op}` expects a string")))
    };

    let range = |arg: &Value| -> Result<Value> {
        if arg.is_number() || arg.is_string() {
            Ok(arg.clone())
        } else {
            Err(Error::InvalidFilter(format!("`{op}` expects a number or string")))
        }
    };

    Ok(match op {
        "$eq" => Condition::Eq(arg.clone()),
        "$ne" => Condition::Ne(arg.clone()),
        "$gt" => Condition::Gt(range(arg)?),
        "$gte" => Condition::Gte(range(arg)?),
        "$lt" => Condition::Lt(range(arg)?),
        "$lte" => Condition::Lte(range(arg)?),
        "$in" => Condition::In(list(arg)?),
        "$nin" => Condition::Nin(list(arg)?),
        "$exists" => Condition::Exists(
            arg.as_bool()
                .ok_or_else(|| Error::InvalidFilter("`$exists` expects a boolean".into()))?,
        ),
        "$text" => {
            let tokens = tokenize(&string(arg)?);
            if tokens.is_empty() {
                return Err(Error::InvalidFilter("`$text` needs at least one word".into()));
            }
            Condition::Text(tokens)
        }
        "$prefix" => Condition::Prefix(string(arg)?),
        "$regex" => Condition::Regex(Pattern::new(&string(arg)?)?),
        "$all" => {
            let values = list(arg)?;
            if values.is_empty() {
                return Err(Error::InvalidFilter("`$all` must not be empty".into()));
            }
            Condition::All(values)
        }
        "$size" => Condition::Size(
            arg.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Error::InvalidFilter("`$size` expects a non-negative integer".into()))?,
        ),
        "$elemMatch" => Condition::ElemMatch(Box::new(parse_elem_match(arg)?)),
        "$geoRadius" => parse_geo_radius(arg)?,
        "$geoBox" => parse_geo_box(arg)?,
        other => return Err(Error::InvalidFilter(format!("unknown operator `{other}`"))),
    })
}

fn lookup<'a>(metadata: Option<&'a Value>, path: &[String]) -> Option<&'a Value> {
    let mut cur = metadata?;
    for key in path {
        cur = cur.as_object()?.get(key)?;
    }
    Some(cur)
}

impl Condition {
    fn eval(&self, value: Option<&Value>) -> bool {
        match self {
            Condition::Exists(want) => value.is_some() == *want,
            Condition::Ne(target) => !value.is_some_and(|v| any_elem(v, |e| json_eq(e, target))),
            Condition::Nin(targets) => {
                !value.is_some_and(|v| any_elem(v, |e| targets.iter().any(|t| json_eq(e, t))))
            }
            Condition::All(targets) => value.is_some_and(|v| {
                let items = v.as_array().map_or(std::slice::from_ref(v), Vec::as_slice);
                targets.iter().all(|t| items.iter().any(|e| json_eq(e, t)))
            }),
            Condition::Size(n) => value.and_then(Value::as_array).is_some_and(|a| a.len() == *n),
            Condition::ElemMatch(f) => value
                .and_then(Value::as_array)
                .is_some_and(|a| a.iter().any(|e| f.matches(Some(e)))),
            _ => {
                let Some(v) = value else { return false };
                any_elem(v, |e| self.eval_scalar(e))
            }
        }
    }

    fn eval_scalar(&self, v: &Value) -> bool {
        match self {
            Condition::Eq(t) => json_eq(v, t),
            Condition::Gt(t) => compare(v, t) == Some(Ordering::Greater),
            Condition::Gte(t) => matches!(compare(v, t), Some(Ordering::Greater | Ordering::Equal)),
            Condition::Lt(t) => compare(v, t) == Some(Ordering::Less),
            Condition::Lte(t) => matches!(compare(v, t), Some(Ordering::Less | Ordering::Equal)),
            Condition::In(ts) => ts.iter().any(|t| json_eq(v, t)),
            Condition::Text(tokens) => v.as_str().is_some_and(|s| {
                let words = tokenize(s);
                tokens.iter().all(|t| words.binary_search(t).is_ok())
            }),
            Condition::Prefix(p) => v.as_str().is_some_and(|s| s.starts_with(p.as_str())),
            Condition::Regex(r) => v.as_str().is_some_and(|s| r.is_match(s)),
            Condition::GeoRadius { center, meters } => {
                GeoPoint::from_value(v).is_some_and(|p| center.meters_to(&p) <= *meters)
            }
            Condition::GeoBox {
                top_left,
                bottom_right,
            } => GeoPoint::from_value(v).is_some_and(|p| p.within_box(top_left, bottom_right)),
            Condition::Ne(_)
            | Condition::Nin(_)
            | Condition::Exists(_)
            | Condition::All(_)
            | Condition::Size(_)
            | Condition::ElemMatch(_) => {
                unreachable!("handled in eval")
            }
        }
    }
}

fn any_elem(v: &Value, pred: impl Fn(&Value) -> bool) -> bool {
    pred(v) || v.as_array().is_some_and(|arr| arr.iter().any(&pred))
}

fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => compare_numbers(x, y) == Some(Ordering::Equal),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| json_eq(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, a)| y.get(k).is_some_and(|b| json_eq(a, b)))
        }
        _ => a == b,
    }
}

pub fn compare(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => compare_numbers(x, y),
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// Integers (i64 or u64, possibly mixed) compare exactly; f64 is only used
/// when either side is a float, so ids above 2^53 stay distinct.
fn compare_numbers(x: &Number, y: &Number) -> Option<Ordering> {
    let int = |n: &Number| n.as_i64().map(i128::from).or_else(|| n.as_u64().map(i128::from));
    match (int(x), int(y)) {
        (Some(x), Some(y)) => Some(x.cmp(&y)),
        _ => x.as_f64()?.partial_cmp(&y.as_f64()?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn filter(v: Value) -> Filter {
        Filter::parse(&v).unwrap()
    }

    #[test]
    fn large_integers_compare_exactly() {
        let md = json!({"id": 9_007_199_254_740_993u64});
        assert!(!filter(json!({"id": 9_007_199_254_740_992u64})).matches(Some(&md)));
        assert!(filter(json!({"id": 9_007_199_254_740_993u64})).matches(Some(&md)));
        assert!(filter(json!({"id": {"$gt": 9_007_199_254_740_992u64}})).matches(Some(&md)));
        assert!(!filter(json!({"id": {"$lt": 9_007_199_254_740_993u64}})).matches(Some(&md)));
        // Mixed signed / unsigned and int / float still compare by value.
        assert!(filter(json!({"n": {"$lt": u64::MAX}})).matches(Some(&json!({"n": -1}))));
        assert!(filter(json!({"n": 1.0})).matches(Some(&json!({"n": 1}))));
        assert!(filter(json!({"n": {"$gte": 1.5}})).matches(Some(&json!({"n": 2}))));
    }

    #[test]
    fn nested_equality_ignores_integer_vs_float() {
        let md = json!({"v": [1, 2], "o": {"a": 1, "b": [3]}});
        assert!(filter(json!({"v": [1.0, 2.0]})).matches(Some(&md)));
        assert!(!filter(json!({"v": [1.0]})).matches(Some(&md)));
        assert!(filter(json!({"o": {"a": 1.0, "b": [3.0]}})).matches(Some(&md)));
        assert!(!filter(json!({"o": {"a": 1.0}})).matches(Some(&md)));
        assert!(!filter(json!({"o": {"a": 1.0, "c": [3.0]}})).matches(Some(&md)));
    }

    #[test]
    fn text_prefix_and_regex() {
        let md = json!({"title": "The Left Hand of Darkness", "tags": ["Sci-Fi", "classic"]});
        assert!(filter(json!({"title": {"$text": "darkness LEFT"}})).matches(Some(&md)));
        assert!(!filter(json!({"title": {"$text": "left right"}})).matches(Some(&md)));
        assert!(filter(json!({"tags": {"$text": "fi"}})).matches(Some(&md)));
        assert!(filter(json!({"title": {"$prefix": "The L"}})).matches(Some(&md)));
        assert!(!filter(json!({"title": {"$prefix": "the"}})).matches(Some(&md)));
        assert!(filter(json!({"tags": {"$regex": "^cla"}})).matches(Some(&md)));
        assert!(filter(json!({"title": {"$regex": "(?i)hand\\s+of"}})).matches(Some(&md)));
        assert!(!filter(json!({"missing": {"$regex": "."}})).matches(Some(&md)));
        for bad in [
            json!({"t": {"$text": " ,; "}}),
            json!({"t": {"$text": 1}}),
            json!({"t": {"$regex": "("}}),
            json!({"t": {"$regex": "a".repeat(MAX_REGEX_LEN + 1)}}),
            json!({"t": {"$prefix": ["a"]}}),
        ] {
            assert!(Filter::parse(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn array_operators() {
        let md = json!({
            "tags": ["a", "b", "c"],
            "one": "a",
            "scores": [3, 8, 12],
            "reviews": [{"by": "ann", "stars": 5}, {"by": "bob", "stars": 2}],
        });
        assert!(filter(json!({"tags": {"$all": ["c", "a"]}})).matches(Some(&md)));
        assert!(!filter(json!({"tags": {"$all": ["a", "z"]}})).matches(Some(&md)));
        assert!(filter(json!({"one": {"$all": ["a"]}})).matches(Some(&md)));
        assert!(filter(json!({"tags": {"$size": 3}})).matches(Some(&md)));
        assert!(!filter(json!({"tags": {"$size": 2}})).matches(Some(&md)));
        assert!(!filter(json!({"one": {"$size": 1}})).matches(Some(&md)));
        assert!(filter(json!({"scores": {"$elemMatch": {"$gt": 5, "$lt": 10}}})).matches(Some(&md)));
        assert!(!filter(json!({"scores": {"$elemMatch": {"$gt": 8, "$lt": 12}}})).matches(Some(&md)));
        assert!(
            filter(json!({"reviews": {"$elemMatch": {"by": "ann", "stars": {"$gte": 4}}}}))
                .matches(Some(&md))
        );
        assert!(
            !filter(json!({"reviews": {"$elemMatch": {"by": "bob", "stars": {"$gte": 4}}}}))
                .matches(Some(&md))
        );
        assert!(
            filter(json!({"reviews": {"$elemMatch": {"$or": [{"by": "zed"}, {"stars": 2}]}}}))
                .matches(Some(&md))
        );
        assert!(!filter(json!({"one": {"$elemMatch": {"$eq": "a"}}})).matches(Some(&md)));
        for bad in [
            json!({"t": {"$all": []}}),
            json!({"t": {"$size": -1}}),
            json!({"t": {"$size": 1.5}}),
            json!({"t": {"$elemMatch": {}}}),
            json!({"t": {"$elemMatch": 3}}),
        ] {
            assert!(Filter::parse(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn geo_conditions() {
        let brussels = json!({"loc": {"lat": 50.85, "lon": 4.35}});
        let antwerp = json!({"loc": {"lat": 51.22, "lon": 4.40}});
        let both = json!({"loc": [{"lat": 0.0, "lon": 0.0}, {"lat": 51.22, "lon": 4.40}]});
        let near =
            filter(json!({"loc": {"$geoRadius": {"center": {"lat": 50.85, "lon": 4.35}, "meters": 50_000}}}));
        assert!(near.matches(Some(&brussels)));
        assert!(near.matches(Some(&antwerp)));
        assert!(near.matches(Some(&both)));
        let tight =
            filter(json!({"loc": {"$geoRadius": {"center": {"lat": 50.85, "lon": 4.35}, "meters": 10_000}}}));
        assert!(!tight.matches(Some(&antwerp)));

        let b = filter(json!({"loc": {"$geoBox": {
            "top_left": {"lat": 51.0, "lon": 4.0},
            "bottom_right": {"lat": 50.0, "lon": 5.0}
        }}}));
        assert!(b.matches(Some(&brussels)));
        assert!(!b.matches(Some(&antwerp)));
        let wrap = filter(json!({"loc": {"$geoBox": {
            "top_left": {"lat": 10.0, "lon": 170.0},
            "bottom_right": {"lat": -10.0, "lon": -170.0}
        }}}));
        assert!(wrap.matches(Some(&json!({"loc": {"lat": 0, "lon": -175}}))));
        assert!(!wrap.matches(Some(&json!({"loc": {"lat": 0, "lon": 0}}))));
        assert!(!b.matches(Some(&json!({"loc": "50,4"}))));

        for bad in [
            json!({"l": {"$geoRadius": {"center": {"lat": 91, "lon": 0}, "meters": 1}}}),
            json!({"l": {"$geoRadius": {"center": {"lat": 0, "lon": 0}, "meters": -1}}}),
            json!({"l": {"$geoRadius": {"center": {"lat": 0, "lon": 0}}}}),
            json!({"l": {"$geoRadius": {"center": {"lat": 0, "lon": 0}, "meters": 1, "x": 1}}}),
            json!({"l": {"$geoBox": {"top_left": {"lat": 0, "lon": 0}, "bottom_right": {"lat": 1, "lon": 1}}}}),
        ] {
            assert!(Filter::parse(&bad).is_err(), "{bad}");
        }
    }
}
