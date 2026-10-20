use std::cmp::Ordering;

use serde_json::{Map, Value};

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

fn parse_condition(op: &str, arg: &Value) -> Result<Condition> {
    let list = |arg: &Value| -> Result<Vec<Value>> {
        arg.as_array()
            .cloned()
            .ok_or_else(|| Error::InvalidFilter(format!("`{op}` expects an array")))
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
            Condition::Ne(_) | Condition::Nin(_) | Condition::Exists(_) => {
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
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

fn compare(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64()?.partial_cmp(&y.as_f64()?),
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}
