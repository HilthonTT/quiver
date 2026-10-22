use std::time::{SystemTime, UNIX_EPOCH};

use quiver_types::{Error, ExpiryConfig, Filter, Result};
use serde_json::{Map, Value};

/// The current Unix time in seconds.
pub(crate) fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// The value at the expiry field of `metadata`, if it has one.
fn lookup<'a>(config: &ExpiryConfig, metadata: Option<&'a Value>) -> Option<&'a Value> {
    config
        .field
        .split('.')
        .try_fold(metadata?, |value, segment| value.as_object()?.get(segment))
}

/// Whether `metadata` gives an expiry time no later than `now`.
pub(crate) fn is_expired(config: &ExpiryConfig, metadata: Option<&Value>, now: f64) -> bool {
    lookup(config, metadata)
        .and_then(Value::as_f64)
        .is_some_and(|at| at <= now)
}

/// Matches the points whose expiry time is no later than `now`.
pub(crate) fn expired_filter(config: &ExpiryConfig, now: f64) -> Filter {
    let mut condition = Map::new();
    condition.insert("$lte".into(), Value::from(now));
    let mut filter = Map::new();
    filter.insert(config.field.clone(), Value::Object(condition));
    Filter::parse(&Value::Object(filter)).expect("expiry fields are valid filter paths")
}

/// Checks that the expiry field of point `id`'s metadata, if present, holds
/// a number.
pub(crate) fn check(config: &ExpiryConfig, id: &str, metadata: Option<&Value>) -> Result<()> {
    match lookup(config, metadata) {
        None | Some(Value::Number(_)) => Ok(()),
        Some(_) => Err(Error::invalid(format!(
            "point `{id}`: `{}` must be its expiry time as a number (Unix seconds)",
            config.field
        ))),
    }
}

/// Checks point `id`'s metadata on its way in and, with a default time to
/// live, gives it an expiry time `default_ttl` seconds after `now` if it
/// has none.
pub(crate) fn stamp(
    config: &ExpiryConfig,
    id: &str,
    metadata: Option<Value>,
    now: f64,
) -> Result<Option<Value>> {
    check(config, id, metadata.as_ref())?;
    let Some(ttl) = config.default_ttl else {
        return Ok(metadata);
    };
    if lookup(config, metadata.as_ref()).is_some() {
        return Ok(metadata);
    }
    let not_an_object = || {
        Error::invalid(format!(
            "point `{id}`: metadata must be an object to hold an expiry time in `{}`",
            config.field
        ))
    };
    let mut metadata = metadata.unwrap_or_else(|| Value::Object(Map::new()));
    let mut segments = config.field.split('.').peekable();
    let mut object = metadata.as_object_mut().ok_or_else(not_an_object)?;
    while let Some(segment) = segments.next() {
        if segments.peek().is_none() {
            // Whole seconds, rounded up so points live at least `ttl` seconds.
            object.insert(segment.to_owned(), Value::from(now.ceil() as u64 + ttl));
            break;
        }
        object = object
            .entry(segment)
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
            .ok_or_else(not_an_object)?;
    }
    Ok(Some(metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expiry_times_are_read_from_the_field() {
        let config = ExpiryConfig::new("ttl.at");
        let meta = json!({"ttl": {"at": 100}});
        assert!(is_expired(&config, Some(&meta), 100.0));
        assert!(is_expired(&config, Some(&meta), 150.5));
        assert!(!is_expired(&config, Some(&meta), 99.9));
        assert!(!is_expired(&config, None, 1e12));
        assert!(!is_expired(&config, Some(&json!({"ttl": 5})), 1e12));
        assert!(expired_filter(&config, 100.0).matches(Some(&meta)));
        assert!(!expired_filter(&config, 99.0).matches(Some(&meta)));
    }

    #[test]
    fn stamping_checks_and_fills_in_expiry_times() {
        let plain = ExpiryConfig::new("exp");
        assert_eq!(stamp(&plain, "a", None, 10.0).unwrap(), None);
        assert_eq!(
            stamp(&plain, "a", Some(json!({"exp": 5.5})), 10.0).unwrap(),
            Some(json!({"exp": 5.5}))
        );
        for bad in [
            json!({"exp": "tomorrow"}),
            json!({"exp": null}),
            json!({"exp": [1]}),
        ] {
            assert!(stamp(&plain, "a", Some(bad.clone()), 10.0).is_err(), "{bad}");
        }

        let ttl = ExpiryConfig::new("a.exp").default_ttl(60);
        assert_eq!(
            stamp(&ttl, "a", None, 10.2).unwrap(),
            Some(json!({"a": {"exp": 71}}))
        );
        assert_eq!(
            stamp(&ttl, "a", Some(json!({"k": 1, "a": {"x": 2}})), 10.0).unwrap(),
            Some(json!({"k": 1, "a": {"x": 2, "exp": 70}}))
        );
        assert_eq!(
            stamp(&ttl, "a", Some(json!({"a": {"exp": 5}})), 10.0).unwrap(),
            Some(json!({"a": {"exp": 5}})),
            "given times are kept"
        );
        assert!(stamp(&ttl, "a", Some(json!([1, 2])), 10.0).is_err());
        assert!(stamp(&ttl, "a", Some(json!({"a": 3})), 10.0).is_err());
    }
}
