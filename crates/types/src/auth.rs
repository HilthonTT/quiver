use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    #[serde(alias = "read_only", alias = "read")]
    ReadOnly,
    #[serde(alias = "read_write", alias = "write")]
    ReadWrite,
}

impl Role {
    pub fn allows(self, required: Role) -> bool {
        self >= required
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiKey {
    pub name: String,
    pub role: Role,
    pub key: String,
    /// The collections this key may use: exact names, or prefixes ending in
    /// `*` (`"tenant-a-*"`). `None` (the default) means every collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collections: Option<Vec<String>>,
}

impl ApiKey {
    /// A key for every collection.
    pub fn new(name: impl Into<String>, role: Role, key: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            role,
            key: key.into(),
            collections: None,
        }
    }

    /// Restricts the key to `collections` (names or `prefix*` patterns).
    #[must_use]
    pub fn with_collections<I, S>(mut self, collections: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.collections = Some(collections.into_iter().map(Into::into).collect());
        self
    }

    /// What a request presenting this key may do.
    pub fn access(&self) -> Access {
        Access {
            role: self.role,
            scope: match &self.collections {
                None => Scope::All,
                Some(patterns) => Scope::Only(Arc::from(patterns.as_slice())),
            },
        }
    }
}

/// Which collections a request may touch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// Every collection, and the server-wide endpoints.
    All,
    /// Collections whose names match one of these patterns: an exact name, or
    /// a prefix followed by `*`.
    Only(Arc<[String]>),
}

impl Scope {
    /// Whether the name `name` matches the scope. For an alias, callers also
    /// check the collection it points to.
    pub fn covers(&self, name: &str) -> bool {
        match self {
            Scope::All => true,
            Scope::Only(patterns) => patterns.iter().any(|p| match p.strip_suffix('*') {
                Some(prefix) => name.starts_with(prefix),
                None => p == name,
            }),
        }
    }
}

/// What a request may do: the role and collection scope of its API key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Access {
    pub role: Role,
    pub scope: Scope,
}

impl Access {
    /// Read-write access to everything, as when no API keys are configured.
    pub fn full() -> Self {
        Self {
            role: Role::ReadWrite,
            scope: Scope::All,
        }
    }

    /// Whether the key is limited to some collections.
    pub fn is_scoped(&self) -> bool {
        matches!(self.scope, Scope::Only(_))
    }

    /// Whether the name `name` is in scope (aliases are not resolved).
    pub fn covers(&self, name: &str) -> bool {
        self.scope.covers(name)
    }
}

fn validate_patterns(key: &ApiKey) -> Result<()> {
    let Some(patterns) = &key.collections else {
        return Ok(());
    };
    if patterns.is_empty() {
        return Err(Error::invalid(format!(
            "API key `{}` has an empty `collections` list; leave it out to allow every collection",
            key.name
        )));
    }
    for p in patterns {
        let body = p.strip_suffix('*').unwrap_or(p);
        if body.contains('*') {
            return Err(Error::invalid(format!(
                "API key `{}`: `*` may only end a collection pattern, in `{p}`",
                key.name
            )));
        }
        if p.is_empty() {
            return Err(Error::invalid(format!(
                "API key `{}` has an empty collection pattern",
                key.name
            )));
        }
    }
    Ok(())
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKey")
            .field("name", &self.name)
            .field("role", &self.role)
            .field("key", &"<redacted>")
            .field("collections", &self.collections)
            .finish()
    }
}

#[derive(Debug, Clone, Default)]
pub struct KeyStore {
    keys: Vec<ApiKey>,
}

impl KeyStore {
    pub fn new(keys: Vec<ApiKey>) -> Result<Self> {
        for (i, k) in keys.iter().enumerate() {
            if k.key.is_empty() {
                return Err(Error::invalid(format!("API key `{}` is empty", k.name)));
            }
            validate_patterns(k)?;
            if keys[..i].iter().any(|other| other.key == k.key) {
                return Err(Error::invalid(format!(
                    "API key `{}` duplicates another key",
                    k.name
                )));
            }
        }
        Ok(Self { keys })
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let keys: Vec<ApiKey> = serde_json::from_slice(&std::fs::read(path)?)?;
        Self::new(keys)
    }

    pub fn push(&mut self, key: ApiKey) -> Result<()> {
        let mut keys = std::mem::take(&mut self.keys);
        keys.push(key);
        *self = Self::new(keys)?;
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn find(&self, presented: &str) -> Option<&ApiKey> {
        let mut found = None;
        for k in &self.keys {
            if constant_time_eq(k.key.as_bytes(), presented.as_bytes()) {
                found = Some(k);
            }
        }
        found
    }
}

/// Compares secrets without leaking the mismatch position through timing.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &str, role: Role, key: &str) -> ApiKey {
        ApiKey::new(name, role, key)
    }

    #[test]
    fn scopes_match_names_and_prefixes() {
        let all = key("a", Role::ReadOnly, "k").access();
        assert!(!all.is_scoped() && all.covers("anything"));

        let scoped = key("a", Role::ReadOnly, "k")
            .with_collections(["books", "tenant-a-*"])
            .access();
        assert!(scoped.is_scoped());
        assert!(scoped.covers("books"));
        assert!(!scoped.covers("books2"));
        assert!(!scoped.covers("book"));
        assert!(scoped.covers("tenant-a-"));
        assert!(scoped.covers("tenant-a-docs"));
        assert!(!scoped.covers("tenant-b-docs"));

        let star = key("a", Role::ReadOnly, "k").with_collections(["*"]).access();
        assert!(star.covers("x"), "a lone `*` matches every name");
    }

    #[test]
    fn bad_scopes_are_refused() {
        let empty: [&str; 0] = [];
        for bad in [
            key("a", Role::ReadOnly, "k").with_collections(empty),
            key("a", Role::ReadOnly, "k").with_collections([""]),
            key("a", Role::ReadOnly, "k").with_collections(["a*b"]),
            key("a", Role::ReadOnly, "k").with_collections(["**"]),
        ] {
            assert!(KeyStore::new(vec![bad.clone()]).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn parse_scoped_keys() {
        let keys: Vec<ApiKey> = serde_json::from_str(
            r#"[{"name":"t","role":"readonly","key":"k","collections":["books","t-*"]}]"#,
        )
        .unwrap();
        assert_eq!(keys[0].collections.as_deref().unwrap(), ["books", "t-*"]);
        assert!(
            serde_json::from_str::<Vec<ApiKey>>(
                r#"[{"name":"t","role":"readonly","key":"k","colections":[]}]"#
            )
            .is_err(),
            "misspelt fields are refused"
        );
    }

    #[test]
    fn roles_are_ordered() {
        assert!(Role::ReadWrite.allows(Role::ReadOnly));
        assert!(Role::ReadWrite.allows(Role::ReadWrite));
        assert!(Role::ReadOnly.allows(Role::ReadOnly));
        assert!(!Role::ReadOnly.allows(Role::ReadWrite));
    }

    #[test]
    fn find_and_validate() {
        let store = KeyStore::new(vec![
            key("admin", Role::ReadWrite, "aaa"),
            key("viewer", Role::ReadOnly, "bbb"),
        ])
        .unwrap();
        assert_eq!(store.find("bbb").unwrap().name, "viewer");
        assert!(store.find("bb").is_none());
        assert!(
            KeyStore::new(vec![
                key("a", Role::ReadOnly, "x"),
                key("b", Role::ReadWrite, "x")
            ])
            .is_err()
        );
        assert!(KeyStore::new(vec![key("a", Role::ReadOnly, "")]).is_err());
    }

    #[test]
    fn parse_keys_file_format() {
        let keys: Vec<ApiKey> = serde_json::from_str(
            r#"[{"name":"a","role":"readwrite","key":"k1"},{"name":"b","role":"readonly","key":"k2"}]"#,
        )
        .unwrap();
        assert_eq!(keys[1].role, Role::ReadOnly);
        assert!(
            !format!("{:?}", keys[0]).contains("k1"),
            "keys are redacted in Debug"
        );
    }

    #[test]
    fn constant_time_eq_works() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secrets"));
    }
}
