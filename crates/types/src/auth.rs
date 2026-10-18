use std::path::Path;

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
pub struct ApiKey {
    pub name: String,
    pub role: Role,
    pub key: String,
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKey")
            .field("name", &self.name)
            .field("role", &self.role)
            .field("key", &"<redacted>")
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
        ApiKey {
            name: name.into(),
            role,
            key: key.into(),
        }
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
