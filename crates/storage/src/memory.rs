use quiver_types::Result;

use crate::{Op, Recovered, StorageEngine, StorageKind, StoreSpec, VectorStore};

/// Keeps no durable state; everything is lost when the process exits.
#[derive(Debug, Default)]
pub struct MemoryEngine;

// Applied marks need no keeping either: after a restart the collection is
// empty, which is the state before the first replicated entry.

impl StorageEngine for MemoryEngine {
    fn kind(&self) -> StorageKind {
        StorageKind::Memory
    }

    fn recover(&mut self, spec: &StoreSpec) -> Result<Recovered> {
        Recovered::empty(spec)
    }

    fn log(&mut self, _op: Op<'_>) -> Result<()> {
        Ok(())
    }

    fn needs_checkpoint(&self) -> bool {
        false
    }

    fn checkpoint(&mut self, _store: &VectorStore, _index: &[u8]) -> Result<()> {
        Ok(())
    }

    fn import(&mut self, _store: &VectorStore, _index: &[u8]) -> Result<()> {
        Ok(())
    }
}
