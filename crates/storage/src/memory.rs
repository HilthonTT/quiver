use quiver_types::Result;

use crate::{Op, Recovered, StorageEngine, StorageKind, VectorStore};

/// Keeps no durable state; everything is lost when the process exits.
#[derive(Debug, Default)]
pub struct MemoryEngine;

impl StorageEngine for MemoryEngine {
    fn kind(&self) -> StorageKind {
        StorageKind::Memory
    }

    fn recover(&mut self, dim: usize) -> Result<Recovered> {
        Ok(Recovered::empty(dim))
    }

    fn log(&mut self, op: Op<'_>) -> Result<()> {
        Ok(())
    }

    fn needs_checkpoint(&self) -> bool {
        false
    }

    fn checkpoint(&mut self, store: &VectorStore, index: &[u8]) -> Result<()> {
        Ok(())
    }

    fn import(&mut self, store: &VectorStore, index: &[u8]) -> Result<()> {
        Ok(())
    }
}
