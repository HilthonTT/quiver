use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use vectordb_types::{Error, Result};

use crate::codec::WriteExt;
use crate::{Mutation, Op};

/// Frames larger than this are treated as corruption rather than allocated.
const MAX_FRAME: usize = 1 << 30;

#[derive(Debug)]
pub(crate) struct Wal {
    file: File,
    size: u64,
    fsync: bool,
}

impl Wal {
    pub fn open(path: &Path, fsync: bool) -> Result<(Self, Vec<Mutation>)> {
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let (entries, valid) = parse_frames(&buf);
        if valid < buf.len() {
            tracing::warn!(
                path = %path.display(),
                discarded_bytes = buf.len() - valid,
                "truncating torn write-ahead log tail"
            );
            file.set_len(valid as u64)?;
            file.sync_data()?;
        }
        Ok((
            Self {
                file,
                size: valid as u64,
                fsync,
            },
            entries,
        ))
    }

    pub fn append(&mut self, record: Op<'_>) -> Result<()> {
        let payload = serde_json::to_vec(&record)?;
        // Replay treats larger frames as corruption and would silently drop
        // this entry and everything logged after it.
        if payload.len() > MAX_FRAME {
            return Err(Error::invalid(format!(
                "write batch too large ({} bytes encoded, limit {MAX_FRAME}); split it up",
                payload.len()
            )));
        }
        let mut frame = Vec::with_capacity(payload.len() + 8);
        frame.put_u32(payload.len() as u32)?;
        frame.put_u32(crc32fast::hash(&payload))?;
        frame.extend_from_slice(&payload);

        let written = self.file.write_all(&frame).and_then(|()| {
            if self.fsync {
                self.file.sync_data()
            } else {
                Ok(())
            }
        });
        if let Err(e) = written {
            // Drop any partial frame: replay stops at the first bad frame, so
            // leaving it would hide every later, successful append.
            let _ = self.file.set_len(self.size);
            return Err(e.into());
        }
        self.size += frame.len() as u64;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.sync_data()?;
        self.size = 0;
        Ok(())
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

fn parse_frames(buf: &[u8]) -> (Vec<Mutation>, usize) {
    let mut entries = Vec::new();
    let mut pos = 0;

    while buf.len() - pos >= 8 {
        let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().expect("4 bytes")) as usize;
        let crc = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().expect("4 bytes"));
        let start = pos + 8;
        if len > MAX_FRAME || buf.len() - start < len {
            break;
        }
        let payload = &buf[start..start + len];
        if crc32fast::hash(payload) != crc {
            break;
        }
        match serde_json::from_slice(payload) {
            Ok(entry) => entries.push(entry),
            Err(_) => break,
        }
        pos = start + len;
    }

    (entries, pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectordb_types::Point;

    fn point(id: &str) -> Point {
        Point::new(id, vec![1.0, 2.0])
    }

    #[test]
    fn append_and_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        {
            let (mut wal, entries) = Wal::open(&path, true).unwrap();
            assert!(entries.is_empty());
            wal.append(Op::Upsert {
                points: &[point("a")],
            })
            .unwrap();
            wal.append(Op::Delete { ids: &["a".into()] }).unwrap();
        }
        let (_, entries) = Wal::open(&path, false).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(matches!(&entries[0], Mutation::Upsert { points } if points[0].id == "a"));
        assert!(matches!(&entries[1], Mutation::Delete { ids } if ids == &["a"]));
    }

    #[test]
    fn torn_tail_is_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        {
            let (mut wal, _) = Wal::open(&path, false).unwrap();
            wal.append(Op::Upsert {
                points: &[point("a")],
            })
            .unwrap();
            wal.append(Op::Upsert {
                points: &[point("b")],
            })
            .unwrap();
        }
        // Simulate a crash in the middle of the second frame.
        let full = std::fs::metadata(&path).unwrap().len();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(full - 3)
            .unwrap();

        let (mut wal, entries) = Wal::open(&path, false).unwrap();
        assert_eq!(entries.len(), 1);
        wal.append(Op::Upsert {
            points: &[point("c")],
        })
        .unwrap();
        drop(wal);

        let (_, entries) = Wal::open(&path, false).unwrap();
        assert_eq!(
            entries.len(),
            2,
            "appends after truncation must be readable"
        );
    }

    #[test]
    fn corrupt_crc_stops_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        {
            let (mut wal, _) = Wal::open(&path, false).unwrap();
            wal.append(Op::Upsert {
                points: &[point("a")],
            })
            .unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, bytes).unwrap();
        let (_, entries) = Wal::open(&path, false).unwrap();
        assert!(entries.is_empty());
    }
}
