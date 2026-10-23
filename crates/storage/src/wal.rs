use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use quiver_types::{Error, Result};

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

        let (entries, valid) = parse_frames(&buf).map_err(|offset| {
            Error::Corrupt(format!(
                "write-ahead log {} is corrupt at byte {offset} with {} bytes of later records; \
                 move it aside to discard them and open anyway",
                path.display(),
                buf.len() - offset
            ))
        })?;
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

        let written = self
            .file
            .write_all(&frame)
            .and_then(|()| if self.fsync { self.file.sync_data() } else { Ok(()) });
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

/// Returns the replayed entries and the length of the valid prefix.
///
/// A bad frame at the end of the log is a torn final write and is dropped,
/// as is a zero-filled tail (a crash can extend the file before the data
/// lands). A bad frame followed by more data is corruption: `Err` holds its
/// offset so the caller can refuse to discard the records after it.
fn parse_frames(buf: &[u8]) -> std::result::Result<(Vec<Mutation>, usize), usize> {
    let mut entries = Vec::new();
    let mut pos = 0;

    while buf.len() - pos >= 8 {
        let zero_from = |from: usize| buf[from..].iter().all(|&b| b == 0);
        let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().expect("4 bytes")) as usize;
        let crc = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().expect("4 bytes"));
        let start = pos + 8;
        if len > MAX_FRAME {
            if zero_from(start) {
                break;
            }
            return Err(pos);
        }
        if buf.len() - start < len {
            break;
        }
        let end = start + len;
        let payload = &buf[start..end];
        if crc32fast::hash(payload) != crc {
            if zero_from(end) {
                break;
            }
            return Err(pos);
        }
        match serde_json::from_slice(payload) {
            Ok(entry) => entries.push(entry),
            // A zero header has a valid checksum (crc32 of nothing is 0).
            Err(_) if zero_from(pos) => break,
            // Any other checksum-valid frame was written completely, so
            // failing to parse it is corruption, never a torn write.
            Err(_) => return Err(pos),
        }
        pos = end;
    }

    Ok((entries, pos))
}

#[cfg(test)]
mod tests {
    use super::*;
    use quiver_types::Point;

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
        assert_eq!(entries.len(), 2, "appends after truncation must be readable");
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

    fn three_frames(path: &Path) -> Vec<u8> {
        {
            let (mut wal, _) = Wal::open(path, false).unwrap();
            for id in ["a", "b", "c"] {
                wal.append(Op::Upsert { points: &[point(id)] }).unwrap();
            }
        }
        std::fs::read(path).unwrap()
    }

    #[test]
    fn corrupt_middle_frame_is_reported_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut bytes = three_frames(&path);
        bytes[10] ^= 0x01; // inside the first frame's payload
        std::fs::write(&path, &bytes).unwrap();

        let err = Wal::open(&path, false).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
        // The intact frames after it must still be on disk for recovery.
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn zero_filled_tail_is_truncated() {
        // After a power loss the file can be extended with zeros the data never reached.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut bytes = three_frames(&path);
        let valid = bytes.len();
        bytes.extend_from_slice(&[0; 64]);
        std::fs::write(&path, &bytes).unwrap();

        let (_, entries) = Wal::open(&path, false).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, valid);
    }

    #[test]
    fn zeros_starting_inside_a_frame_are_a_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut bytes = three_frames(&path);
        let second = bytes.len() / 3;
        for b in &mut bytes[second + 12..] {
            *b = 0;
        }
        std::fs::write(&path, &bytes).unwrap();

        let (_, entries) = Wal::open(&path, false).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, second);
    }

    #[test]
    fn oversized_length_before_later_frames_is_reported_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut bytes = three_frames(&path);
        let second = bytes.len() / 3;
        bytes[second + 3] ^= 0x80;
        std::fs::write(&path, &bytes).unwrap();

        let err = Wal::open(&path, false).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "{err:?}");
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn unparsable_frame_with_valid_checksum_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut bytes = three_frames(&path);
        let payload = b"not json";
        let mut frame = Vec::new();
        frame.put_u32(payload.len() as u32).unwrap();
        frame.put_u32(crc32fast::hash(payload)).unwrap();
        frame.extend_from_slice(payload);
        bytes.extend_from_slice(&frame);
        std::fs::write(&path, &bytes).unwrap();

        // A checksum-valid frame was fully written, so this is not a torn tail.
        assert!(matches!(Wal::open(&path, false), Err(Error::Corrupt(_))));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }
}
