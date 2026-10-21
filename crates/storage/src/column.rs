use std::borrow::Cow;
use std::cell::RefCell;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use quiver_types::{Error, Metric, Result};

/// How one column of vectors is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColumnSpec {
    pub dim: usize,
    /// Scalar-quantized codes in memory (see [`quiver_types::Quantization`]).
    pub quantized: bool,
    /// The exact vectors in a file rather than in memory.
    pub on_disk: bool,
}

impl ColumnSpec {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            quantized: false,
            on_disk: false,
        }
    }
}

/// File names of on-disk columns start with this; anything else in the
/// directory is left alone when stale columns are cleaned up.
pub const DISK_FILE_PREFIX: &str = "vectors-";
const DISK_FILE_SUFFIX: &str = ".f32";

/// Scalar-quantized vectors: one code byte per component, decoded as
/// `min + code * step` with `[min, step]` stored per vector.
#[derive(Debug, Clone, Default)]
struct Codes {
    codes: Vec<u8>,
    ranges: Vec<[f32; 2]>,
}

impl Codes {
    fn push(&mut self, v: &[f32]) {
        let (min, step, code) = quantize(v);
        self.codes.extend(code);
        self.ranges.push([min, step]);
    }

    fn decode_into(&self, dim: usize, slot: u32, out: &mut Vec<f32>) {
        let start = slot as usize * dim;
        let [min, step] = self.ranges[slot as usize];
        out.clear();
        out.extend(
            self.codes[start..start + dim]
                .iter()
                .map(|&c| decode(c, min, step)),
        );
    }

    fn bytes(&self) -> usize {
        self.codes.len() + self.ranges.len() * 8
    }
}

/// Exact vectors in a file, slot `s` at byte `s * dim * 4`, optionally with
/// quantized codes in memory to search with.
#[derive(Debug)]
struct Disk {
    file: File,
    path: PathBuf,
    codes: Option<Codes>,
}

#[derive(Debug)]
enum Data {
    F32(Vec<f32>),
    Sq8(Codes),
    Disk(Disk),
}

thread_local! {
    /// Decoding space for quantized and on-disk distances, so searches do not allocate.
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static BYTES: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

/// One vector per slot, all of the same dimension.
///
/// On-disk columns treat I/O errors on their file as fatal: a read or write
/// that fails panics, failing the request. Every change is in the write-ahead
/// log before it reaches the column, so nothing acknowledged is lost.
#[derive(Debug)]
pub struct VectorColumn {
    dim: usize,
    slots: usize,
    data: Data,
}

impl VectorColumn {
    /// An empty in-memory column of full floats.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            slots: 0,
            data: Data::F32(Vec::new()),
        }
    }

    /// An empty column laid out as `spec` says. On-disk columns get a new
    /// file in `dir`, named after `name`.
    pub fn create(spec: ColumnSpec, dir: Option<&Path>, name: &str) -> Result<Self> {
        let mut column = Self::new(spec.dim);
        if spec.on_disk {
            let dir = dir.ok_or_else(|| {
                Error::invalid("vectors can only be kept on disk with a durable storage engine")
            })?;
            let path = dir.join(new_file_name(name));
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)?;
            column.data = Data::Disk(Disk {
                file,
                path,
                codes: spec.quantized.then(Codes::default),
            });
        } else if spec.quantized {
            column.data = Data::Sq8(Codes::default());
        }
        Ok(column)
    }

    /// Takes over an on-disk column's file, keeping its first `slots`
    /// vectors: later ones were never checkpointed and get replayed.
    pub fn adopt(spec: ColumnSpec, path: &Path, slots: usize) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let want = (slots as u64)
            .checked_mul(spec.dim as u64 * 4)
            .ok_or_else(|| Error::Corrupt("vector file size overflows".into()))?;
        let have = file.metadata()?.len();
        if have < want {
            return Err(Error::Corrupt(format!(
                "{} holds {have} bytes, but {slots} vectors need {want}",
                path.display()
            )));
        }
        if have > want {
            file.set_len(want)?;
        }
        let mut column = Self {
            dim: spec.dim,
            slots,
            data: Data::Disk(Disk {
                file,
                path: path.to_path_buf(),
                codes: None,
            }),
        };
        if spec.quantized {
            column.quantize();
        }
        Ok(column)
    }

    /// An empty column held the same way as this one (with a new file if on disk).
    pub fn empty_like(&self, name: &str) -> Result<Self> {
        let dir = self.disk_path().and_then(Path::parent);
        Self::create(self.spec(), dir, name)
    }

    /// A copy of this column held as `spec` says, e.g. moved to or from disk.
    pub fn convert(&self, spec: ColumnSpec, dir: Option<&Path>, name: &str) -> Result<Self> {
        debug_assert_eq!(spec.dim, self.dim);
        let mut out = Self::create(spec, dir, name)?;
        for slot in 0..self.slots as u32 {
            out.push(&self.vector(slot));
        }
        Ok(out)
    }

    pub fn spec(&self) -> ColumnSpec {
        ColumnSpec {
            dim: self.dim,
            quantized: self.is_quantized(),
            on_disk: self.is_on_disk(),
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn is_quantized(&self) -> bool {
        match &self.data {
            Data::F32(_) => false,
            Data::Sq8(_) => true,
            Data::Disk(d) => d.codes.is_some(),
        }
    }

    pub fn is_on_disk(&self) -> bool {
        matches!(self.data, Data::Disk(_))
    }

    /// The file of an on-disk column.
    pub fn disk_path(&self) -> Option<&Path> {
        match &self.data {
            Data::Disk(d) => Some(&d.path),
            _ => None,
        }
    }

    /// Whether searches see approximate distances that the exact ones can
    /// improve on: quantized codes in memory with exact vectors on disk.
    pub fn rescores(&self) -> bool {
        matches!(&self.data, Data::Disk(d) if d.codes.is_some())
    }

    /// Bytes of memory the vectors take, tombstones included.
    pub fn vector_bytes(&self) -> usize {
        match &self.data {
            Data::F32(data) => data.len() * 4,
            Data::Sq8(codes) => codes.bytes(),
            Data::Disk(d) => d.codes.as_ref().map_or(0, Codes::bytes),
        }
    }

    /// Switches to scalar-quantized codes, encoding what is held. Decoded
    /// vectors re-encode to the same codes, so this is idempotent.
    pub fn quantize(&mut self) {
        let codes = match &self.data {
            Data::Sq8(_) | Data::Disk(Disk { codes: Some(_), .. }) => return,
            _ => {
                let mut codes = Codes {
                    codes: Vec::with_capacity(self.slots * self.dim),
                    ranges: Vec::with_capacity(self.slots),
                };
                for slot in 0..self.slots as u32 {
                    codes.push(&self.vector(slot));
                }
                codes
            }
        };
        match &mut self.data {
            Data::Disk(d) => d.codes = Some(codes),
            data => *data = Data::Sq8(codes),
        }
    }

    /// Switches back to full floats: the decoded values for codes in
    /// memory, the exact vectors for on-disk columns.
    pub fn dequantize(&mut self) {
        match &mut self.data {
            Data::F32(_) => {}
            Data::Disk(d) => d.codes = None,
            Data::Sq8(codes) => {
                let mut data = Vec::with_capacity(self.slots * self.dim);
                let mut v = Vec::with_capacity(self.dim);
                for slot in 0..self.slots as u32 {
                    codes.decode_into(self.dim, slot, &mut v);
                    data.extend_from_slice(&v);
                }
                self.data = Data::F32(data);
            }
        }
    }

    /// The vector in `slot`: exact for on-disk columns, decoded when
    /// quantized in memory (and so allocated unless held as full floats).
    #[inline]
    pub fn vector(&self, slot: u32) -> Cow<'_, [f32]> {
        match &self.data {
            Data::F32(data) => {
                let start = slot as usize * self.dim;
                Cow::Borrowed(&data[start..start + self.dim])
            }
            Data::Sq8(codes) => {
                let mut out = Vec::with_capacity(self.dim);
                codes.decode_into(self.dim, slot, &mut out);
                Cow::Owned(out)
            }
            Data::Disk(d) => {
                let mut out = Vec::with_capacity(self.dim);
                self.read_into(d, slot, &mut out);
                Cow::Owned(out)
            }
        }
    }

    /// `metric`'s distance from `query` to the vector in `slot`, as searches
    /// see it: from the codes when quantized, and without allocating.
    #[inline]
    pub fn distance(&self, metric: Metric, query: &[f32], slot: u32) -> f32 {
        match &self.data {
            Data::F32(data) => {
                let start = slot as usize * self.dim;
                metric.distance(query, &data[start..start + self.dim])
            }
            Data::Sq8(codes)
            | Data::Disk(Disk {
                codes: Some(codes), ..
            }) => SCRATCH.with_borrow_mut(|buf| {
                codes.decode_into(self.dim, slot, buf);
                metric.distance(query, buf)
            }),
            Data::Disk(d) => self.exact_from_disk(d, metric, query, slot),
        }
    }

    /// Like [`distance`](Self::distance), but from the exact vector when it
    /// is on disk.
    pub fn exact_distance(&self, metric: Metric, query: &[f32], slot: u32) -> f32 {
        match &self.data {
            Data::Disk(d) => self.exact_from_disk(d, metric, query, slot),
            _ => self.distance(metric, query, slot),
        }
    }

    fn exact_from_disk(&self, d: &Disk, metric: Metric, query: &[f32], slot: u32) -> f32 {
        SCRATCH.with_borrow_mut(|buf| {
            self.read_into(d, slot, buf);
            metric.distance(query, buf)
        })
    }

    fn read_into(&self, d: &Disk, slot: u32, out: &mut Vec<f32>) {
        debug_assert!((slot as usize) < self.slots);
        let len = self.dim * 4;
        BYTES.with_borrow_mut(|bytes| {
            bytes.resize(len, 0);
            read_exact_at(&d.file, bytes, slot as u64 * len as u64)
                .unwrap_or_else(|e| panic!("cannot read vector {slot} from {}: {e}", d.path.display()));
            out.clear();
            out.extend(bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
        });
    }

    /// Appends a vector in the next slot.
    pub fn push(&mut self, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dim);
        match &mut self.data {
            Data::F32(data) => data.extend_from_slice(v),
            Data::Sq8(codes) => codes.push(v),
            Data::Disk(d) => {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                let offset = self.slots as u64 * bytes.len() as u64;
                write_all_at(&d.file, &bytes, offset)
                    .unwrap_or_else(|e| panic!("cannot write to {}: {e}", d.path.display()));
                if let Some(codes) = &mut d.codes {
                    codes.push(v);
                }
            }
        }
        self.slots += 1;
    }

    /// Makes an on-disk column's file durable.
    pub fn sync(&self) -> Result<()> {
        if let Data::Disk(d) = &self.data {
            d.file.sync_data()?;
        }
        Ok(())
    }
}

fn new_file_name(name: &str) -> String {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let n = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    format!(
        "{DISK_FILE_PREFIX}{name}-{millis:x}-{:x}-{n:x}{DISK_FILE_SUFFIX}",
        std::process::id()
    )
}

/// Whether `name` is the file name of an on-disk column.
pub fn is_disk_file(name: &str) -> bool {
    name.starts_with(DISK_FILE_PREFIX) && name.ends_with(DISK_FILE_SUFFIX)
}

/// Deletes on-disk column files in `dir` other than `keep`.
pub fn remove_stale_disk_files(dir: &Path, keep: &[&Path]) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let path = entry.path();
        if name.to_str().is_some_and(is_disk_file) && !keep.contains(&path.as_path()) {
            fs::remove_file(&path)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(unix)]
fn write_all_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset)? {
            0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            n => {
                buf = &mut std::mem::take(&mut buf)[n..];
                offset += n as u64;
            }
        }
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_write(buf, offset)? {
            0 => return Err(std::io::ErrorKind::WriteZero.into()),
            n => {
                buf = &buf[n..];
                offset += n as u64;
            }
        }
    }
    Ok(())
}

fn encode(x: f32, min: f32, step: f32) -> u8 {
    if step > 0.0 {
        ((f64::from(x) - f64::from(min)) / f64::from(step))
            .round()
            .clamp(0.0, 255.0) as u8
    } else {
        0
    }
}

#[inline]
fn decode(code: u8, min: f32, step: f32) -> f32 {
    min + f32::from(code) * step
}

/// `[min, step]` for quantizing `v`. Of the step nearest `(max - min) / 255`
/// and its closest neighbouring floats, takes the one that reconstructs `v`
/// best. Decoded vectors thus get back exactly the step they were decoded
/// with, and checkpoints (which store decoded values) reload unchanged.
fn scalar_range(v: &[f32]) -> (f32, f32) {
    let Some(min) = v.iter().copied().reduce(f32::min) else {
        return (0.0, 0.0);
    };
    let max = v.iter().copied().fold(min, f32::max);
    // In f64: `max - min` can overflow f32 for finite inputs.
    let base = ((f64::from(max) - f64::from(min)) / 255.0) as f32;
    if base == 0.0 {
        return (min, 0.0);
    }
    let error = |step: f32| -> f64 {
        v.iter()
            .map(|&x| {
                let d = f64::from(decode(encode(x, min, step), min, step)) - f64::from(x);
                d * d
            })
            .sum()
    };
    let candidates = [
        base,
        base.next_down(),
        base.next_up(),
        base.next_down().next_down(),
        base.next_up().next_up(),
    ];
    let mut best = (base, error(base));
    for step in candidates
        .into_iter()
        .skip(1)
        .filter(|s| *s > 0.0 && s.is_finite())
    {
        let e = error(step);
        if e < best.1 {
            best = (step, e);
        }
    }
    (min, best.0)
}

/// Quantizes `v` to codes whose decoded values quantize to themselves, so
/// checkpoints (which store decoded values) reload exactly. Usually the first
/// encoding already is such a fixed point; rare ones get there in a round
/// or two, moving each value by far less than the quantization error.
fn quantize(v: &[f32]) -> (f32, f32, Vec<u8>) {
    let mut values = v.to_vec();
    let mut decoded = Vec::with_capacity(v.len());
    let mut last = (0.0, 0.0, Vec::new());
    for _ in 0..8 {
        let (min, step) = scalar_range(&values);
        let codes: Vec<u8> = values.iter().map(|&x| encode(x, min, step)).collect();
        decoded.clear();
        decoded.extend(codes.iter().map(|&c| decode(c, min, step)));
        let (min2, step2) = scalar_range(&decoded);
        let stable = decoded
            .iter()
            .all(|&x| decode(encode(x, min2, step2), min2, step2) == x);
        last = (min, step, codes);
        if stable {
            break;
        }
        std::mem::swap(&mut values, &mut decoded);
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(spec: ColumnSpec, dir: Option<&Path>, vectors: &[Vec<f32>]) -> VectorColumn {
        let mut c = VectorColumn::create(spec, dir, "t").unwrap();
        for v in vectors {
            c.push(v);
        }
        c
    }

    #[test]
    fn quantized_vectors_stay_close_and_reencode_identically() {
        let vectors = [
            vec![0.1, -0.7, 0.3, 0.25],
            vec![5.0, 5.0, 5.0, 5.0],
            vec![-1e30, 1e30, 0.0, 3.0],
            vec![1e-8, -1e-8, 0.0, 2e-8],
        ];
        let mut s = filled(ColumnSpec::new(4), None, &vectors);
        s.quantize();
        assert!(s.is_quantized());
        assert_eq!(s.vector_bytes(), 4 * (4 + 8));
        for (i, v) in vectors.iter().enumerate() {
            let got = s.vector(i as u32);
            let (lo, hi) = v
                .iter()
                .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
            let tolerance = (f64::from(hi) - f64::from(lo)) / 510.0 * 1.001;
            for (a, b) in v.iter().zip(got.iter()) {
                assert!(
                    (f64::from(*a) - f64::from(*b)).abs() <= tolerance,
                    "{v:?} -> {got:?}"
                );
                assert!(b.is_finite());
            }
            let q = [0.2, 0.1, -0.3, 0.5];
            assert_eq!(
                s.distance(Metric::Euclidean, &q, i as u32),
                Metric::Euclidean.distance(&q, &got)
            );
        }

        // Writing decoded values back (as checkpoints do) changes nothing.
        let mut again = VectorColumn::new(4);
        for slot in 0..s.slots() as u32 {
            again.push(&s.vector(slot));
        }
        again.quantize();
        for slot in 0..s.slots() as u32 {
            assert_eq!(again.vector(slot), s.vector(slot));
        }

        let decoded = s.vector(0).to_vec();
        s.dequantize();
        assert!(!s.is_quantized());
        assert_eq!(*s.vector(0), *decoded);
    }

    #[test]
    fn reencoding_decoded_vectors_is_exact() {
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut misses = 0;
        let trials = 50_000;
        for _ in 0..trials {
            let scale = 10f64.powf(next() * 16.0 - 8.0);
            let offset = if next() < 0.3 {
                (next() - 0.5) * scale * 100.0
            } else {
                0.0
            };
            let v: Vec<f32> = (0..16)
                .map(|_| (offset + (next() - 0.5) * scale) as f32)
                .collect();
            let (min, step, codes) = quantize(&v);
            let decoded: Vec<f32> = codes.iter().map(|&c| decode(c, min, step)).collect();
            let (min2, step2) = scalar_range(&decoded);
            let again: Vec<f32> = decoded
                .iter()
                .map(|&x| decode(encode(x, min2, step2), min2, step2))
                .collect();
            misses += usize::from(again != decoded);
        }
        assert_eq!(misses, 0, "{misses} of {trials} vectors changed when re-encoded");
    }

    #[test]
    fn disk_columns_read_back_exactly_and_take_no_memory() {
        let dir = tempfile::tempdir().unwrap();
        let vectors = vec![vec![1.0, 2.0], vec![3.0, -4.5], vec![0.1, 0.2]];
        let spec = ColumnSpec {
            on_disk: true,
            ..ColumnSpec::new(2)
        };
        let c = filled(spec, Some(dir.path()), &vectors);
        assert!(c.is_on_disk() && !c.rescores());
        assert_eq!(c.vector_bytes(), 0);
        for (i, v) in vectors.iter().enumerate() {
            assert_eq!(*c.vector(i as u32), **v);
        }
        assert_eq!(c.distance(Metric::Euclidean, &[1.0, 2.0], 0), 0.0);

        // Quantized: approximate distances from memory, exact ones from disk.
        let q = filled(
            ColumnSpec {
                quantized: true,
                ..spec
            },
            Some(dir.path()),
            &vectors,
        );
        assert!(q.rescores());
        assert_eq!(q.vector_bytes(), 3 * (2 + 8));
        assert_eq!(*q.vector(1), [3.0, -4.5], "returned vectors are exact");
        assert_eq!(q.exact_distance(Metric::Euclidean, &[0.1, 0.2], 2), 0.0);

        // Adopting keeps the first `slots` vectors and drops the rest.
        c.sync().unwrap();
        let path = c.disk_path().unwrap().to_path_buf();
        drop(c);
        let adopted = VectorColumn::adopt(spec, &path, 2).unwrap();
        assert_eq!(adopted.slots(), 2);
        assert_eq!(fs::metadata(&path).unwrap().len(), 16);
        assert!(VectorColumn::adopt(spec, &path, 3).is_err(), "too short");

        let in_memory = adopted.convert(ColumnSpec::new(2), None, "t").unwrap();
        assert!(!in_memory.is_on_disk());
        assert_eq!(*in_memory.vector(1), [3.0, -4.5]);

        let keep = q.disk_path().unwrap().to_path_buf();
        fs::write(dir.path().join("other.txt"), b"x").unwrap();
        remove_stale_disk_files(dir.path(), &[&keep]).unwrap();
        assert!(!path.exists());
        assert!(keep.exists());
        assert!(dir.path().join("other.txt").exists());
    }

    #[test]
    fn disk_columns_need_a_directory() {
        let spec = ColumnSpec {
            on_disk: true,
            ..ColumnSpec::new(2)
        };
        assert!(VectorColumn::create(spec, None, "t").is_err());
    }
}
