use std::borrow::Cow;
use std::cell::RefCell;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use quiver_types::distance::dot;
use quiver_types::{Error, Metric, Quantization, Result};

/// How one column of vectors is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColumnSpec {
    pub dim: usize,
    /// Quantized codes in memory to search with (see [`Quantization`]).
    pub quantization: Quantization,
    /// The exact vectors in a file rather than in memory.
    pub on_disk: bool,
}

impl ColumnSpec {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            quantization: Quantization::None,
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
struct Sq8 {
    codes: Vec<u8>,
    ranges: Vec<[f32; 2]>,
}

impl Sq8 {
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

    fn distance(&self, metric: Metric, query: &[f32], slot: u32) -> f32 {
        SCRATCH.with_borrow_mut(|buf| {
            self.decode_into(query.len(), slot, buf);
            metric.distance(query, buf)
        })
    }

    fn bytes(&self) -> usize {
        self.codes.len() + self.ranges.len() * 8
    }
}

/// Binary-quantized vectors: one bit per component, set where it is
/// positive, `bytes_per(dim)` bytes per vector, plus each vector's mean
/// magnitude. A vector decodes to its mean magnitude times the signs.
#[derive(Debug, Clone, Default)]
struct Bits {
    bits: Vec<u8>,
    scales: Vec<f32>,
}

impl Bits {
    fn bytes_per(dim: usize) -> usize {
        dim.div_ceil(8)
    }

    fn push(&mut self, v: &[f32]) {
        let (chunks, rest) = v.as_chunks::<8>();
        self.bits.extend(chunks.iter().map(|c| sign_bits(c)));
        if !rest.is_empty() {
            self.bits.push(sign_bits(rest));
        }
        let magnitude = v.iter().map(|&x| f64::from(x.abs())).sum::<f64>();
        self.scales.push((magnitude / v.len() as f64) as f32);
    }

    fn of(&self, dim: usize, slot: u32) -> (&[u8], f32) {
        let n = Self::bytes_per(dim);
        let start = slot as usize * n;
        (&self.bits[start..start + n], self.scales[slot as usize])
    }

    /// `metric`'s distance from `query` to the decoded vector in `slot`,
    /// computed from the bits without decoding.
    fn distance(&self, metric: Metric, query: &[f32], slot: u32) -> f32 {
        let dim = query.len();
        let (bits, scale) = self.of(dim, slot);
        let signed = signed_sum(query, bits);
        match metric {
            // Cosine vectors are unit length, as are the signs over √dim.
            Metric::Cosine => 1.0 - signed / (dim as f32).sqrt(),
            Metric::DotProduct => -scale * signed,
            Metric::Euclidean => {
                let squared = dot(query, query) - 2.0 * scale * signed + scale * scale * dim as f32;
                squared.max(0.0).sqrt()
            }
            Metric::Manhattan | Metric::Hamming => SCRATCH.with_borrow_mut(|buf| {
                self.decode_into(dim, slot, buf);
                metric.distance(query, buf)
            }),
        }
    }

    fn decode_into(&self, dim: usize, slot: u32, out: &mut Vec<f32>) {
        let (bits, scale) = self.of(dim, slot);
        out.clear();
        out.extend((0..dim).map(|i| {
            if bits[i / 8] >> (i % 8) & 1 == 1 {
                scale
            } else {
                -scale
            }
        }));
    }

    fn bytes(&self) -> usize {
        self.bits.len() + self.scales.len() * 4
    }
}

/// Bit `i` set where `v[i]` is positive.
fn sign_bits(v: &[f32]) -> u8 {
    v.iter()
        .enumerate()
        .fold(0, |byte, (i, &x)| byte | (u8::from(x > 0.0) << i))
}

/// `Σ ±query[i]`: plus where bit `i` is set, minus where it is clear. The
/// dot product of `query` with the signs, without branches.
#[inline]
fn signed_sum(query: &[f32], bits: &[u8]) -> f32 {
    #[inline]
    fn signed(x: f32, byte: u8, i: usize) -> f32 {
        let clear = u32::from(!byte >> i & 1);
        f32::from_bits(x.to_bits() ^ (clear << 31))
    }
    let mut acc = [0.0f32; 8];
    let (chunks, rest) = query.as_chunks::<8>();
    for (q, &byte) in chunks.iter().zip(bits) {
        for i in 0..8 {
            acc[i] += signed(q[i], byte, i);
        }
    }
    let tail: f32 = match bits.get(chunks.len()) {
        Some(&byte) => rest.iter().enumerate().map(|(i, &x)| signed(x, byte, i)).sum(),
        None => 0.0,
    };
    acc.iter().sum::<f32>() + tail
}

/// Codes in memory that searches score instead of the exact vectors.
#[derive(Debug, Clone)]
enum Codes {
    Scalar(Sq8),
    Binary(Bits),
}

impl Codes {
    fn new(quantization: Quantization) -> Option<Self> {
        match quantization {
            Quantization::None => None,
            Quantization::Scalar => Some(Codes::Scalar(Sq8::default())),
            Quantization::Binary => Some(Codes::Binary(Bits::default())),
        }
    }

    fn quantization(&self) -> Quantization {
        match self {
            Codes::Scalar(_) => Quantization::Scalar,
            Codes::Binary(_) => Quantization::Binary,
        }
    }

    fn push(&mut self, v: &[f32]) {
        match self {
            Codes::Scalar(c) => c.push(v),
            Codes::Binary(b) => b.push(v),
        }
    }

    fn distance(&self, metric: Metric, query: &[f32], slot: u32) -> f32 {
        match self {
            Codes::Scalar(c) => c.distance(metric, query, slot),
            Codes::Binary(b) => b.distance(metric, query, slot),
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Codes::Scalar(c) => c.bytes(),
            Codes::Binary(b) => b.bytes(),
        }
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
    /// Exact vectors in memory, with binary codes to search with if `bits`.
    F32 {
        data: Vec<f32>,
        bits: Option<Bits>,
    },
    /// Scalar codes in memory, in place of the vectors.
    Sq8(Sq8),
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
            data: Data::F32 {
                data: Vec::new(),
                bits: None,
            },
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
                codes: Codes::new(spec.quantization),
            });
        } else {
            match spec.quantization {
                Quantization::None => {}
                Quantization::Scalar => column.data = Data::Sq8(Sq8::default()),
                Quantization::Binary => {
                    column.data = Data::F32 {
                        data: Vec::new(),
                        bits: Some(Bits::default()),
                    }
                }
            }
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
        column.quantize(spec.quantization);
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
            quantization: self.quantization(),
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

    /// How the codes searches score are quantized, if they are.
    pub fn quantization(&self) -> Quantization {
        match &self.data {
            Data::F32 { bits: None, .. } => Quantization::None,
            Data::F32 { bits: Some(_), .. } => Quantization::Binary,
            Data::Sq8(_) => Quantization::Scalar,
            Data::Disk(d) => d.codes.as_ref().map_or(Quantization::None, Codes::quantization),
        }
    }

    pub fn is_quantized(&self) -> bool {
        !self.quantization().is_none()
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
    /// improve on: quantized codes in memory with the exact vectors kept,
    /// on disk or (for binary codes) in memory.
    pub fn rescores(&self) -> bool {
        match &self.data {
            Data::F32 { bits, .. } => bits.is_some(),
            Data::Sq8(_) => false,
            Data::Disk(d) => d.codes.is_some(),
        }
    }

    /// Bytes of memory the vectors take, tombstones included.
    pub fn vector_bytes(&self) -> usize {
        match &self.data {
            Data::F32 { data, bits } => data.len() * 4 + bits.as_ref().map_or(0, Bits::bytes),
            Data::Sq8(codes) => codes.bytes(),
            Data::Disk(d) => d.codes.as_ref().map_or(0, Codes::bytes),
        }
    }

    /// Switches to codes quantized as `quantization` says, encoding what is
    /// held, or back to full floats for [`Quantization::None`]: the decoded
    /// values of scalar codes, the exact vectors otherwise. Decoded scalar
    /// vectors re-encode to the same codes, so this is idempotent.
    pub fn quantize(&mut self, quantization: Quantization) {
        if self.quantization() == quantization {
            return;
        }
        // Scalar codes in memory are all there is of the vectors: decode
        // them into floats to start from.
        if let Data::Sq8(_) = self.data {
            let mut data = Vec::with_capacity(self.slots * self.dim);
            for slot in 0..self.slots as u32 {
                data.extend_from_slice(&self.vector(slot));
            }
            self.data = Data::F32 { data, bits: None };
        }
        let codes = Codes::new(quantization).map(|mut codes| {
            for slot in 0..self.slots as u32 {
                codes.push(&self.vector(slot));
            }
            codes
        });
        match codes {
            Some(Codes::Scalar(sq8)) if !self.is_on_disk() => self.data = Data::Sq8(sq8),
            codes => match &mut self.data {
                Data::Disk(d) => d.codes = codes,
                Data::F32 { bits, .. } => {
                    *bits = codes.map(|c| match c {
                        Codes::Binary(b) => b,
                        Codes::Scalar(_) => unreachable!("scalar codes replace the vectors"),
                    });
                }
                Data::Sq8(_) => unreachable!("decoded above"),
            },
        }
    }

    /// The vector in `slot`: exact for on-disk columns and binary codes,
    /// decoded for scalar codes in memory (and so allocated unless held as
    /// full floats).
    #[inline]
    pub fn vector(&self, slot: u32) -> Cow<'_, [f32]> {
        match &self.data {
            Data::F32 { data, .. } => {
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
            Data::F32 {
                data: _,
                bits: Some(bits),
            } => bits.distance(metric, query, slot),
            Data::F32 { data, bits: None } => {
                let start = slot as usize * self.dim;
                metric.distance(query, &data[start..start + self.dim])
            }
            Data::Sq8(codes) => codes.distance(metric, query, slot),
            Data::Disk(Disk {
                codes: Some(codes), ..
            }) => codes.distance(metric, query, slot),
            Data::Disk(d) => self.exact_from_disk(d, metric, query, slot),
        }
    }

    /// Like [`distance`](Self::distance), but from the exact vector when it
    /// is kept: on disk, or in memory beside binary codes.
    pub fn exact_distance(&self, metric: Metric, query: &[f32], slot: u32) -> f32 {
        match &self.data {
            Data::Disk(d) => self.exact_from_disk(d, metric, query, slot),
            Data::F32 { data, .. } => {
                let start = slot as usize * self.dim;
                metric.distance(query, &data[start..start + self.dim])
            }
            Data::Sq8(_) => self.distance(metric, query, slot),
        }
    }

    /// The distance indexes are built with: exact where binary codes are
    /// too coarse to place a vector among its neighbours, else as searches
    /// see it, so scalar codes spare on-disk columns a read per distance.
    pub fn index_distance(&self, metric: Metric, query: &[f32], slot: u32) -> f32 {
        if self.quantization() == Quantization::Binary {
            self.exact_distance(metric, query, slot)
        } else {
            self.distance(metric, query, slot)
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
            Data::F32 { data, bits } => {
                data.extend_from_slice(v);
                if let Some(bits) = bits {
                    bits.push(v);
                }
            }
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
        s.quantize(Quantization::Scalar);
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
        again.quantize(Quantization::Scalar);
        for slot in 0..s.slots() as u32 {
            assert_eq!(again.vector(slot), s.vector(slot));
        }

        let decoded = s.vector(0).to_vec();
        s.quantize(Quantization::None);
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
                quantization: Quantization::Scalar,
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

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    #[test]
    fn binary_distances_match_the_decoded_vectors() {
        // 13 components: a partial last byte.
        let vectors = random_vectors(20, 13, 3);
        let mut c = filled(
            ColumnSpec {
                quantization: Quantization::Binary,
                ..ColumnSpec::new(13)
            },
            None,
            &vectors,
        );
        assert!(c.rescores());
        assert_eq!(c.quantization(), Quantization::Binary);
        assert_eq!(
            c.vector_bytes(),
            20 * (13 * 4 + 2 + 4),
            "exact vectors plus codes"
        );
        let mut query = random_vectors(1, 13, 9).remove(0);
        Metric::Cosine.prepare(&mut query);
        for (slot, v) in vectors.iter().enumerate() {
            let slot = slot as u32;
            assert_eq!(*c.vector(slot), **v, "returned vectors are exact");
            let Data::F32 { bits: Some(bits), .. } = &c.data else {
                panic!()
            };
            let mut decoded = Vec::new();
            bits.decode_into(13, slot, &mut decoded);
            let scale = (v.iter().map(|x| f64::from(x.abs())).sum::<f64>() / 13.0) as f32;
            for (d, x) in decoded.iter().zip(v) {
                assert_eq!(*d, if *x > 0.0 { scale } else { -scale });
            }
            for metric in [Metric::Euclidean, Metric::DotProduct, Metric::Manhattan] {
                let want = metric.distance(&query, &decoded);
                let got = c.distance(metric, &query, slot);
                assert!((want - got).abs() < 1e-4, "{metric:?}: {want} vs {got}");
                assert_eq!(c.exact_distance(metric, &query, slot), metric.distance(&query, v));
                assert_eq!(c.index_distance(metric, &query, slot), metric.distance(&query, v));
            }
            let signs: Vec<f32> = decoded.iter().map(|d| d.signum() / 13f32.sqrt()).collect();
            let want = Metric::Cosine.distance(&query, &signs);
            assert!((want - c.distance(Metric::Cosine, &query, slot)).abs() < 1e-5);
        }

        // Switching between layouts keeps the exact vectors where they are kept.
        c.quantize(Quantization::Scalar);
        assert_eq!(c.quantization(), Quantization::Scalar);
        assert!(!c.rescores());
        c.quantize(Quantization::Binary);
        assert_eq!(c.quantization(), Quantization::Binary);
        c.quantize(Quantization::None);
        assert!(!c.is_quantized() && !c.rescores());
        assert_eq!(c.vector_bytes(), 20 * 13 * 4);
        c.quantize(Quantization::Binary);
        assert_eq!(
            *c.vector(4),
            *c.convert(ColumnSpec::new(13), None, "t").unwrap().vector(4)
        );
    }

    #[test]
    fn binary_codes_on_disk_take_a_bit_per_component() {
        let dir = tempfile::tempdir().unwrap();
        let vectors = random_vectors(10, 64, 5);
        let spec = ColumnSpec {
            dim: 64,
            quantization: Quantization::Binary,
            on_disk: true,
        };
        let c = filled(spec, Some(dir.path()), &vectors);
        assert!(c.rescores());
        assert_eq!(c.vector_bytes(), 10 * (8 + 4));
        assert_eq!(*c.vector(3), *vectors[3]);
        c.sync().unwrap();
        let path = c.disk_path().unwrap().to_path_buf();
        drop(c);
        let adopted = VectorColumn::adopt(spec, &path, 10).unwrap();
        assert_eq!(adopted.quantization(), Quantization::Binary);
        let q = &vectors[7];
        assert_eq!(adopted.exact_distance(Metric::Euclidean, q, 7), 0.0);
        assert!(
            adopted.distance(Metric::Euclidean, q, 7) > 0.0,
            "codes are approximate"
        );
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
