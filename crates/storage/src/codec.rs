use std::io::{self, Write};

use quiver_types::{Error, Result};

pub trait WriteExt: Write {
    fn put_u8(&mut self, v: u8) -> io::Result<()> {
        self.write_all(&[v])
    }

    fn put_u32(&mut self, v: u32) -> io::Result<()> {
        self.write_all(&v.to_le_bytes())
    }

    fn put_u64(&mut self, v: u64) -> io::Result<()> {
        self.write_all(&v.to_le_bytes())
    }

    fn put_f32s(&mut self, v: &[f32]) -> io::Result<()> {
        for x in v {
            self.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }

    fn put_bytes(&mut self, b: &[u8]) -> io::Result<()> {
        let len =
            u32::try_from(b.len()).map_err(|_| io::Error::other("field larger than 4 GiB"))?;
        self.put_u32(len)?;
        self.write_all(b)
    }
}

impl<W: Write + ?Sized> WriteExt for W {}

#[derive(Debug)]
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| Error::Corrupt("unexpected end of data".into()))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    pub fn f32s_into(&mut self, n: usize, out: &mut Vec<f32>) -> Result<()> {
        let bytes = self.take(
            n.checked_mul(4)
                .ok_or_else(|| Error::Corrupt("length overflow".into()))?,
        )?;
        out.clear();
        out.extend(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))),
        );
        Ok(())
    }

    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.u32()? as usize;
        self.take(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut buf = Vec::new();
        buf.put_u8(7).unwrap();
        buf.put_u32(42).unwrap();
        buf.put_u64(u64::MAX).unwrap();
        buf.put_f32s(&[1.5, -2.0]).unwrap();
        buf.put_bytes(b"hi").unwrap();

        let mut r = Reader::new(&buf);
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.u32().unwrap(), 42);
        assert_eq!(r.u64().unwrap(), u64::MAX);
        let mut v = Vec::new();
        r.f32s_into(2, &mut v).unwrap();
        assert_eq!(v, vec![1.5, -2.0]);
        assert_eq!(r.bytes().unwrap(), b"hi");
        assert!(r.is_empty());
        assert!(r.u8().is_err());
    }
}
