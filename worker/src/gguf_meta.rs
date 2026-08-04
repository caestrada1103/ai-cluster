//! Minimal GGUF header reader — just the few architecture numbers the
//! `llamaserver` admission check needs (`model_loader.rs`), read before
//! spawning `llama-server` and without loading the model. Always compiled,
//! no `llamacpp`/llama.cpp dependency. Format: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use std::io::{self, Read};
use std::path::Path;

use crate::error::WorkerError;

/// Refuse to read more than this many bytes of metadata — bounds a
/// malformed/hostile array-length field to a fast, clear error instead of an
/// unbounded read.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// The architecture numbers needed to size a KV-cache reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufArchMeta {
    /// Transformer layer count (`<arch>.block_count`).
    pub n_layer: u64,
    /// Attention heads (`<arch>.attention.head_count`).
    pub n_head: u64,
    /// KV heads (`<arch>.attention.head_count_kv`, defaults to `n_head`).
    pub n_head_kv: u64,
    /// Embedding width (`<arch>.embedding_length`).
    pub n_embd: u64,
}

impl GgufArchMeta {
    /// `n_embd / n_head`, guarded against a zero `n_head`.
    pub fn head_dim(&self) -> u64 {
        self.n_embd / self.n_head.max(1)
    }
}

/// Read [`GgufArchMeta`] from a local GGUF file's header.
pub fn read_arch_meta(path: &Path) -> Result<GgufArchMeta, WorkerError> {
    let file = std::fs::File::open(path).map_err(|e| {
        WorkerError::ModelLoad(format!("failed to open GGUF '{}': {e}", path.display()))
    })?;
    parse_arch_meta(file).map_err(|e| {
        WorkerError::ModelLoad(format!(
            "failed to read GGUF metadata from '{}': {e}",
            path.display()
        ))
    })
}

/// Value types from the GGUF spec's `gguf_metadata_value_type` enum.
mod value_type {
    pub const UINT8: u32 = 0;
    pub const INT8: u32 = 1;
    pub const UINT16: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT32: u32 = 4;
    pub const INT32: u32 = 5;
    pub const FLOAT32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const UINT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const FLOAT64: u32 = 12;
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)?;
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip<R: Read>(r: &mut R, n: u64) -> io::Result<()> {
    io::copy(&mut r.take(n), &mut io::sink())?;
    Ok(())
}

/// A scalar GGUF value, narrowed to what admission sizing needs.
enum GgufValue {
    UInt(u64),
    Other,
}

/// Read one value of type `vtype`, consuming exactly its bytes (recursing
/// into `ARRAY` elements) so the caller can keep parsing subsequent keys.
fn read_value<R: Read>(r: &mut R, vtype: u32) -> io::Result<GgufValue> {
    match vtype {
        value_type::UINT8 | value_type::INT8 | value_type::BOOL => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            Ok(GgufValue::UInt(b[0] as u64))
        }
        value_type::UINT16 | value_type::INT16 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            Ok(GgufValue::UInt(u16::from_le_bytes(b) as u64))
        }
        value_type::UINT32 => Ok(GgufValue::UInt(read_u32(r)? as u64)),
        value_type::INT32 => {
            let v = read_u32(r)? as i32;
            Ok(if v >= 0 {
                GgufValue::UInt(v as u64)
            } else {
                GgufValue::Other
            })
        }
        value_type::UINT64 => Ok(GgufValue::UInt(read_u64(r)?)),
        value_type::INT64 => {
            let v = read_u64(r)? as i64;
            Ok(if v >= 0 {
                GgufValue::UInt(v as u64)
            } else {
                GgufValue::Other
            })
        }
        value_type::FLOAT32 => {
            skip(r, 4)?;
            Ok(GgufValue::Other)
        }
        value_type::FLOAT64 => {
            skip(r, 8)?;
            Ok(GgufValue::Other)
        }
        value_type::STRING => {
            read_string(r)?;
            Ok(GgufValue::Other)
        }
        value_type::ARRAY => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                read_value(r, elem_type)?;
            }
            Ok(GgufValue::Other)
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF value type {other}"),
        )),
    }
}

/// Parse just the metadata KV section of a GGUF stream (tensor data, which
/// can be tens of GB, is never reached) and pull out the numeric keys
/// [`GgufArchMeta`] needs, matched by suffix so any `<arch>.` prefix works.
pub(crate) fn parse_arch_meta<R: Read>(reader: R) -> io::Result<GgufArchMeta> {
    let mut r = reader.take(MAX_HEADER_BYTES);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a GGUF file (bad magic)",
        ));
    }
    let version = read_u32(&mut r)?;
    if version < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported GGUF version {version} (expected >= 2)"),
        ));
    }
    let _tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;

    let mut n_layer: Option<u64> = None;
    let mut n_head: Option<u64> = None;
    let mut n_head_kv: Option<u64> = None;
    let mut n_embd: Option<u64> = None;

    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let value_type = read_u32(&mut r)?;
        let value = read_value(&mut r, value_type)?;
        let GgufValue::UInt(n) = value else { continue };
        if key.ends_with(".block_count") {
            n_layer = Some(n);
        } else if key.ends_with(".attention.head_count_kv") {
            n_head_kv = Some(n);
        } else if key.ends_with(".attention.head_count") {
            n_head = Some(n);
        } else if key.ends_with(".embedding_length") {
            n_embd = Some(n);
        }
    }

    let missing = |field: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("GGUF header has no '<arch>.{field}' key"),
        )
    };
    let n_layer = n_layer.ok_or_else(|| missing("block_count"))?;
    let n_head = n_head.ok_or_else(|| missing("attention.head_count"))?;
    let n_embd = n_embd.ok_or_else(|| missing("embedding_length"))?;
    // Absent head_count_kv means standard multi-head attention (no GQA) —
    // llama.cpp's own hparams default.
    let n_head_kv = n_head_kv.unwrap_or(n_head);

    Ok(GgufArchMeta {
        n_layer,
        n_head,
        n_head_kv,
        n_embd,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal valid GGUF byte buffer with the given uint32 metadata
    /// keys (tensor_count = 0, version = 3).
    fn build_gguf(kv: &[(&str, u32)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&(kv.len() as u64).to_le_bytes()); // kv_count
        for (key, value) in kv {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&value_type::UINT32.to_le_bytes());
            buf.extend_from_slice(&value.to_le_bytes());
        }
        buf
    }

    #[test]
    fn parses_standard_key_layout() {
        let buf = build_gguf(&[
            ("general.architecture", 0), // unmatched key — must be ignored, not misparsed
            ("qwen2.block_count", 32),
            ("qwen2.attention.head_count", 16),
            ("qwen2.attention.head_count_kv", 4),
            ("qwen2.embedding_length", 2048),
        ]);
        let meta = parse_arch_meta(Cursor::new(buf)).unwrap();
        assert_eq!(meta.n_layer, 32);
        assert_eq!(meta.n_head, 16);
        assert_eq!(meta.n_head_kv, 4);
        assert_eq!(meta.n_embd, 2048);
        assert_eq!(meta.head_dim(), 128);
    }

    #[test]
    fn head_count_kv_defaults_to_head_count_when_absent() {
        let buf = build_gguf(&[
            ("llama.block_count", 12),
            ("llama.attention.head_count", 8),
            ("llama.embedding_length", 1024),
        ]);
        let meta = parse_arch_meta(Cursor::new(buf)).unwrap();
        assert_eq!(meta.n_head_kv, 8);
    }

    #[test]
    fn missing_required_key_is_an_error() {
        let buf = build_gguf(&[
            ("llama.attention.head_count", 8),
            ("llama.embedding_length", 1024),
        ]);
        let err = parse_arch_meta(Cursor::new(buf)).unwrap_err();
        assert!(err.to_string().contains("block_count"));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = build_gguf(&[]);
        buf[0] = b'X';
        assert!(parse_arch_meta(Cursor::new(buf)).is_err());
    }

    #[test]
    fn skips_unrelated_string_and_array_values_without_derailing_parse() {
        // A STRING value and an ARRAY-of-uint32 value between two keys we
        // care about must not throw off parsing of the keys that follow.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes()); // 4 kv pairs

        // 1: general.architecture = "qwen2" (STRING)
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&value_type::STRING.to_le_bytes());
        let val = "qwen2";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val.as_bytes());

        // 2: qwen2.rope.dimension_count = [1, 2, 3] (ARRAY of UINT32)
        let key = "qwen2.rope.dimension_count";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&value_type::ARRAY.to_le_bytes());
        buf.extend_from_slice(&value_type::UINT32.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        for v in [1u32, 2, 3] {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // 3 & 4: the keys we actually need.
        for (key, value) in [
            ("qwen2.block_count", 24u32),
            ("qwen2.embedding_length", 896),
        ] {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&value_type::UINT32.to_le_bytes());
            buf.extend_from_slice(&value.to_le_bytes());
        }

        let err = parse_arch_meta(Cursor::new(buf)).unwrap_err();
        // head_count is still missing — proves the string/array values were
        // correctly skipped rather than desyncing the byte stream (which
        // would surface as a totally different error, e.g. UnexpectedEof).
        assert!(err.to_string().contains("head_count"));
    }
}
