//! Memory-mapped safetensors access (single file or sharded with an index).

use anyhow::{Context, Result};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    BF16,
    F16,
    F32,
    U8,
    F8E4M3,
}

struct Entry {
    file: usize,
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

pub struct SafeTensors {
    maps: Vec<Mmap>,
    entries: HashMap<String, Entry>,
}

pub struct TensorView<'a> {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl SafeTensors {
    pub fn open(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let files: Vec<String> = if index.exists() {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
            let mut f: Vec<String> = v["weight_map"]
                .as_object()
                .context("bad index")?
                .values()
                .filter_map(|x| x.as_str().map(String::from))
                .collect();
            f.sort();
            f.dedup();
            f
        } else {
            vec!["model.safetensors".to_string()]
        };
        let mut maps = Vec::new();
        let mut entries = HashMap::new();
        for (fi, name) in files.iter().enumerate() {
            let path = dir.join(name);
            let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: the file is opened read-only and not expected to be modified while mapped.
            let map = unsafe { Mmap::map(&file)? };
            let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
            let header: Value = serde_json::from_slice(&map[8..8 + n])?;
            let base = 8 + n;
            for (k, v) in header.as_object().context("bad header")? {
                if k == "__metadata__" {
                    continue;
                }
                let dtype = match v["dtype"].as_str() {
                    Some("BF16") => DType::BF16,
                    Some("F16") => DType::F16,
                    Some("F32") => DType::F32,
                    Some("U8") => DType::U8,
                    Some("F8_E4M3") => DType::F8E4M3,
                    _ => continue,
                };
                let shape = v["shape"]
                    .as_array()
                    .context("shape")?
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect();
                let off = v["data_offsets"].as_array().context("offsets")?;
                entries.insert(
                    k.clone(),
                    Entry {
                        file: fi,
                        dtype,
                        shape,
                        start: base + off[0].as_u64().unwrap() as usize,
                        end: base + off[1].as_u64().unwrap() as usize,
                    },
                );
            }
            maps.push(map);
        }
        Ok(Self { maps, entries })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    pub fn get(&self, name: &str) -> Result<TensorView<'_>> {
        let e = self
            .entries
            .get(name)
            .with_context(|| format!("tensor `{name}` not found in checkpoint"))?;
        Ok(TensorView {
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: &self.maps[e.file][e.start..e.end],
        })
    }
}

#[inline]
fn f32_to_bf16_bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

impl TensorView<'_> {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The tensor as raw bf16 bit patterns (converted if stored otherwise).
    pub fn to_bf16_bytes(&self) -> Vec<u8> {
        match self.dtype {
            DType::BF16 => self.data.to_vec(),
            DType::F16 => self
                .data
                .chunks_exact(2)
                .flat_map(|c| {
                    let f = half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                    f32_to_bf16_bits(f).to_le_bytes()
                })
                .collect(),
            DType::F32 => self
                .data
                .chunks_exact(4)
                .flat_map(|c| {
                    f32_to_bf16_bits(f32::from_le_bytes(c.try_into().unwrap())).to_le_bytes()
                })
                .collect(),
            DType::U8 | DType::F8E4M3 => panic!("tensor of dtype {:?} cannot be converted to bf16", self.dtype),
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            DType::BF16 => self
                .data
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            DType::F16 => self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            DType::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            DType::U8 => self.data.iter().map(|&b| b as f32).collect(),
            DType::F8E4M3 => self.data.iter().map(|&b| fp8_e4m3_to_f32(b)).collect(),
        }
    }
}

/// OCP FP8 E4M3 (fn variant: no infinities, 0x7f/0xff are NaN).
pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 0xf && (b & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 { s * m / 8.0 * 2f32.powi(-6) } else { s * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}
