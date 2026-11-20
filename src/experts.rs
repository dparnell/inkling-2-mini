//! MoE expert weights: kept in their checkpoint formats (NVFP4 / MXFP8) and cached on the GPU
//! in fixed-size slots. Missing experts are uploaded from the memory-mapped checkpoint on demand
//! with LRU eviction, so the model runs on GPUs that cannot hold all 15+ GB of experts.

use crate::gpu::{Buf, Gpu};
use crate::weights::SafeTensors;
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::sync::Arc;

pub const FMT_NVFP4: u32 = 0;
pub const FMT_MXFP8: u32 = 1;

/// Word offsets of one expert slot: [q13 | s13 | q2 | s2 | f13 f2 | pad]
#[derive(Clone, Copy, Debug)]
pub struct SlotLayout {
    pub fmt: u32,
    pub hidden: u32,
    pub inter: u32,
    pub qw13: u32,
    pub sw13: u32,
    pub qw2: u32,
    pub sw2: u32,
    pub off_s13: u32,
    pub off_q2: u32,
    pub off_s2: u32,
    pub off_f13: u32,
    pub off_f2: u32,
    pub slot_words: u32,
}

impl SlotLayout {
    pub fn new(fmt: u32, hidden: u32, inter: u32) -> Self {
        // w13: [2 * inter rows][hidden], w2: [hidden rows][inter]
        let (qw13, sw13, qw2, sw2) = if fmt == FMT_NVFP4 {
            (hidden / 8, hidden / 16 / 4, inter / 8, inter / 16 / 4)
        } else {
            (hidden / 4, hidden / 32 / 4, inter / 4, inter / 32 / 4)
        };
        let rows13 = 2 * inter;
        let rows2 = hidden;
        let off_s13 = rows13 * qw13;
        let off_q2 = off_s13 + rows13 * sw13;
        let off_s2 = off_q2 + rows2 * qw2;
        let off_f13 = off_s2 + rows2 * sw2;
        let off_f2 = off_f13 + 1;
        let slot_words = (off_f2 + 1).next_multiple_of(64);
        Self { fmt, hidden, inter, qw13, sw13, qw2, sw2, off_s13, off_q2, off_s2, off_f13, off_f2, slot_words }
    }

    pub fn slot_bytes(&self) -> u64 {
        self.slot_words as u64 * 4
    }
}

/// Where one layer's experts live in the checkpoint.
pub struct LayerSrc {
    pub fmt: u32,
    w13: String,
    s13: String,
    w2: String,
    s2: String,
    f13: Vec<f32>,
    f2: Vec<f32>,
}

pub struct Pool {
    pub buf: Buf,
    pub slots: u32,
}

/// One cache per weight format (slot sizes differ).
pub struct Cache {
    pub layout: SlotLayout,
    pub pools: Vec<Pool>,
    per_pool: u32,
    owner: Vec<Option<(u16, u16)>>,
    stamp: Vec<u64>,
    map: HashMap<(u16, u16), u32>,
    pub hits: u64,
    pub misses: u64,
    pub uploaded_bytes: u64,
}

impl Cache {
    fn new(gpu: &Gpu, layout: SlotLayout, slots: u32, label: &str) -> Self {
        let max_bind = gpu.max_binding().min(1 << 31);
        let per_pool = ((max_bind / layout.slot_bytes()) as u32).max(1).min(slots);
        let mut pools = Vec::new();
        let mut left = slots;
        while left > 0 {
            let n = left.min(per_pool);
            pools.push(Pool { buf: gpu.storage(label, n as u64 * layout.slot_bytes()), slots: n });
            left -= n;
        }
        Self {
            layout,
            pools,
            per_pool,
            owner: vec![None; slots as usize],
            stamp: vec![0; slots as usize],
            map: HashMap::new(),
            hits: 0,
            misses: 0,
            uploaded_bytes: 0,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.owner.len() as u32
    }

    /// (pool index, slot within pool)
    pub fn locate(&self, slot: u32) -> (usize, u32) {
        ((slot / self.per_pool) as usize, slot % self.per_pool)
    }
}

pub struct ExpertStore {
    st: Arc<SafeTensors>,
    /// Decayed use counts per (layer, expert): the eviction score (LFU with aging), so a
    /// frequently used set stays resident instead of thrashing under cyclic prefill access.
    freq: Vec<Vec<u32>>,
    calls: u64,
    pub layers: Vec<LayerSrc>,
    pub caches: Vec<Cache>,
    clock: u64,
    pub n_experts: usize,
}

impl ExpertStore {
    /// `budget` bytes of VRAM for expert slots, split across formats in proportion to their
    /// total size, but never below one layer's worth of experts per format (needed for prefill).
    pub fn new(gpu: &Gpu, st: Arc<SafeTensors>, n_layers: usize, n_experts: usize, hidden: usize, inter: usize, budget: u64) -> Result<Self> {
        let mut layers = Vec::new();
        for l in 0..n_layers {
            let p = format!("model.llm.layers.{l}.mlp.experts.");
            let w13 = format!("{p}w13_weight");
            let t = st.get(&w13)?;
            let fmt = match t.dtype {
                crate::weights::DType::U8 => FMT_NVFP4,
                crate::weights::DType::F8E4M3 => FMT_MXFP8,
                d => bail!("unexpected expert dtype {d:?} in layer {l}"),
            };
            let (f13, f2) = if fmt == FMT_NVFP4 {
                let fac = |w: &str| -> Result<Vec<f32>> {
                    let amax = st.get(&format!("{p}{w}_weight_amax"))?.to_f32();
                    let den = st.get(&format!("{p}{w}_weight_outer_scale_denominator"))?.to_f32()[0];
                    Ok(amax.iter().map(|a| a / den).collect())
                };
                (fac("w13")?, fac("w2")?)
            } else {
                (vec![1.0; n_experts], vec![1.0; n_experts])
            };
            layers.push(LayerSrc { fmt, w13, s13: format!("{p}w13_weight_scale"), w2: format!("{p}w2_weight"), s2: format!("{p}w2_weight_scale"), f13, f2 });
        }
        let mut caches = Vec::new();
        let fmts = [FMT_NVFP4, FMT_MXFP8];
        let layouts: Vec<SlotLayout> = fmts.iter().map(|&f| SlotLayout::new(f, hidden as u32, inter as u32)).collect();
        let totals: Vec<u64> = fmts
            .iter()
            .zip(&layouts)
            .map(|(&f, lay)| layers.iter().filter(|l| l.fmt == f).count() as u64 * n_experts as u64 * lay.slot_bytes())
            .collect();
        let all: u64 = totals.iter().sum();
        for (i, &f) in fmts.iter().enumerate() {
            let n_layers_f = layers.iter().filter(|l| l.fmt == f).count();
            if n_layers_f == 0 {
                caches.push(Cache::new(gpu, layouts[i], 1, "experts"));
                continue;
            }
            let share = (budget as f64 * totals[i] as f64 / all as f64) as u64;
            let min_slots = n_experts as u64;
            let max_slots = (n_layers_f * n_experts) as u64;
            let slots = (share / layouts[i].slot_bytes()).clamp(min_slots, max_slots) as u32;
            tracing::info!(
                "expert cache {}: {} slots of {:.2} MB ({:.2} GB, {:.0}% of {} experts)",
                if f == FMT_NVFP4 { "NVFP4" } else { "MXFP8" },
                slots,
                layouts[i].slot_bytes() as f64 / 1e6,
                slots as f64 * layouts[i].slot_bytes() as f64 / 1e9,
                100.0 * slots as f64 / max_slots as f64,
                max_slots
            );
            caches.push(Cache::new(gpu, layouts[i], slots, "experts"));
        }
        // Warm the OS page cache with the expert weights in the background, so the first
        // requests stream experts from RAM instead of faulting them in from disk.
        let names: Vec<String> = layers.iter().flat_map(|l| [l.w13.clone(), l.s13.clone(), l.w2.clone(), l.s2.clone()]).collect();
        let stw = st.clone();
        std::thread::Builder::new()
            .name("expert-prefetch".into())
            .spawn(move || {
                let t0 = std::time::Instant::now();
                let mut sum = 0u64;
                let mut bytes = 0usize;
                for n in &names {
                    if let Ok(t) = stw.get(n) {
                        for page in t.data.chunks(4096) {
                            sum = sum.wrapping_add(page[0] as u64);
                        }
                        bytes += t.data.len();
                    }
                }
                tracing::info!("expert weights paged in ({:.1} GB in {:.0} s, {sum:x})", bytes as f64 / 1e9, t0.elapsed().as_secs_f64());
            })
            .ok();
        Ok(Self { st, freq: vec![vec![0; n_experts]; n_layers], calls: 0, layers, caches, clock: 1, n_experts })
    }

    pub fn total_bytes(&self) -> u64 {
        self.caches.iter().map(|c| c.capacity() as u64 * c.layout.slot_bytes()).sum()
    }

    /// Make the given experts of `layer` resident; returns the global slot of each (same order).
    /// Experts in `experts` are never evicted by this call.
    pub fn ensure(&mut self, gpu: &Gpu, layer: usize, experts: &[u16]) -> Result<Vec<u32>> {
        self.clock += 1;
        let clock = self.clock;
        self.calls += 1;
        if self.calls % 2048 == 0 {
            for l in self.freq.iter_mut() {
                for f in l.iter_mut() {
                    *f /= 2;
                }
            }
        }
        for &e in experts {
            self.freq[layer][e as usize] = self.freq[layer][e as usize].saturating_add(1);
        }
        let freq = &self.freq;
        let fmt = self.layers[layer].fmt as usize;
        let cache = &mut self.caches[fmt];
        if experts.len() > cache.capacity() as usize {
            bail!("expert cache too small: {} experts needed, {} slots", experts.len(), cache.capacity());
        }
        let mut out = Vec::with_capacity(experts.len());
        let mut missing = Vec::new();
        for (i, &e) in experts.iter().enumerate() {
            if let Some(&s) = cache.map.get(&(layer as u16, e)) {
                cache.stamp[s as usize] = clock;
                cache.hits += 1;
                out.push(s);
            } else {
                cache.misses += 1;
                missing.push(i);
                out.push(u32::MAX);
            }
        }
        if missing.is_empty() {
            return Ok(out);
        }
        // pick victims: free slots first, then the least frequently used (ties: least recent),
        // never one stamped by this call
        let mut order: Vec<u32> = (0..cache.capacity()).filter(|&s| cache.stamp[s as usize] != clock).collect();
        order.sort_unstable_by_key(|&s| match cache.owner[s as usize] {
            None => (0u8, 0u32, 0u64),
            Some((l, e)) => (1, freq[l as usize][e as usize], cache.stamp[s as usize]),
        });
        let src = &self.layers[layer];
        let lay = cache.layout;
        for (k, &i) in missing.iter().enumerate() {
            let slot = order[k];
            if let Some(old) = cache.owner[slot as usize].take() {
                cache.map.remove(&old);
            }
            let e = experts[i];
            cache.owner[slot as usize] = Some((layer as u16, e));
            cache.map.insert((layer as u16, e), slot);
            cache.stamp[slot as usize] = clock;
            out[i] = slot;
            // upload
            let (pi, local) = cache.locate(slot);
            let base = local as u64 * lay.slot_bytes();
            let buf = &cache.pools[pi].buf.raw;
            let parts: [(&str, u32); 4] = [(&src.w13, 0), (&src.s13, lay.off_s13), (&src.w2, lay.off_q2), (&src.s2, lay.off_s2)];
            for (name, off) in parts {
                let t = self.st.get(name)?;
                let per = t.data.len() / self.n_experts;
                let bytes = &t.data[e as usize * per..(e as usize + 1) * per];
                gpu.queue.write_buffer(buf, base + off as u64 * 4, bytes);
                cache.uploaded_bytes += per as u64;
            }
            let f = [src.f13[e as usize], src.f2[e as usize]];
            gpu.queue.write_buffer(buf, base + lay.off_f13 as u64 * 4, bytemuck::cast_slice(&f));
        }
        Ok(out)
    }

    pub fn layout(&self, layer: usize) -> SlotLayout {
        self.caches[self.layers[layer].fmt as usize].layout
    }

    pub fn cache(&self, layer: usize) -> &Cache {
        &self.caches[self.layers[layer].fmt as usize]
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        self.caches.iter().fold((0, 0, 0), |a, c| (a.0 + c.hits, a.1 + c.misses, a.2 + c.uploaded_bytes))
    }
}
