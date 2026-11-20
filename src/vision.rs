//! Inkling-2 vision tower (native-resolution ViT, 20 px patches, 2x2 merge into one LM token)
//! following transformers' `Inkling2VisionModel` and `Inkling2ImageProcessor`:
//! * the image is kept at native size (optionally downscaled to a token budget here), padded to a
//!   multiple of 40 px with -1 on the 0..255 scale, normalised to [-1, 1];
//! * patches are ordered by merge block (raster), then 2x2 within the block;
//! * a class token leads the sequence; learned 32x32 position grid bilinearly interpolated;
//! * 2-D axial RoPE (theta 1e4) on (row + 1, col + 1), class token at (0, 0);
//! * 23 pre-LN blocks with layer scale; merger = Linear(4096) -> GELU -> Linear(2304).

use crate::gpu::{Buf, Gpu, K, Recorder, Slice, grid};
use crate::model::{Matrix, f32_bytes};
use crate::weights::SafeTensors;
use anyhow::{Context, Result, bail};
use std::hash::{Hash, Hasher};

const HID: usize = 1024;
const HEADS: usize = 16;
const HD: usize = 64;
const FF: usize = 4096;
const PATCH: usize = 20;
const MERGE: usize = 2;
const BLOCK: usize = PATCH * MERGE;
const PDIM: usize = 3 * PATCH * PATCH; // 1200
const PDIM_PAD: usize = 1216; // multiple of 32 for the GEMM
const GRID: usize = 32;
const TEXT_HID: usize = 2304;

/// A preprocessed image: patch vectors in merge-block order.
pub struct ImageInput {
    pub rows: usize, // merge blocks vertically
    pub cols: usize,
    pub patches: Vec<f32>, // [rows * cols * 4][PDIM]
}

impl ImageInput {
    pub fn n_tokens(&self) -> usize {
        self.rows * self.cols
    }

    pub fn content_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.rows, self.cols).hash(&mut h);
        bytemuck::cast_slice::<f32, u8>(&self.patches).hash(&mut h);
        h.finish()
    }

    /// Decode an encoded image (png/jpeg/webp/gif/bmp) and preprocess it. Images larger than
    /// `max_tokens` merge blocks are downscaled (aspect preserved) to fit.
    pub fn from_bytes(bytes: &[u8], max_tokens: usize) -> Result<Self> {
        let img = image::load_from_memory(bytes).context("decoding image")?;
        let (w, h) = (img.width() as usize, img.height() as usize);
        if w == 0 || h == 0 {
            bail!("empty image");
        }
        let rgba = img.to_rgba8();
        let has_alpha = rgba.pixels().any(|p| p[3] != 255);
        // composite transparency onto the reference's gray checkerboard
        let mut rgb = image::RgbImage::new(w as u32, h as u32);
        for (x, y, p) in rgba.enumerate_pixels() {
            let bg: i32 = if ((x / 8 + y / 8) % 2) == 0 { 204 } else { 153 };
            let a = p[3] as i32;
            let mut o = [0u8; 3];
            for c in 0..3 {
                o[c] = if has_alpha { ((p[c] as i32 * a + bg * (255 - a) + 127) / 255) as u8 } else { p[c] };
            }
            rgb.put_pixel(x, y, image::Rgb(o));
        }
        let tokens = w.div_ceil(BLOCK) * h.div_ceil(BLOCK);
        let rgb = if tokens > max_tokens {
            let s = (max_tokens as f64 / tokens as f64).sqrt();
            let mut nw = ((w as f64 * s).floor() as usize).max(1);
            let mut nh = ((h as f64 * s).floor() as usize).max(1);
            while nw.div_ceil(BLOCK) * nh.div_ceil(BLOCK) > max_tokens {
                nw = (nw as f64 * 0.98) as usize;
                nh = (nh as f64 * 0.98) as usize;
            }
            tracing::info!("downscaling image {w}x{h} -> {nw}x{nh} ({tokens} -> {} tokens)", nw.div_ceil(BLOCK) * nh.div_ceil(BLOCK));
            image::imageops::resize(&rgb, nw as u32, nh as u32, image::imageops::FilterType::CatmullRom)
        } else {
            rgb
        };
        Ok(Self::from_rgb(&rgb))
    }

    pub fn from_rgb(rgb: &image::RgbImage) -> Self {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let rows = h.div_ceil(BLOCK);
        let cols = w.div_ceil(BLOCK);
        let pad = (-1.0f32 / 255.0 - 0.5) / 0.5;
        let px = |x: usize, y: usize, c: usize| -> f32 {
            if x < w && y < h { (rgb.get_pixel(x as u32, y as u32)[c] as f32 / 255.0 - 0.5) / 0.5 } else { pad }
        };
        let mut patches = Vec::with_capacity(rows * cols * 4 * PDIM);
        for br in 0..rows {
            for bc in 0..cols {
                for mr in 0..MERGE {
                    for mc in 0..MERGE {
                        let y0 = br * BLOCK + mr * PATCH;
                        let x0 = bc * BLOCK + mc * PATCH;
                        for py in 0..PATCH {
                            for pxx in 0..PATCH {
                                for c in 0..3 {
                                    patches.push(px(x0 + pxx, y0 + py, c));
                                }
                            }
                        }
                    }
                }
            }
        }
        Self { rows, cols, patches }
    }
}

struct Layer {
    ln1_w: Buf,
    ln1_b: Buf,
    ln2_w: Buf,
    ln2_b: Buf,
    ls1: Buf,
    ls2: Buf,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    qb: Buf,
    kb: Buf,
    vb: Buf,
    ob: Buf,
    fc: Matrix,
    fcb: Buf,
    pr: Matrix,
    prb: Buf,
}

pub struct Vision {
    patch: Matrix,
    pos: Vec<f32>, // [GRID * GRID][HID]
    cls: Vec<f32>,
    ln_pre_w: Buf,
    ln_pre_b: Buf,
    layers: Vec<Layer>,
    proj_in: Matrix,
    proj_out: Matrix,
    zeros: Buf,
    max_n: usize,
    pub max_tokens: usize,
    b: Bufs,
}

struct Bufs {
    pin: Buf,
    pos: Buf,
    pe: Buf,
    h: Buf,
    h2: Buf,
    x: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    ao: Buf,
    o: Buf,
    f: Buf,
    cs: Buf,
    m1: Buf,
    out: Buf,
    stage: Buf,
}

impl Vision {
    pub fn load(gpu: &Gpu, st: &SafeTensors, max_tokens: usize) -> Result<Self> {
        let p = "model.visual.";
        let f = |n: &str| -> Result<Vec<f32>> { Ok(st.get(&format!("{p}{n}"))?.to_f32()) };
        let v = |n: &str| -> Result<Buf> { Ok(gpu.storage_init(n, f32_bytes(&f(n)?))) };
        let mat = |n: &str| -> Result<Matrix> {
            let t = st.get(&format!("{p}{n}"))?;
            Ok(Matrix::upload(gpu, n, &t.to_bf16_bytes(), t.shape[0], t.shape[1]))
        };
        // patch embedding padded from 1200 to 1216 input columns
        let pw = st.get(&format!("{p}backbone.embeddings.patch_embedding.weight"))?.to_bf16_bytes();
        let mut padded = vec![0u8; HID * PDIM_PAD * 2];
        for r in 0..HID {
            padded[r * PDIM_PAD * 2..r * PDIM_PAD * 2 + PDIM * 2].copy_from_slice(&pw[r * PDIM * 2..(r + 1) * PDIM * 2]);
        }
        let patch = Matrix::upload(gpu, "patch_embedding", &padded, HID, PDIM_PAD);
        let cls_e = f("backbone.embeddings.class_embedding")?;
        let cls_p = f("backbone.embeddings.cls_position_embedding")?;
        let cls: Vec<f32> = cls_e.iter().zip(&cls_p).map(|(a, b)| a + b).collect();
        let mut layers = Vec::new();
        for i in 0.. {
            let b = format!("backbone.layers.{i}.");
            if !st.contains(&format!("{p}{b}ln_1.weight")) {
                break;
            }
            layers.push(Layer {
                ln1_w: v(&format!("{b}ln_1.weight"))?,
                ln1_b: v(&format!("{b}ln_1.bias"))?,
                ln2_w: v(&format!("{b}ln_2.weight"))?,
                ln2_b: v(&format!("{b}ln_2.bias"))?,
                ls1: v(&format!("{b}ls_1.gamma"))?,
                ls2: v(&format!("{b}ls_2.gamma"))?,
                q: mat(&format!("{b}self_attn.q_proj.weight"))?,
                k: mat(&format!("{b}self_attn.k_proj.weight"))?,
                v: mat(&format!("{b}self_attn.v_proj.weight"))?,
                o: mat(&format!("{b}self_attn.out_proj.weight"))?,
                qb: v(&format!("{b}self_attn.q_proj.bias"))?,
                kb: v(&format!("{b}self_attn.k_proj.bias"))?,
                vb: v(&format!("{b}self_attn.v_proj.bias"))?,
                ob: v(&format!("{b}self_attn.out_proj.bias"))?,
                fc: mat(&format!("{b}mlp.c_fc.weight"))?,
                fcb: v(&format!("{b}mlp.c_fc.bias"))?,
                pr: mat(&format!("{b}mlp.c_proj.weight"))?,
                prb: v(&format!("{b}mlp.c_proj.bias"))?,
            });
        }
        let max_n = max_tokens * 4 + 1;
        let s = |n: usize| gpu.storage("vision", (n * 4) as u64);
        let b = Bufs {
            pin: s((max_n - 1) * PDIM_PAD),
            pos: s((max_n - 1) * HID),
            pe: s((max_n - 1) * HID),
            h: s(max_n * HID),
            h2: s(max_n * HID),
            x: s(max_n * HID),
            q: s(max_n * HID),
            k: s(max_n * HID),
            v: s(max_n * HID),
            ao: s(max_n * HID),
            o: s(max_n * HID),
            f: s(max_n * FF),
            cs: s(max_n * HD),
            m1: s(max_tokens * FF),
            out: s(max_tokens * TEXT_HID),
            stage: gpu.readback_buffer((max_tokens * TEXT_HID * 4) as u64),
        };
        tracing::info!("vision tower loaded ({} layers, up to {max_tokens} tokens per image)", layers.len());
        Ok(Self {
            patch,
            pos: f("backbone.embeddings.position_embedding")?,
            cls,
            ln_pre_w: v("backbone.ln_pre.weight")?,
            ln_pre_b: v("backbone.ln_pre.bias")?,
            layers,
            proj_in: mat("proj_in.weight")?,
            proj_out: mat("proj_out.weight")?,
            zeros: gpu.storage("zeros", (FF * 4) as u64),
            max_n,
            max_tokens,
            b,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(gpu: &mut Gpu, rec: &mut Recorder, w: &Matrix, x: Slice, y: &Buf, t: u32, bias: Option<(&Buf, u32)>) {
        let k = w.k as u32;
        let n = w.n as u32;
        for (buf, r0, rows) in &w.chunks {
            let rows = *rows as u32;
            let tiles_n = rows.div_ceil(64);
            gpu.dispatch(rec, K::Gemm, &[t, k, rows, k, n, *r0 as u32, tiles_n], &[x, buf.all(), y.all()], grid(tiles_n * t.div_ceil(64)));
        }
        if let Some((b, act)) = bias {
            gpu.dispatch(rec, K::BiasAct, &[t, n, n, act, 0], &[b.all(), y.all()], grid((t * n).div_ceil(256)));
        }
    }

    fn layernorm(gpu: &mut Gpu, rec: &mut Recorder, x: &Buf, w: &Buf, b: &Buf, y: &Buf, rows: u32) {
        gpu.dispatch(rec, K::LayerNorm, &[rows, HID as u32, HID as u32, HID as u32, 1e-5f32.to_bits()], &[x.all(), w.all(), b.all(), y.all()], grid(rows));
    }

    /// LM-space features for every image token, [tokens][2304].
    pub fn encode(&mut self, gpu: &mut Gpu, img: &ImageInput) -> Result<Vec<f32>> {
        let ntok = img.n_tokens();
        if ntok > self.max_tokens {
            bail!("image has {ntok} tokens, more than the vision budget of {}", self.max_tokens);
        }
        let np = ntok * 4;
        let n = np + 1;
        assert!(n <= self.max_n);
        let (gh, gw) = (img.rows * MERGE, img.cols * MERGE);

        // host: padded patch rows, interpolated position embeddings, RoPE table
        let mut pin = vec![0f32; np * PDIM_PAD];
        for i in 0..np {
            pin[i * PDIM_PAD..i * PDIM_PAD + PDIM].copy_from_slice(&img.patches[i * PDIM..(i + 1) * PDIM]);
        }
        let mut pos = vec![0f32; np * HID];
        let mut cs = vec![0f32; n * HD];
        let inv_freq: Vec<f32> = (0..HD / 4).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / (HD / 2) as f32)).collect();
        let taps = |idx: usize, size: usize| -> [(usize, f32); 2] {
            let src = (idx as f32 + 0.5) * GRID as f32 / size as f32 - 0.5;
            let f0 = src.floor();
            let mut out = [(0usize, 0f32); 2];
            for (j, o) in out.iter_mut().enumerate() {
                let t = f0 + j as f32;
                let w = (1.0 - (src - t).abs()).max(0.0);
                *o = (t.clamp(0.0, (GRID - 1) as f32) as usize, w);
            }
            out
        };
        let blocks_w = gw / MERGE;
        let rope_row = |row: usize, r: f32, c: f32, cs: &mut [f32]| {
            // [cos/sin pairs] for 32 pair slots: first 16 from width (col), then 16 from height (row)
            for (j, &f) in inv_freq.iter().enumerate() {
                let (a, b) = ((c * f).cos(), (c * f).sin());
                cs[row * HD + 2 * j] = a;
                cs[row * HD + 2 * j + 1] = b;
                let (a, b) = ((r * f).cos(), (r * f).sin());
                cs[row * HD + 2 * (j + HD / 4)] = a;
                cs[row * HD + 2 * (j + HD / 4) + 1] = b;
            }
        };
        rope_row(0, 0.0, 0.0, &mut cs);
        for i in 0..np {
            let row = i / (MERGE * MERGE * blocks_w) * MERGE + i / MERGE % MERGE;
            let col = i / (MERGE * MERGE) % blocks_w * MERGE + i % MERGE;
            let rt = taps(row, gh);
            let ct = taps(col, gw);
            let o = &mut pos[i * HID..(i + 1) * HID];
            for &(ri, rw) in &rt {
                for &(ci, cw) in &ct {
                    let w = rw * cw;
                    if w == 0.0 {
                        continue;
                    }
                    let e = &self.pos[(ri * GRID + ci) * HID..(ri * GRID + ci + 1) * HID];
                    for d in 0..HID {
                        o[d] += w * e[d];
                    }
                }
            }
            rope_row(i + 1, (row + 1) as f32, (col + 1) as f32, &mut cs);
        }
        let b = &self.b;
        gpu.queue.write_buffer(&b.pin.raw, 0, f32_bytes(&pin));
        gpu.queue.write_buffer(&b.pos.raw, 0, f32_bytes(&pos));
        gpu.queue.write_buffer(&b.cs.raw, 0, f32_bytes(&cs));
        gpu.queue.write_buffer(&b.h2.raw, 0, f32_bytes(&self.cls));

        let mut rec = gpu.recorder();
        let (n32, np32) = (n as u32, np as u32);
        Self::gemm(gpu, &mut rec, &self.patch, b.pin.all(), &b.pe, np32, None);
        // h2[1..] = pe + pos  (row 0 holds the class token)
        gpu.dispatch(&mut rec, K::Ew, &[np32 * HID as u32, 0, HID as u32, 0, 0, HID as u32], &[b.pe.all(), b.pos.all(), b.pos.all(), b.h2.all()], grid((np32 * HID as u32).div_ceil(256)));
        Self::layernorm(gpu, &mut rec, &b.h2, &self.ln_pre_w, &self.ln_pre_b, &b.h, n32);
        let qblocks = n32.div_ceil(8);
        let scale = (1.0 / (HD as f32).sqrt()).to_bits();
        for (li, l) in self.layers.iter().enumerate() {
            Self::layernorm(gpu, &mut rec, &b.h, &l.ln1_w, &l.ln1_b, &b.x, n32);
            Self::gemm(gpu, &mut rec, &l.q, b.x.all(), &b.q, n32, Some((&l.qb, 0)));
            Self::gemm(gpu, &mut rec, &l.k, b.x.all(), &b.k, n32, Some((&l.kb, 0)));
            Self::gemm(gpu, &mut rec, &l.v, b.x.all(), &b.v, n32, Some((&l.vb, 0)));
            gpu.dispatch(&mut rec, K::Rope2d, &[n32, HEADS as u32, HD as u32], &[b.cs.all(), b.q.all(), b.k.all()], grid((n32 * (HEADS * HD / 2) as u32).div_ceil(256)));
            gpu.dispatch(&mut rec, K::VitAttn, &[n32, HEADS as u32, scale, qblocks], &[b.q.all(), b.k.all(), b.v.all(), b.ao.all()], grid(HEADS as u32 * qblocks));
            Self::gemm(gpu, &mut rec, &l.o, b.ao.all(), &b.o, n32, Some((&l.ob, 0)));
            gpu.dispatch(&mut rec, K::Ew, &[n32 * HID as u32, 4, HID as u32, 0, 0, 0], &[b.h.all(), b.o.all(), l.ls1.all(), b.h2.all()], grid((n32 * HID as u32).div_ceil(256)));
            Self::layernorm(gpu, &mut rec, &b.h2, &l.ln2_w, &l.ln2_b, &b.x, n32);
            Self::gemm(gpu, &mut rec, &l.fc, b.x.all(), &b.f, n32, Some((&l.fcb, 1)));
            Self::gemm(gpu, &mut rec, &l.pr, b.f.all(), &b.o, n32, Some((&l.prb, 0)));
            gpu.dispatch(&mut rec, K::Ew, &[n32 * HID as u32, 4, HID as u32, 0, 0, 0], &[b.h2.all(), b.o.all(), l.ls2.all(), b.h.all()], grid((n32 * HID as u32).div_ceil(256)));
            if li % 4 == 3 {
                gpu.flush(&mut rec);
            }
        }
        // merger over patch rows (skip the class token): [tokens][4 * 1024]
        let off = (HID * 4) as u64;
        let pr = Slice { buf: &b.h, offset: off, size: (np * HID * 4) as u64 };
        let nt = ntok as u32;
        Self::gemm(gpu, &mut rec, &self.proj_in, pr, &b.m1, nt, Some((&self.zeros, 1)));
        Self::gemm(gpu, &mut rec, &self.proj_out, b.m1.all(), &b.out, nt, None);
        let bytes = (ntok * TEXT_HID * 4) as u64;
        gpu.copy_at_end(&mut rec, &b.out, 0, &b.stage, 0, bytes);
        gpu.submit(rec);
        let out = gpu.read(&b.stage, bytes)?;
        Ok(bytemuck::cast_slice(&out).to_vec())
    }
}
