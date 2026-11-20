//! Independent CPU reference of the Inkling-2 loop text model, transcribed from the transformers
//! PR #51050 `modeling_inkling2_loop.py` (full-sequence forward, no cache, f32 math, experts
//! dequantized to bf16 like `Inkling2LoopDequantizeExperts`). Used to validate the GPU kernels.
//! It shares no code with the GPU path except the safetensors reader.

use crate::config::TextConfig;
use crate::weights::{SafeTensors, fp8_e4m3_to_f32};
use anyhow::Result;
use rayon::prelude::*;

type M = Vec<f32>; // row-major [rows][cols]

pub struct Reference<'a> {
    st: &'a SafeTensors,
    c: TextConfig,
    pub dump: Vec<Vec<f32>>,
}

fn bf16r(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

impl<'a> Reference<'a> {
    pub fn new(st: &'a SafeTensors, c: &TextConfig) -> Self {
        Self { st, c: c.clone(), dump: Vec::new() }
    }

    fn w(&self, name: &str) -> Vec<f32> {
        self.st.get(name).unwrap_or_else(|e| panic!("{e}")).to_f32()
    }

    /// y[t][n] = sum_k x[t][k] W[n][k]
    fn linear(&self, x: &[f32], t: usize, name: &str) -> (M, usize) {
        let tv = self.st.get(name).unwrap();
        let n = tv.shape[0];
        let k: usize = tv.shape[1..].iter().product();
        let w = tv.to_f32();
        (linear_w(x, t, &w, n, k), n)
    }

    fn rms(&self, x: &[f32], d: usize, w: Option<&[f32]>, eps: f32) -> M {
        let mut out = vec![0.0; x.len()];
        for (r, o) in x.chunks(d).zip(out.chunks_mut(d)) {
            let ms: f64 = r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
            let inv = 1.0 / ((ms as f32) + eps).sqrt();
            for i in 0..d {
                o[i] = r[i] * inv * w.map_or(1.0, |w| w[i]);
            }
        }
        out
    }

    /// Block attention residual: softmax over [blocks..., cur] (each unweighted-RMS normed).
    fn attn_res(&self, blocks: &[&M], cur: &M, queries: &[f32], t: usize) -> Vec<M> {
        let d = self.c.hidden_size;
        let nq = queries.len() / d;
        let mut stack: Vec<M> = blocks.iter().map(|b| self.rms(b, d, None, 1e-6)).collect();
        stack.push(self.rms(cur, d, None, 1e-6));
        let scale = 1.0 / (d as f32).sqrt();
        let mut outs = vec![vec![0.0; t * d]; nq];
        for q in 0..nq {
            let qv = &queries[q * d..(q + 1) * d];
            for tt in 0..t {
                let sc: Vec<f32> = stack.iter().map(|s| dot(qv, &s[tt * d..(tt + 1) * d]) * scale).collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = sc.iter().map(|x| (x - m).exp()).collect();
                let z: f32 = e.iter().sum();
                let o = &mut outs[q][tt * d..(tt + 1) * d];
                for (s, &w) in stack.iter().zip(&e) {
                    let w = w / z;
                    for i in 0..d {
                        o[i] += w * s[tt * d + i];
                    }
                }
            }
        }
        outs
    }

    fn msg_gate(&self, prefix: &str, h: &M, m: &M, t: usize) -> M {
        let d = self.c.hidden_size;
        let hn = self.rms(h, d, None, self.c.rms_norm_eps as f32);
        let mn = self.rms(m, d, None, self.c.rms_norm_eps as f32);
        let mut cat = Vec::with_capacity(t * 2 * d);
        for tt in 0..t {
            cat.extend_from_slice(&hn[tt * d..(tt + 1) * d]);
            cat.extend_from_slice(&mn[tt * d..(tt + 1) * d]);
        }
        let (g, _) = self.linear(&cat, t, &format!("{prefix}down_weight"));
        let (f, _) = self.linear(&g, t, &format!("{prefix}forget_weight"));
        let (gn, _) = self.linear(&g, t, &format!("{prefix}gain_weight"));
        (0..t * d).map(|i| (1.0 - f[i].tanh()) * h[i] + gn[i].tanh() * m[i]).collect()
    }

    /// Short causal depthwise conv (static, conv1d layout [C,1,4]) with residual / sqrt(5).
    fn sconv(&self, x: &M, c: usize, t: usize, wname: &str) -> M {
        let w = self.w(wname);
        let mut y = vec![0.0; t * c];
        for tt in 0..t {
            for ch in 0..c {
                let mut acc = 0.0;
                for j in 0..4 {
                    let src = tt as isize - 3 + j as isize;
                    if src >= 0 {
                        acc += w[ch * 4 + j] * x[src as usize * c + ch];
                    }
                }
                y[tt * c + ch] = (acc + x[tt * c + ch]) / 5f32.sqrt();
            }
        }
        y
    }

    /// Dynamic short conv: taps = static_w[lag][c] + weight_proj(dyn_in)[t][head * 4 + lag].
    fn dyn_conv(&self, x: &M, dyn_in: &M, c: usize, t: usize, prefix: &str) -> M {
        let sw = self.w(&format!("{prefix}static_w"));
        let (dw, _) = self.linear(dyn_in, t, &format!("{prefix}weight_proj"));
        let hd = 128;
        let mut y = vec![0.0; t * c];
        for tt in 0..t {
            for ch in 0..c {
                let h = ch / hd;
                let mut acc = 0.0;
                for lag in 0..4 {
                    if tt >= lag {
                        let wt = sw[lag * c + ch] + dw[tt * 32 + h * 4 + lag];
                        acc += wt * x[(tt - lag) * c + ch];
                    }
                }
                y[tt * c + ch] = (acc + x[tt * c + ch]) / 3.0;
            }
        }
        y
    }

    fn scaled_clip(&self, x: &mut M, scale: &[f32], heads: usize, dh: usize, thr: f32, log_scale: bool) {
        let c = &self.c;
        for (row, chunk) in x.chunks_mut(heads * dh).enumerate() {
            let tau = 1.0 + c.log_scaling_alpha as f32 * (((row + 1) as f32 / c.log_scaling_n_floor as f32).max(1.0)).ln();
            for h in 0..heads {
                let v = &mut chunk[h * dh..(h + 1) * dh];
                for i in 0..dh {
                    v[i] *= scale[h * dh + i];
                }
                let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
                let f = (thr / (n + 1e-6)).min(1.0);
                for a in v.iter_mut() {
                    *a *= f;
                    if log_scale {
                        *a *= tau;
                    }
                }
            }
        }
    }

    fn attention(&self, l: usize, xq: &M, xk: &M, xv: &M, t: usize) -> M {
        let c = &self.c;
        let full = !c.is_local(l);
        let p = format!("model.llm.layers.{l}.attn.");
        let nh = c.num_attention_heads;
        let nkv = c.num_key_value_heads;
        let hd = 128;
        let (mut q, _) = self.linear(xq, t, &format!("{p}wq_du.weight"));
        self.scaled_clip(&mut q, &self.w(&format!("{p}q_l2_clip.scale")), nh, hd, c.qk_l2_clip_threshold as f32, full);
        let (kl, _) = self.linear(xk, t, &format!("{p}wk_dv.weight"));
        let (vl, _) = self.linear(xv, t, &format!("{p}wv_dv.weight"));
        let (k, v) = if full {
            (self.dyn_conv(&kl, xk, nkv * hd, t, &format!("{p}k_sconv.")), self.dyn_conv(&vl, xv, nkv * hd, t, &format!("{p}v_sconv.")))
        } else {
            (self.sconv(&kl, nkv * hd, t, &format!("{p}k_sconv.weight")), self.sconv(&vl, nkv * hd, t, &format!("{p}v_sconv.weight")))
        };
        // key: L2 clip; value: unweighted RMS norm
        let mut k = k;
        for h in k.chunks_mut(hd) {
            let n = h.iter().map(|a| a * a).sum::<f32>().sqrt();
            let f = (c.qk_l2_clip_threshold as f32 / (n + 1e-6)).min(1.0);
            h.iter_mut().for_each(|a| *a *= f);
        }
        let v = self.rms(&v, hd, None, c.rms_norm_eps as f32);
        let (mut rs, _) = self.linear(xq, t, &format!("{p}wr_du.weight"));
        self.scaled_clip(&mut rs, &self.w(&format!("{p}rel_l2_clip.scale")), nh, 16, c.rel_l2_clip_threshold as f32, full);
        let proj = self.w(&format!("{p}rel_logits_proj.proj"));
        let rel_extent = proj.len() / 16;
        let (sinks, _) = self.linear(xq, t, &format!("{p}attn_sink_proj.weight"));
        let window = if full { usize::MAX } else { c.sliding_window_size };
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; t * nh * hd];
        out.par_chunks_mut(nh * hd).enumerate().for_each(|(qi, o)| {
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let qv = &q[qi * nh * hd + h * hd..][..hd];
                let rsv = &rs[qi * nh * 16 + h * 16..][..16];
                let mut logits = Vec::new();
                for kj in 0..=qi {
                    let dist = qi - kj;
                    if dist >= window {
                        logits.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let kv = &k[kj * nkv * hd + kvh * hd..][..hd];
                    let mut s = dot(qv, kv) * scale;
                    if dist < rel_extent {
                        s += (0..16).map(|r| rsv[r] * proj[r * rel_extent + dist]).sum::<f32>();
                    }
                    logits.push(s);
                }
                let sink = sinks[qi * nh + h];
                let m = logits.iter().cloned().fold(sink, f32::max);
                let mut z = (sink - m).exp();
                let e: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
                z += e.iter().sum::<f32>();
                let oh = &mut o[h * hd..(h + 1) * hd];
                for (kj, &w) in e.iter().enumerate() {
                    let vv = &v[kj * nkv * hd + kvh * hd..][..hd];
                    for i in 0..hd {
                        oh[i] += w / z * vv[i];
                    }
                }
            }
        });
        let (gate, _) = self.linear(xq, t, &format!("{p}gate_attn.weight"));
        for i in 0..out.len() {
            out[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        }
        self.linear(&out, t, &format!("{p}wo_ud.weight")).0
    }

    fn moe(&self, l: usize, x: &M, t: usize) -> M {
        let c = &self.c;
        let d = c.hidden_size;
        let inter = c.intermediate_size;
        let p = format!("model.llm.layers.{l}.mlp.");
        let (logits, ne) = self.linear(x, t, &format!("{p}gate.weight"));
        let bias = self.w(&format!("{p}gate.bias"));
        let topk = c.num_experts_per_tok;
        let mut out = vec![0.0f32; t * d];
        let mut routes = Vec::new();
        for tt in 0..t {
            let scores: Vec<f32> = logits[tt * ne..(tt + 1) * ne].iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| (scores[b] + bias[b]).total_cmp(&(scores[a] + bias[a])).then(a.cmp(&b)));
            let sel = &idx[..topk];
            let margin = (scores[idx[topk - 1]] + bias[idx[topk - 1]]) - (scores[idx[topk]] + bias[idx[topk]]);
            if margin < 2e-4 {
                eprintln!("  near-tie routing: layer {l} token {tt} margin {margin:.2e} (experts {} vs {})", idx[topk - 1], idx[topk]);
            }
            let sum: f32 = sel.iter().map(|&e| scores[e]).sum();
            for &e in sel {
                routes.push((tt, e, scores[e] / (sum + 1e-20) * c.route_scale as f32));
            }
        }
        let mut experts: Vec<usize> = routes.iter().map(|r| r.1).collect();
        experts.sort();
        experts.dedup();
        let results: Vec<(usize, Vec<(usize, f32, Vec<f32>)>)> = experts
            .par_iter()
            .map(|&e| {
                let (w13, w2) = self.dequant_expert(l, e);
                let mine: Vec<(usize, f32)> = routes.iter().filter(|r| r.1 == e).map(|r| (r.0, r.2)).collect();
                let mut res = Vec::new();
                for (tt, wt) in mine {
                    let xv = &x[tt * d..(tt + 1) * d];
                    let mut act = vec![0.0f32; inter];
                    for j in 0..inter {
                        let g = dot(xv, &w13[(2 * j) * d..(2 * j + 1) * d]);
                        let u = dot(xv, &w13[(2 * j + 1) * d..(2 * j + 2) * d]);
                        let gc = c.mlp_gate_softcap as f32;
                        let uc = c.mlp_up_softcap as f32;
                        act[j] = (g / gc).tanh() * gc * (1.0 / (1.0 + (-g).exp())) * (u / uc).tanh() * uc;
                    }
                    let y: Vec<f32> = (0..d).map(|r| dot(&act, &w2[r * inter..(r + 1) * inter]) * wt).collect();
                    res.push((tt, wt, y));
                }
                (e, res)
            })
            .collect();
        for (_, res) in results {
            for (tt, _, y) in res {
                for i in 0..d {
                    out[tt * d + i] += y[i];
                }
            }
        }
        out
    }

    /// Expert weights as bf16-rounded f32: w13 [2I][D] (gate/up rows interleaved), w2 [D][I].
    pub fn dequant_expert(&self, l: usize, e: usize) -> (Vec<f32>, Vec<f32>) {
        let p = format!("model.llm.layers.{l}.mlp.experts.");
        let one = |w: &str| -> Vec<f32> {
            let q = self.st.get(&format!("{p}{w}_weight")).unwrap();
            let s = self.st.get(&format!("{p}{w}_weight_scale")).unwrap();
            let ne = q.shape[0];
            let rows = q.shape[1];
            let qb = &q.data[e * q.data.len() / ne..(e + 1) * q.data.len() / ne];
            let sb = &s.data[e * s.data.len() / ne..(e + 1) * s.data.len() / ne];
            let nvfp4 = matches!(q.dtype, crate::weights::DType::U8);
            if nvfp4 {
                let amax = self.st.get(&format!("{p}{w}_weight_amax")).unwrap().to_f32()[e];
                let den = self.st.get(&format!("{p}{w}_weight_outer_scale_denominator")).unwrap().to_f32()[0];
                let fac = amax / den;
                let cols = q.shape[2] * 2;
                let mut out = vec![0.0; rows * cols];
                for r in 0..rows {
                    for k in 0..cols {
                        let b = qb[r * cols / 2 + k / 2];
                        let nib = if k % 2 == 0 { b & 0xf } else { b >> 4 };
                        let sc = fp8_e4m3_to_f32(sb[r * cols / 16 + k / 16]);
                        out[r * cols + k] = bf16r(e2m1(nib) * sc * fac);
                    }
                }
                out
            } else {
                let cols = q.shape[2];
                let mut out = vec![0.0; rows * cols];
                for r in 0..rows {
                    for k in 0..cols {
                        let v = fp8_e4m3_to_f32(qb[r * cols + k]);
                        let sc = 2f32.powi(sb[r * cols / 32 + k / 32] as i32 - 127);
                        out[r * cols + k] = bf16r(v * sc);
                    }
                }
                out
            }
        };
        (one("w13"), one("w2"))
    }

    fn layer(&mut self, l: usize, h: &M, blocks: &mut Vec<M>, messages: Option<&[M]>, t: usize) -> M {
        let c = self.c.clone();
        let d = c.hidden_size;
        let p = format!("model.llm.layers.{l}.");
        let full = !c.is_local(l);
        let boundary = l % c.n_layers_per_attn_res_block == 0;
        let block_residual = h.clone();
        let brefs: Vec<&M> = blocks.iter().collect();
        let (mut sq, mut sk, mut sv) = if l == 0 {
            (h.clone(), h.clone(), h.clone())
        } else {
            let q = self.w(&format!("model.llm.attn_res.attn_queries.{}", l - 1));
            let o = self.attn_res(&brefs, h, &q, t);
            if o.len() == 3 { (o[0].clone(), o[1].clone(), o[2].clone()) } else { (o[0].clone(), o[0].clone(), o[0].clone()) }
        };
        if let Some(msg) = messages {
            let cp = format!("{p}controllers.");
            sq = self.msg_gate(&format!("{cp}attn."), &sq, &msg[0], t);
            sk = self.msg_gate(&format!("{cp}k."), &sk, &msg[1], t);
            sv = self.msg_gate(&format!("{cp}v."), &sv, &msg[2], t);
        }
        let eps = c.rms_norm_eps as f32;
        let (nq, nk, nv) = if full {
            (self.w(&format!("{p}q_input_norm.weight")), self.w(&format!("{p}k_input_norm.weight")), self.w(&format!("{p}v_input_norm.weight")))
        } else {
            let w = self.w(&format!("{p}attn_norm.weight"));
            (w.clone(), w.clone(), w)
        };
        let xq = self.rms(&sq, d, Some(&nq), eps);
        let xk = self.rms(&sk, d, Some(&nk), eps);
        let xv = self.rms(&sv, d, Some(&nv), eps);
        let ao = self.attention(l, &xq, &xk, &xv, t);
        let hs = self.sconv(&ao, d, t, &format!("{p}attn_sconv.weight"));
        let residual = if boundary {
            blocks.push(block_residual.clone());
            hs
        } else {
            (0..t * d).map(|i| block_residual[i] + hs[i]).collect()
        };
        let brefs: Vec<&M> = blocks.iter().collect();
        let mut m = self.attn_res(&brefs, &residual, &self.w(&format!("model.llm.attn_res.mlp_queries.{l}")), t).remove(0);
        if let Some(msg) = messages {
            m = self.msg_gate(&format!("{p}controllers.mlp."), &m, &msg[3], t);
        }
        let xm = self.rms(&m, d, Some(&self.w(&format!("{p}mlp_norm.weight"))), eps);
        let mo = self.moe(l, &xm, t);
        let mc = self.sconv(&mo, d, t, &format!("{p}mlp_sconv.weight"));
        let out: M = (0..t * d).map(|i| residual[i] + mc[i]).collect();
        // the outer loop appends the layer input at block boundaries; `blocks` was already
        // extended inside, matching `block_states = cat([block_states, hidden_states])`.
        out
    }

    /// Logits of the last token for `tokens`.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let c = self.c.clone();
        let d = c.hidden_size;
        let t = tokens.len();
        let emb = self.st.get("model.llm.embed.weight")?;
        let mut x = Vec::with_capacity(t * d);
        for &id in tokens {
            let r = &emb.data[id as usize * d * 2..(id as usize + 1) * d * 2];
            x.extend(r.chunks_exact(2).map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()));
        }
        let mut h = self.rms(&x, d, Some(&self.w("model.llm.embed_norm.weight")), c.rms_norm_eps as f32);
        let lc = c.loop_config.clone();
        let mut blocks: Vec<M> = Vec::new();
        let t0 = std::time::Instant::now();
        for l in 0..lc.core_start {
            h = self.layer(l, &h, &mut blocks, None, t);
            self.dump.push(h.clone());
            eprintln!("  ref layer {l} ({:.1}s)", t0.elapsed().as_secs_f64());
        }
        let pre_h = h.clone();
        let pre_blocks = blocks.clone();
        let n_pre = pre_blocks.len();
        for l in lc.core_start..lc.core_end {
            h = self.layer(l, &h, &mut blocks, None, t);
            self.dump.push(h.clone());
            eprintln!("  ref layer {l} pass 0 ({:.1}s)", t0.elapsed().as_secs_f64());
        }
        let qm = self.w("model.llm.middle_cycle_cross_pass_queries");
        let kvq = self.w("model.llm.middle_cycle_cross_pass_kv_queries");
        for pass in 1..lc.num_passes {
            let prev_blocks: Vec<M> = blocks[n_pre + 1..].to_vec();
            let prev_h = h.clone();
            h = pre_h.clone();
            blocks = pre_blocks.clone();
            for (li, l) in (lc.core_start..lc.core_end).enumerate() {
                let mut q = Vec::new();
                q.extend_from_slice(&qm[2 * li * d..(2 * li + 1) * d]);
                q.extend_from_slice(&kvq[2 * li * d..(2 * li + 1) * d]);
                q.extend_from_slice(&kvq[(2 * li + 1) * d..(2 * li + 2) * d]);
                q.extend_from_slice(&qm[(2 * li + 1) * d..(2 * li + 2) * d]);
                let pb: Vec<&M> = prev_blocks.iter().collect();
                let msgs = self.attn_res(&pb, &prev_h, &q, t);
                h = self.layer(l, &h, &mut blocks, Some(&msgs), t);
                self.dump.push(h.clone());
                eprintln!("  ref layer {l} pass {pass} ({:.1}s)", t0.elapsed().as_secs_f64());
            }
        }
        for l in lc.core_end..c.num_hidden_layers {
            h = self.layer(l, &h, &mut blocks, None, t);
            self.dump.push(h.clone());
            eprintln!("  ref layer {l} ({:.1}s)", t0.elapsed().as_secs_f64());
        }
        let brefs: Vec<&M> = blocks.iter().collect();
        let fin = self.attn_res(&brefs, &h, &self.w("model.llm.attn_res.output_query"), t).remove(0);
        let hn = self.rms(&fin[(t - 1) * d..], d, Some(&self.w("model.llm.norm.weight")), c.rms_norm_eps as f32);
        let un = self.st.get("model.llm.unembed.weight")?;
        let nv = c.unpadded_vocab_size;
        let w = &un.data[..nv * d * 2];
        let sc = c.final_logit_softcapping as f32;
        let logits: Vec<f32> = (0..nv)
            .into_par_iter()
            .map(|r| {
                let row = &w[r * d * 2..(r + 1) * d * 2];
                let s: f32 = row.chunks_exact(2).zip(&hn).map(|(b, &x)| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() * x).sum();
                (s / sc).tanh() * sc
            })
            .collect();
        Ok(logits)
    }
}

fn e2m1(n: u8) -> f32 {
    const V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = V[(n & 7) as usize];
    if n & 8 != 0 { -v } else { v }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn linear_w(x: &[f32], t: usize, w: &[f32], n: usize, k: usize) -> M {
    let mut y = vec![0.0f32; t * n];
    y.par_chunks_mut(n).enumerate().for_each(|(tt, row)| {
        let xv = &x[tt * k..(tt + 1) * k];
        for (j, o) in row.iter_mut().enumerate() {
            *o = dot(xv, &w[j * k..(j + 1) * k]);
        }
    });
    y
}
