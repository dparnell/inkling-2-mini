//! Inkling-2 looped text model on the GPU.
//!
//! Layer structure (from the reference `Inkling2LoopTextModel`):
//! * 24 weight layers; layers [core_start, core_end) run `num_passes` times. Later passes restart
//!   from the pre-loop state and receive "messages" from the previous pass through gated mixing.
//! * Every layer: block attention-residual input, attention (sliding or full), post-attention
//!   short conv, residual (or block start), block attention-residual MLP input, top-8 sigmoid MoE,
//!   post-MLP short conv, residual.
//! * Each layer *application* (48 of them) has its own KV cache and conv states.

use crate::config::TextConfig;
use crate::experts::ExpertStore;
use crate::gpu::{Buf, Gpu, K, Recorder, grid};
use crate::weights::SafeTensors;
use anyhow::{Context, Result, bail};
use std::sync::Arc;

pub fn f32_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

/// A bf16 weight matrix [n, k] split by rows into chunks that fit one storage binding.
pub struct Matrix {
    pub chunks: Vec<(Buf, usize, usize)>,
    pub n: usize,
    pub k: usize,
}

impl Matrix {
    pub fn upload(gpu: &Gpu, label: &str, bytes: &[u8], n: usize, k: usize) -> Self {
        assert_eq!(bytes.len(), n * k * 2);
        let row_bytes = (k * 2) as u64;
        let max_rows = ((gpu.max_binding().min(1 << 31) / row_bytes) as usize).max(1);
        let max_rows = if max_rows > 64 { max_rows / 64 * 64 } else { max_rows };
        let mut chunks = Vec::new();
        let mut r = 0;
        while r < n {
            let rows = max_rows.min(n - r);
            chunks.push((gpu.storage_init(label, &bytes[r * k * 2..(r + rows) * k * 2]), r, rows));
            r += rows;
        }
        Matrix { chunks, n, k }
    }

}

struct Loader<'a> {
    st: &'a SafeTensors,
    gpu: &'a Gpu,
}

impl Loader<'_> {
    fn mat(&self, name: &str) -> Result<Matrix> {
        let t = self.st.get(name)?;
        let k: usize = t.shape[1..].iter().product();
        Ok(Matrix::upload(self.gpu, name, &t.to_bf16_bytes(), t.shape[0], k))
    }
    /// Row-concatenation of several [*, k] tensors.
    fn mat_cat(&self, names: &[&str]) -> Result<Matrix> {
        let mut bytes = Vec::new();
        let (mut n, mut k) = (0, 0);
        for name in names {
            let t = self.st.get(name)?;
            k = t.shape[1..].iter().product();
            n += t.shape[0];
            bytes.extend_from_slice(&t.to_bf16_bytes());
        }
        Ok(Matrix::upload(self.gpu, names[0], &bytes, n, k))
    }
    fn vec(&self, name: &str) -> Result<Buf> {
        Ok(self.gpu.storage_init(name, f32_bytes(&self.st.get(name)?.to_f32())))
    }
    fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.st.get(name)?.to_f32())
    }
}

struct Ctrl {
    down: Matrix, // [64, 2D]
    fg: Matrix,   // [2D, 64]: forget rows then gain rows
}

struct LayerW {
    full: bool,
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wr: Matrix,
    wgate: Matrix,
    wo: Matrix,
    wsink: Matrix,
    router: Matrix,
    router_bias: Buf,
    k_dyn: Option<Matrix>,
    v_dyn: Option<Matrix>,
    k_conv: Buf,
    v_conv: Buf,
    q_scale: Buf,
    rel_scale: Buf,
    rel_proj: Buf,
    rel_extent: u32,
    window: u32,
    norm_q: Buf,
    norm_k: Buf,
    norm_v: Buf,
    mlp_norm: Buf,
    attn_conv: Buf,
    mlp_conv: Buf,
    attn_q: Option<Buf>,
    n_attn_q: u32,
    mlp_q: Buf,
    ctrl: Option<[Ctrl; 4]>,
}

/// Per layer-application state: KV cache and the four conv histories.
struct AppState {
    k: Buf,
    v: Buf,
    cap: u32,
    hk: Buf,
    hv: Buf,
    ha: Buf,
    hm: Buf,
}

/// Activation buffers for up to `tm` tokens.
struct Acts {
    cur: Buf,
    next: Buf,
    blocks: Buf, // [4][tm][D]
    prev: Buf,   // [2][tm][D]: previous pass block (layer core_end - block input) and output
    ares: [Buf; 3],
    msg: [Buf; 4],
    gated: [Buf; 4],
    xq: Buf,
    xk: Buf,
    xv: Buf,
    cat: Buf,
    g64: Buf,
    fg: Buf,
    q: Buf,
    klin: Buf,
    vlin: Buf,
    kc: Buf,
    vc: Buf,
    dk: Buf,
    dv: Buf,
    rs: Buf,
    sinks: Buf,
    gate: Buf,
    ao: Buf,
    part: Buf,
    o: Buf,
    c: Buf,
    resid: Buf,
    m: Buf,
    xm: Buf,
    rlog: Buf,
    ids: Buf,
    wts: Buf,
    sorted: Buf,
    inv: Buf,
    tiles: Vec<Buf>,
    y13: Buf,
    act: Buf,
    y2: Buf,
    mo: Buf,
    mc: Buf,
    fin: Buf,
    finn: Buf,
    logits: Buf,
    route_stage: Buf,
    logit_stage: Buf,
    dbg_stage: Buf,
}

pub struct Model {
    pub cfg: TextConfig,
    st: Arc<SafeTensors>,
    layers: Vec<LayerW>,
    embed_norm: Buf,
    final_norm: Buf,
    final_q: Buf,
    msg_q: Buf, // [loop layers][4][D] (q, k, v, mlp)
    unembed: Matrix,
    ones: Buf,
    zeros: Buf,
    pub experts: ExpertStore,
    apps: Vec<AppState>,
    acts: Acts,
    pub tm: usize,
    pub ctx: usize,
    pub n_vocab: usize,
    /// Debug: capture each layer application's output for the first `dump_rows` rows.
    pub dump: Option<Vec<Vec<f32>>>,
    pub timings: Timings,
    /// Conv histories copied at `snap_pos` (KV caches are position-addressed and need no copy).
    snap: Buf,
    pub snap_pos: Option<usize>,
    rewind_window: usize,
}

#[derive(Default, Clone, Debug)]
pub struct Timings {
    pub route_syncs: u64,
    pub route_wait_s: f64,
    pub ensure_s: f64,
}

pub struct ModelOptions {
    pub ctx: usize,
    pub chunk: usize,
    pub expert_budget: u64,
    pub rewind_slack: usize,
}

const D: usize = 2304;
const MAX_SPLITS: usize = 64;

impl Model {
    pub fn app_count(cfg: &TextConfig) -> usize {
        let lc = &cfg.loop_config;
        cfg.num_hidden_layers + (lc.num_passes - 1) * (lc.core_end - lc.core_start)
    }

    fn app_index(&self, layer: usize, pass: usize) -> usize {
        let lc = &self.cfg.loop_config;
        if pass == 0 { layer } else { self.cfg.num_hidden_layers + (pass - 1) * (lc.core_end - lc.core_start) + layer - lc.core_start }
    }

    pub fn load(gpu: &Gpu, st: Arc<SafeTensors>, cfg: &TextConfig, opts: &ModelOptions) -> Result<Self> {
        if cfg.hidden_size != D || cfg.head_dim != 128 || cfg.swa_head_dim() != 128 || cfg.d_rel != 16 || cfg.sconv_kernel_size != 4 {
            bail!("unsupported model geometry (expects hidden 2304, head_dim 128, d_rel 16, sconv 4)");
        }
        let ld = Loader { st: &st, gpu };
        let lc = &cfg.loop_config;
        let mut layers = Vec::new();
        let t0 = std::time::Instant::now();
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.llm.layers.{l}.");
            let full = !cfg.is_local(l);
            let a = format!("{p}attn.");
            let (k_dyn, v_dyn, k_conv, v_conv) = if full {
                (
                    Some(ld.mat(&format!("{a}k_sconv.weight_proj"))?),
                    Some(ld.mat(&format!("{a}v_sconv.weight_proj"))?),
                    ld.vec(&format!("{a}k_sconv.static_w"))?,
                    ld.vec(&format!("{a}v_sconv.static_w"))?,
                )
            } else {
                (None, None, ld.vec(&format!("{a}k_sconv.weight"))?, ld.vec(&format!("{a}v_sconv.weight"))?)
            };
            let (norm_q, norm_k, norm_v) = if full {
                (ld.vec(&format!("{p}q_input_norm.weight"))?, ld.vec(&format!("{p}k_input_norm.weight"))?, ld.vec(&format!("{p}v_input_norm.weight"))?)
            } else {
                let w = ld.f32s(&format!("{p}attn_norm.weight"))?;
                (gpu.storage_init("attn_norm", f32_bytes(&w)), gpu.storage_init("attn_norm", f32_bytes(&w)), gpu.storage_init("attn_norm", f32_bytes(&w)))
            };
            let (attn_q, n_attn_q) = if l > 0 {
                let name = format!("model.llm.attn_res.attn_queries.{}", l - 1);
                let v = ld.f32s(&name)?;
                let n = (v.len() / D) as u32;
                (Some(gpu.storage_init(&name, f32_bytes(&v))), n)
            } else {
                (None, 0)
            };
            let in_loop = l >= lc.core_start && l < lc.core_end;
            let ctrl = if in_loop {
                let c = |n: &str| -> Result<Ctrl> {
                    let b = format!("{p}controllers.{n}.");
                    Ok(Ctrl { down: ld.mat(&format!("{b}down_weight"))?, fg: ld.mat_cat(&[&format!("{b}forget_weight"), &format!("{b}gain_weight")])? })
                };
                Some([c("attn")?, c("k")?, c("v")?, c("mlp")?])
            } else {
                None
            };
            let rel_extent = if full { cfg.rel_extent } else { cfg.sliding_window_size } as u32;
            layers.push(LayerW {
                full,
                wq: ld.mat(&format!("{a}wq_du.weight"))?,
                wk: ld.mat(&format!("{a}wk_dv.weight"))?,
                wv: ld.mat(&format!("{a}wv_dv.weight"))?,
                wr: ld.mat(&format!("{a}wr_du.weight"))?,
                wgate: ld.mat(&format!("{a}gate_attn.weight"))?,
                wo: ld.mat(&format!("{a}wo_ud.weight"))?,
                wsink: ld.mat(&format!("{a}attn_sink_proj.weight"))?,
                router: ld.mat(&format!("{p}mlp.gate.weight"))?,
                router_bias: ld.vec(&format!("{p}mlp.gate.bias"))?,
                k_dyn,
                v_dyn,
                k_conv,
                v_conv,
                q_scale: ld.vec(&format!("{a}q_l2_clip.scale"))?,
                rel_scale: ld.vec(&format!("{a}rel_l2_clip.scale"))?,
                rel_proj: ld.vec(&format!("{a}rel_logits_proj.proj"))?,
                rel_extent,
                window: if full { 0 } else { cfg.sliding_window_size as u32 },
                norm_q,
                norm_k,
                norm_v,
                mlp_norm: ld.vec(&format!("{p}mlp_norm.weight"))?,
                attn_conv: ld.vec(&format!("{p}attn_sconv.weight"))?,
                mlp_conv: ld.vec(&format!("{p}mlp_sconv.weight"))?,
                attn_q,
                n_attn_q,
                mlp_q: ld.vec(&format!("model.llm.attn_res.mlp_queries.{l}"))?,
                ctrl,
            });
        }
        // message queries: checkpoint stores [q, mlp] pairs and [k, v] pairs per loop layer
        let n_loop = lc.core_end - lc.core_start;
        let qm = ld.f32s("model.llm.middle_cycle_cross_pass_queries")?;
        let kv = ld.f32s("model.llm.middle_cycle_cross_pass_kv_queries")?;
        let mut mq = Vec::with_capacity(n_loop * 4 * D);
        for i in 0..n_loop {
            mq.extend_from_slice(&qm[(2 * i) * D..(2 * i + 1) * D]);
            mq.extend_from_slice(&kv[(2 * i) * D..(2 * i + 1) * D]);
            mq.extend_from_slice(&kv[(2 * i + 1) * D..(2 * i + 2) * D]);
            mq.extend_from_slice(&qm[(2 * i + 1) * D..(2 * i + 2) * D]);
        }
        let msg_q = gpu.storage_init("msg_q", f32_bytes(&mq));
        let n_vocab = cfg.unpadded_vocab_size;
        let un = st.get("model.llm.unembed.weight")?;
        let unembed = Matrix::upload(gpu, "unembed", &un.to_bf16_bytes()[..n_vocab * D * 2], n_vocab, D);
        tracing::info!("dense weights uploaded in {:.1} s", t0.elapsed().as_secs_f64());

        let experts = ExpertStore::new(gpu, st.clone(), cfg.num_hidden_layers, cfg.n_routed_experts, D, cfg.intermediate_size, opts.expert_budget)?;

        // per-application state
        let n_apps = Self::app_count(cfg);
        let mut apps = Vec::new();
        for app in 0..n_apps {
            let layer = if app < cfg.num_hidden_layers { app } else { lc.core_start + (app - cfg.num_hidden_layers) % n_loop };
            let full = !cfg.is_local(layer);
            let cap = if full { opts.ctx } else { (cfg.sliding_window_size + opts.chunk + opts.rewind_slack).min(opts.ctx) } as u32;
            let kv_bytes = cfg.num_key_value_heads as u64 * cap as u64 * 128 * 2;
            let kvh = cfg.num_key_value_heads * 128;
            apps.push(AppState {
                k: gpu.storage("kcache", kv_bytes),
                v: gpu.storage("vcache", kv_bytes),
                cap,
                hk: gpu.storage("hist", (4 * kvh * 4) as u64),
                hv: gpu.storage("hist", (4 * kvh * 4) as u64),
                ha: gpu.storage("hist", (4 * D * 4) as u64),
                hm: gpu.storage("hist", (4 * D * 4) as u64),
            });
        }
        let tm = opts.chunk;
        let f = |n: usize| gpu.storage("act", (n * 4) as u64);
        let topk = cfg.num_experts_per_tok;
        let a = tm * topk;
        let inter = cfg.intermediate_size;
        let max_tiles = a / 64 + cfg.n_routed_experts + 1;
        let n_pools = experts.caches.iter().map(|c| c.pools.len()).max().unwrap_or(1);
        let acts = Acts {
            cur: f(tm * D),
            next: f(tm * D),
            blocks: f(4 * tm * D),
            prev: f(2 * tm * D),
            ares: [f(tm * D), f(tm * D), f(tm * D)],
            msg: [f(tm * D), f(tm * D), f(tm * D), f(tm * D)],
            gated: [f(tm * D), f(tm * D), f(tm * D), f(tm * D)],
            xq: f(tm * D),
            xk: f(tm * D),
            xv: f(tm * D),
            cat: f(tm * 2 * D),
            g64: f(tm * 64),
            fg: f(tm * 2 * D),
            q: f(tm * 3072),
            klin: f(tm * 1024),
            vlin: f(tm * 1024),
            kc: f(tm * 1024),
            vc: f(tm * 1024),
            dk: f(tm * 32),
            dv: f(tm * 32),
            rs: f(tm * 384),
            sinks: f(tm * 32),
            gate: f(tm * 3072),
            ao: f(tm * 3072),
            part: f(24 * 8 * MAX_SPLITS * 130),
            o: f(tm * D),
            c: f(tm * D),
            resid: f(tm * D),
            m: f(tm * D),
            xm: f(tm * D),
            rlog: f(tm * 128),
            ids: f(a),
            wts: f(a),
            sorted: f(a),
            inv: f(a),
            tiles: (0..n_pools).map(|_| f(max_tiles * 4)).collect(),
            y13: f(a * 2 * inter),
            act: f(a * inter),
            y2: f(a * D),
            mo: f(tm * D),
            mc: f(tm * D),
            fin: f(D),
            finn: f(D),
            logits: f(n_vocab),
            route_stage: gpu.readback_buffer((a * 8) as u64),
            logit_stage: gpu.readback_buffer((n_vocab * 4) as u64),
            dbg_stage: gpu.readback_buffer((tm * D * 4) as u64),
        };
        let snap_bytes: u64 = apps.iter().map(|a| a.hk.size + a.hv.size + a.ha.size + a.hm.size).sum();
        let snap = gpu.storage("snapshot", snap_bytes);
        let ones = gpu.storage_init("ones", f32_bytes(&vec![1.0f32; 2 * D]));
        let zeros = gpu.storage("zeros", 64 * 4);
        tracing::info!("model GPU buffers: {:.2} GB", crate::metrics::GPU_BYTES.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9);
        Ok(Self {
            cfg: cfg.clone(),
            layers,
            embed_norm: ld.vec("model.llm.embed_norm.weight")?,
            final_norm: ld.vec("model.llm.norm.weight")?,
            final_q: ld.vec("model.llm.attn_res.output_query")?,
            msg_q,
            unembed,
            ones,
            zeros,
            experts,
            apps,
            acts,
            tm,
            ctx: opts.ctx,
            n_vocab,
            st,
            dump: None,
            timings: Timings::default(),
            snap,
            snap_pos: None,
            rewind_window: opts.chunk + opts.rewind_slack,
        })
    }

    fn hist_bufs(&self) -> impl Iterator<Item = &Buf> {
        self.apps.iter().flat_map(|a| [&a.hk, &a.hv, &a.ha, &a.hm])
    }

    /// Remember the conv state at position `pos` (call when the model has consumed `pos` rows).
    pub fn snapshot(&mut self, gpu: &Gpu, pos: usize) {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut off = 0;
        for b in self.hist_bufs() {
            enc.copy_buffer_to_buffer(&b.raw, 0, &self.snap.raw, off, b.size);
            off += b.size;
        }
        gpu.queue.submit([enc.finish()]);
        self.snap_pos = Some(pos);
    }

    /// Rewind to the snapshot if the sliding-window rings still hold its keys; `cur` is the
    /// number of rows consumed so far.
    pub fn restore(&mut self, gpu: &Gpu, cur: usize) -> Option<usize> {
        let sp = self.snap_pos?;
        if sp > cur || cur - sp > self.rewind_window {
            return None;
        }
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut off = 0;
        for b in self.hist_bufs() {
            enc.copy_buffer_to_buffer(&self.snap.raw, off, &b.raw, 0, b.size);
            off += b.size;
        }
        gpu.queue.submit([enc.finish()]);
        Some(sp)
    }

    /// Forget all sequence state (conv histories are zeroed; KV caches are masked by position).
    pub fn reset(&mut self, gpu: &Gpu) {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        for b in self.hist_bufs() {
            enc.clear_buffer(&b.raw, 0, None);
        }
        gpu.queue.submit([enc.finish()]);
        self.snap_pos = None;
    }

    /// Host-side embedding gather (bf16 -> f32) for `tokens`.
    pub fn embed_rows(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let t = self.st.get("model.llm.embed.weight")?;
        let mut out = Vec::with_capacity(tokens.len() * D);
        for &id in tokens {
            let r = &t.data[id as usize * D * 2..(id as usize + 1) * D * 2];
            out.extend(r.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()));
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- dispatch helpers

    fn matmul(&self, gpu: &mut Gpu, rec: &mut Recorder, w: &Matrix, x: &Buf, y: &Buf, t: u32) {
        let k = w.k as u32;
        let n = w.n as u32;
        for (buf, r0, rows) in &w.chunks {
            let rows = *rows as u32;
            if t == 1 {
                gpu.dispatch(rec, K::Gemv, &[k, rows, 0, *r0 as u32], &[x.all(), buf.all(), y.all()], grid(rows.div_ceil(4)));
            } else {
                let tiles_n = rows.div_ceil(64);
                let tiles_m = t.div_ceil(64);
                gpu.dispatch(rec, K::Gemm, &[t, k, rows, k, n, *r0 as u32, tiles_n], &[x.all(), buf.all(), y.all()], grid(tiles_n * tiles_m));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rmsnorm(&self, gpu: &mut Gpu, rec: &mut Recorder, x: &Buf, w: &Buf, y: &Buf, t: u32, y_stride: u32, y_off: u32) {
        let eps = self.cfg.rms_norm_eps as f32;
        let ws = 0u32;
        gpu.dispatch(rec, K::RmsNorm, &[t, D as u32, D as u32, y_stride, 0, y_off, eps.to_bits(), ws], &[x.all(), w.all(), y.all()], grid(t));
    }

    #[allow(clippy::too_many_arguments)]
    fn ew(&self, gpu: &mut Gpu, rec: &mut Recorder, mode: u32, n: u32, a: &Buf, b: &Buf, c: &Buf, y: &Buf) {
        gpu.dispatch(rec, K::Ew, &[n, mode, D as u32, 0, 0, 0], &[a.all(), b.all(), c.all(), y.all()], grid(n.div_ceil(256)));
    }


    /// Attention residual over block slots (`nb` slots 0..nb or an explicit list) and `cur`.
    #[allow(clippy::too_many_arguments)]
    fn attn_res(&self, gpu: &mut Gpu, rec: &mut Recorder, blocks: &Buf, slots: &[u32], cur: &Buf, cur_off: u32, queries: &Buf, q_off: u32, nq: u32, out: &[&Buf], t: u32) {
        let packed = slots.iter().enumerate().fold(0u32, |acc, (i, &s)| acc | (s << (4 * i)));
        let slot_stride = (self.tm * D) as u32;
        // outputs are separate buffers; write them via one dispatch per query
        for qi in 0..nq as usize {
            gpu.dispatch(
                rec,
                K::AttnRes,
                &[t, D as u32, slots.len() as u32, 1, packed, q_off + (qi * D) as u32, slot_stride, 0, 1e-6f32.to_bits(), cur_off, 0],
                &[blocks.all(), cur.all(), queries.all(), out[qi].all()],
                grid(t),
            );
        }
    }

    /// Message gate: y = (1 - tanh(f)) * h + tanh(g) * m, [f | g] = fg(down([rms(h), rms(m)]))
    fn msg_gate(&self, gpu: &mut Gpu, rec: &mut Recorder, c: &Ctrl, h: &Buf, m: &Buf, y: &Buf, t: u32) {
        let a = &self.acts;
        self.rmsnorm(gpu, rec, h, &self.ones, &a.cat, t, 2 * D as u32, 0);
        self.rmsnorm(gpu, rec, m, &self.ones, &a.cat, t, 2 * D as u32, D as u32);
        self.matmul(gpu, rec, &c.down, &a.cat, &a.g64, t);
        self.matmul(gpu, rec, &c.fg, &a.g64, &a.fg, t);
        self.ew(gpu, rec, 2, t * D as u32, h, m, &a.fg, y);
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(&self, gpu: &mut Gpu, rec: &mut Recorder, x: &Buf, hist: &Buf, w: &Buf, dw: Option<&Buf>, y: &Buf, c: u32, t: u32, pos0: u32) {
        let dynamic = dw.is_some() as u32;
        let dwb = dw.unwrap_or(&self.zeros);
        gpu.dispatch(rec, K::Conv, &[t, c, pos0, dynamic, 128, 32, c, 0], &[x.all(), hist.all(), w.all(), dwb.all(), y.all()], grid((t * c).div_ceil(256)));
        gpu.dispatch(rec, K::ConvHist, &[t, c, pos0, c, 0], &[x.all(), hist.all()], grid((t.min(4) * c).div_ceil(256)));
    }

    #[allow(clippy::too_many_arguments)]
    fn scaled_clip(&self, gpu: &mut Gpu, rec: &mut Recorder, x: &Buf, scale: &Buf, t: u32, h: u32, dh: u32, thr: f64, log_scale: bool, pos0: u32) {
        let c = &self.cfg;
        gpu.dispatch(
            rec,
            K::ScaledClip,
            &[t, h, dh, h * dh, 0, 1, (thr as f32).to_bits(), log_scale as u32, (c.log_scaling_alpha as f32).to_bits(), (c.log_scaling_n_floor as f32).to_bits(), pos0],
            &[scale.all(), x.all()],
            grid(t * h),
        );
    }

    // ---------------------------------------------------------------- forward

    /// Run `t` new rows (embeddings already normed/injected, in `acts.cur`) at positions pos0..
    /// Returns the logits of the last row if `logits`.
    fn run_stack(&mut self, gpu: &mut Gpu, t: u32, pos0: u32, logits: bool) -> Result<Option<Vec<f32>>> {
        let lc = self.cfg.loop_config.clone();
        let n_layers = self.cfg.num_hidden_layers;
        let mut rec = gpu.recorder();
        for l in 0..lc.core_start {
            rec = self.layer(gpu, rec, l, 0, t, pos0)?;
        }
        for pass in 0..lc.num_passes {
            if pass > 0 {
                let n = t * D as u32;
                let so = (self.tm * D) as u32;
                // previous pass block (slot 2) -> slot 4, previous output -> slot 5, restart from slot 1
                let bytes = (n * 4) as u64;
                let sb = (so * 4) as u64;
                gpu.copy_inline(&mut rec, &self.acts.blocks, 2 * sb, &self.acts.prev, 0, bytes);
                gpu.copy_inline(&mut rec, &self.acts.cur, 0, &self.acts.prev, sb, bytes);
                gpu.copy_inline(&mut rec, &self.acts.blocks, sb, &self.acts.cur, 0, bytes);
            }
            for l in lc.core_start..lc.core_end {
                rec = self.layer(gpu, rec, l, pass, t, pos0)?;
            }
        }
        for l in lc.core_end..n_layers {
            rec = self.layer(gpu, rec, l, 0, t, pos0)?;
        }
        if !logits {
            gpu.submit(rec);
            return Ok(None);
        }
        // final: attention residual over the 4 blocks + cur (last row only), norm, unembed
        let a = &self.acts;
        let packed = (0u32..4).fold(0u32, |acc, s| acc | (s << (4 * s)));
        gpu.dispatch(
            &mut rec,
            K::AttnRes,
            &[1, D as u32, 4, 1, packed, 0, (self.tm * D) as u32, 0, 1e-6f32.to_bits(), 0, t - 1],
            &[a.blocks.all(), a.cur.all(), self.final_q.all(), a.fin.all()],
            grid(1),
        );
        let eps = self.cfg.rms_norm_eps as f32;
        gpu.dispatch(&mut rec, K::RmsNorm, &[1, D as u32, D as u32, D as u32, 0, 0, eps.to_bits(), 0], &[a.fin.all(), self.final_norm.all(), a.finn.all()], grid(1));
        self.matmul(gpu, &mut rec, &self.unembed, &self.acts.finn, &self.acts.logits, 1);
        gpu.copy_at_end(&mut rec, &self.acts.logits, 0, &self.acts.logit_stage, 0, (self.n_vocab * 4) as u64);
        gpu.submit(rec);
        let bytes = gpu.read(&self.acts.logit_stage, (self.n_vocab * 4) as u64)?;
        let sc = self.cfg.final_logit_softcapping as f32;
        Ok(Some(bytemuck::cast_slice::<u8, f32>(&bytes).iter().map(|&x| (x / sc).tanh() * sc).collect()))
    }

    fn layer(&mut self, gpu: &mut Gpu, mut rec: Recorder, l: usize, pass: usize, t: u32, pos0: u32) -> Result<Recorder> {
        let app = self.app_index(l, pass);
        let lc = &self.cfg.loop_config;
        let block = self.cfg.n_layers_per_attn_res_block;
        let n = t * D as u32;
        let so = (self.tm * D) as u32;
        let lw = &self.layers[l];
        let a = &self.acts;
        let st = &self.apps[app];
        let nb_before = l.div_ceil(block);
        let boundary = l % block == 0;
        let slots_before: Vec<u32> = (0..nb_before as u32).collect();

        // ---- attention inputs
        let have_msg = pass > 0;
        if have_msg {
            let li = (l - lc.core_start) as u32;
            let qoff = li * 4 * D as u32;
            self.attn_res(gpu, &mut rec, &a.prev, &[0], &a.prev, so, &self.msg_q, qoff, 4, &[&a.msg[0], &a.msg[1], &a.msg[2], &a.msg[3]], t);
        }
        // sources for q, k, v before the input norms
        let (sq, sk, sv): (&Buf, &Buf, &Buf) = if let Some(qb) = &lw.attn_q {
            if lw.n_attn_q == 3 {
                self.attn_res(gpu, &mut rec, &a.blocks, &slots_before, &a.cur, 0, qb, 0, 3, &[&a.ares[0], &a.ares[1], &a.ares[2]], t);
                (&a.ares[0], &a.ares[1], &a.ares[2])
            } else {
                self.attn_res(gpu, &mut rec, &a.blocks, &slots_before, &a.cur, 0, qb, 0, 1, &[&a.ares[0]], t);
                (&a.ares[0], &a.ares[0], &a.ares[0])
            }
        } else {
            (&a.cur, &a.cur, &a.cur)
        };
        let (sq, sk, sv) = if have_msg {
            let c = lw.ctrl.as_ref().context("loop layer without controllers")?;
            self.msg_gate(gpu, &mut rec, &c[0], sq, &a.msg[0], &a.gated[0], t);
            self.msg_gate(gpu, &mut rec, &c[1], sk, &a.msg[1], &a.gated[1], t);
            self.msg_gate(gpu, &mut rec, &c[2], sv, &a.msg[2], &a.gated[2], t);
            (&a.gated[0], &a.gated[1], &a.gated[2])
        } else {
            (sq, sk, sv)
        };
        self.rmsnorm(gpu, &mut rec, sq, &lw.norm_q, &a.xq, t, D as u32, 0);
        self.rmsnorm(gpu, &mut rec, sk, &lw.norm_k, &a.xk, t, D as u32, 0);
        self.rmsnorm(gpu, &mut rec, sv, &lw.norm_v, &a.xv, t, D as u32, 0);

        // ---- attention
        let c = &self.cfg;
        let nh = c.num_attention_heads as u32;
        let nkv = c.num_key_value_heads as u32;
        self.matmul(gpu, &mut rec, &lw.wq, &a.xq, &a.q, t);
        self.scaled_clip(gpu, &mut rec, &a.q, &lw.q_scale, t, nh, 128, c.qk_l2_clip_threshold, lw.full, pos0);
        self.matmul(gpu, &mut rec, &lw.wr, &a.xq, &a.rs, t);
        self.scaled_clip(gpu, &mut rec, &a.rs, &lw.rel_scale, t, nh, 16, c.rel_l2_clip_threshold, lw.full, pos0);
        self.matmul(gpu, &mut rec, &lw.wsink, &a.xq, &a.sinks, t);
        self.matmul(gpu, &mut rec, &lw.wgate, &a.xq, &a.gate, t);
        self.matmul(gpu, &mut rec, &lw.wk, &a.xk, &a.klin, t);
        self.matmul(gpu, &mut rec, &lw.wv, &a.xv, &a.vlin, t);
        let kvw = nkv * 128;
        if lw.full {
            self.matmul(gpu, &mut rec, lw.k_dyn.as_ref().unwrap(), &a.xk, &a.dk, t);
            self.matmul(gpu, &mut rec, lw.v_dyn.as_ref().unwrap(), &a.xv, &a.dv, t);
            self.conv(gpu, &mut rec, &a.klin, &st.hk, &lw.k_conv, Some(&a.dk), &a.kc, kvw, t, pos0);
            self.conv(gpu, &mut rec, &a.vlin, &st.hv, &lw.v_conv, Some(&a.dv), &a.vc, kvw, t, pos0);
        } else {
            self.conv(gpu, &mut rec, &a.klin, &st.hk, &lw.k_conv, None, &a.kc, kvw, t, pos0);
            self.conv(gpu, &mut rec, &a.vlin, &st.hv, &lw.v_conv, None, &a.vc, kvw, t, pos0);
        }
        let thr = (c.qk_l2_clip_threshold as f32).to_bits();
        let eps = (c.rms_norm_eps as f32).to_bits();
        gpu.dispatch(&mut rec, K::KvStore, &[t, nkv, pos0, st.cap, 0, thr, eps, kvw], &[a.kc.all(), st.k.all()], grid(t * nkv));
        gpu.dispatch(&mut rec, K::KvStore, &[t, nkv, pos0, st.cap, 1, thr, eps, kvw], &[a.vc.all(), st.v.all()], grid(t * nkv));
        let qblocks = t.div_ceil(8);
        let scale = (1.0f32 / 128f32.sqrt()).to_bits();
        // split-K over the key range for short query blocks (decode)
        let last = pos0 + t - 1;
        let first = if lw.window > 0 { (pos0 + 1).saturating_sub(lw.window) } else { 0 } / 32 * 32;
        let span = last + 1 - first;
        let (splits, split_len) = if t <= 8 && span > 256 && std::env::var_os("INKLING_NO_SPLIT").is_none() {
            let s = span.div_ceil(256).min(MAX_SPLITS as u32);
            (s, span.div_ceil(s).next_multiple_of(32))
        } else {
            (1, span)
        };
        gpu.dispatch(
            &mut rec,
            K::Attention,
            &[t, nh, nkv, pos0, st.cap, lw.window, lw.rel_extent, scale, nh * 128, nh * 16, nh, qblocks, splits, split_len],
            &[a.q.all(), st.k.all(), st.v.all(), a.rs.all(), lw.rel_proj.all(), a.sinks.all(), a.ao.all(), a.part.all()],
            grid(nh * qblocks * splits),
        );
        if splits > 1 {
            gpu.dispatch(&mut rec, K::AttnCombine, &[t, nh, splits, nh], &[a.part.all(), a.sinks.all(), a.ao.all()], grid(t * nh));
        }
        self.ew(gpu, &mut rec, 1, t * nh * 128, &a.ao, &a.gate, &a.gate, &a.q);
        self.matmul(gpu, &mut rec, &lw.wo, &a.q, &a.o, t);
        self.conv(gpu, &mut rec, &a.o, &st.ha, &lw.attn_conv, None, &a.c, D as u32, t, pos0);

        // ---- residual / block boundary
        if boundary {
            gpu.copy_inline(&mut rec, &a.cur, 0, &a.blocks, ((l / block) as u32 * so * 4) as u64, (n * 4) as u64);
            gpu.copy_inline(&mut rec, &a.c, 0, &a.resid, 0, (n * 4) as u64);
        } else {
            self.ew(gpu, &mut rec, 0, n, &a.cur, &a.c, &a.c, &a.resid);
        }
        let nb_after = if boundary { l / block + 1 } else { nb_before };
        let slots_after: Vec<u32> = (0..nb_after as u32).collect();
        self.attn_res(gpu, &mut rec, &a.blocks, &slots_after, &a.resid, 0, &lw.mlp_q, 0, 1, &[&a.m], t);
        let msrc = if have_msg {
            let cw = lw.ctrl.as_ref().unwrap();
            self.msg_gate(gpu, &mut rec, &cw[3], &a.m, &a.msg[3], &a.gated[3], t);
            &a.gated[3]
        } else {
            &a.m
        };
        self.rmsnorm(gpu, &mut rec, msrc, &lw.mlp_norm, &a.xm, t, D as u32, 0);

        // ---- MoE (routing needs a host round trip to stream experts)
        self.matmul(gpu, &mut rec, &lw.router, &a.xm, &a.rlog, t);
        let ne = c.n_routed_experts as u32;
        let topk = c.num_experts_per_tok as u32;
        gpu.dispatch(&mut rec, K::Router, &[t, ne, topk, (c.route_scale as f32).to_bits()], &[a.rlog.all(), lw.router_bias.all(), a.ids.all(), a.wts.all()], grid(t));
        let na = t * topk;
        gpu.copy_at_end(&mut rec, &a.ids, 0, &a.route_stage, 0, (na * 4) as u64);
        gpu.copy_at_end(&mut rec, &a.wts, 0, &a.route_stage, (na * 4) as u64, (na * 4) as u64);
        gpu.submit(rec);
        let tw = std::time::Instant::now();
        let bytes = gpu.read(&self.acts.route_stage, (na * 8) as u64)?;
        self.timings.route_syncs += 1;
        self.timings.route_wait_s += tw.elapsed().as_secs_f64();
        let ids: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&bytes[..(na * 4) as usize]).to_vec();
        let mut rec = gpu.recorder();
        self.moe(gpu, &mut rec, l, &ids, t)?;

        let a = &self.acts;
        let st = &self.apps[app];
        let lw = &self.layers[l];
        self.conv(gpu, &mut rec, &a.mo, &st.hm, &lw.mlp_conv, None, &a.mc, D as u32, t, pos0);
        self.ew(gpu, &mut rec, 0, n, &a.resid, &a.mc, &a.mc, &a.cur);

        if self.dump.is_some() {
            gpu.copy_at_end(&mut rec, &a.cur, 0, &a.dbg_stage, 0, (n * 4) as u64);
            gpu.submit(rec);
            let b = gpu.read(&self.acts.dbg_stage, (n * 4) as u64)?;
            self.dump.as_mut().unwrap().push(bytemuck::cast_slice::<u8, f32>(&b).to_vec());
            rec = gpu.recorder();
        }
        Ok(rec)
    }

    fn moe(&mut self, gpu: &mut Gpu, rec: &mut Recorder, l: usize, ids: &[u32], t: u32) -> Result<()> {
        let topk = self.cfg.num_experts_per_tok as u32;
        let ne = self.cfg.n_routed_experts;
        let na = ids.len();
        let mut by_e: Vec<Vec<u32>> = vec![Vec::new(); ne];
        for (i, &e) in ids.iter().enumerate() {
            by_e[e as usize].push(i as u32);
        }
        let used: Vec<u16> = (0..ne as u16).filter(|&e| !by_e[e as usize].is_empty()).collect();
        let te = std::time::Instant::now();
        let slots = self.experts.ensure(gpu, l, &used)?;
        self.timings.ensure_s += te.elapsed().as_secs_f64();
        let lay = self.experts.layout(l);
        let cache = self.experts.cache(l);
        let n_pools = cache.pools.len();
        let a = &self.acts;
        let inter = self.cfg.intermediate_size as u32;

        // assignments sorted by expert; per pool: gemm tiles, or gemv items for both passes
        let mut sorted = Vec::with_capacity(na);
        let mut inv = vec![0u32; na];
        let mut tiles: Vec<Vec<[u32; 4]>> = vec![Vec::new(); n_pools];
        let mut items: Vec<Vec<[u32; 4]>> = vec![Vec::new(); n_pools];
        for (ui, &e) in used.iter().enumerate() {
            let (pi, local) = cache.locate(slots[ui]);
            let list = &by_e[e as usize];
            let start = sorted.len() as u32;
            for &asg in list {
                let si = sorted.len() as u32;
                inv[asg as usize] = si;
                items[pi].push([local, asg / topk, si, 0]);
                sorted.push(asg);
            }
            let mut off = 0;
            while off < list.len() {
                let cnt = (list.len() - off).min(64);
                tiles[pi].push([local, start + off as u32, cnt as u32, 0]);
                off += cnt;
            }
        }
        gpu.queue.write_buffer(&a.sorted.raw, 0, bytemuck::cast_slice(&sorted));
        gpu.queue.write_buffer(&a.inv.raw, 0, bytemuck::cast_slice(&inv));
        let use_gemv = na <= 32;
        // per pool: gemm tiles, or [pass-0 items | pad to 16 | pass-1 items] for gemv
        let mut item2_off = vec![0u64; n_pools];
        for pi in 0..n_pools {
            if use_gemv {
                if items[pi].is_empty() {
                    continue;
                }
                let mut list = items[pi].clone();
                let pad = list.len().next_multiple_of(16);
                list.resize(pad, [0; 4]);
                item2_off[pi] = (pad * 16) as u64;
                list.extend(items[pi].iter().map(|it| [it[0], it[2], it[2], 0]));
                gpu.queue.write_buffer(&a.tiles[pi].raw, 0, bytemuck::cast_slice(&list));
            } else if !tiles[pi].is_empty() {
                gpu.queue.write_buffer(&a.tiles[pi].raw, 0, bytemuck::cast_slice(&tiles[pi]));
            }
        }
        for pass in 0..2u32 {
            let (k, nrows, qw, sw, off_q, off_s, off_f) = if pass == 0 {
                (D as u32, 2 * inter, lay.qw13, lay.sw13, 0, lay.off_s13, lay.off_f13)
            } else {
                (inter, D as u32, lay.qw2, lay.sw2, lay.off_q2, lay.off_s2, lay.off_f2)
            };
            let (x, y) = if pass == 0 { (&a.xm, &a.y13) } else { (&a.act, &a.y2) };
            for pi in 0..n_pools {
                let pool = &cache.pools[pi].buf;
                if use_gemv {
                    let cnt = items[pi].len() as u32;
                    if cnt == 0 {
                        continue;
                    }
                    let tiles_n = nrows.div_ceil(4);
                    let params = [lay.fmt, pass, k, nrows, k, nrows, lay.slot_words, 0, off_s, off_q, qw, sw, off_f, tiles_n, cnt, 0];
                    let tb = &a.tiles[pi];
                    let off = if pass == 0 { 0 } else { item2_off[pi] };
                    let sl = crate::gpu::Slice { buf: tb, offset: off, size: tb.size - off };
                    gpu.dispatch(rec, K::MoeGemv, &params, &[x.all(), pool.all(), sl, y.all()], grid(cnt * tiles_n));
                } else {
                    let cnt = tiles[pi].len() as u32;
                    if cnt == 0 {
                        continue;
                    }
                    let tiles_n = nrows.div_ceil(64);
                    let gather = if pass == 0 { topk } else { 0 };
                    let params = [lay.fmt, pass, k, nrows, k, nrows, lay.slot_words, gather, off_s, off_q, qw, sw, off_f, tiles_n, cnt, 0];
                    gpu.dispatch(rec, K::MoeGemm, &params, &[x.all(), pool.all(), a.tiles[pi].all(), a.sorted.all(), y.all()], grid(cnt * tiles_n));
                }
            }
            if pass == 0 {
                gpu.dispatch(
                    rec,
                    K::Swiglu,
                    &[na as u32, inter, (self.cfg.mlp_gate_softcap as f32).to_bits(), (self.cfg.mlp_up_softcap as f32).to_bits()],
                    &[a.y13.all(), a.act.all()],
                    grid((na as u32 * inter).div_ceil(256)),
                );
            }
        }
        gpu.dispatch(rec, K::MoeCombine, &[t, D as u32, topk], &[a.y2.all(), a.inv.all(), a.wts.all(), a.mo.all()], grid((t * D as u32).div_ceil(256)));
        Ok(())
    }

    /// Run `rows` (raw bf16-decoded embeddings, `t` x D) at positions pos0..; rows listed in
    /// `inject` are replaced after the embedding norm (image features). Returns last-row logits.
    pub fn forward(&mut self, gpu: &mut Gpu, rows: &[f32], inject: &[(usize, Vec<f32>)], pos0: usize, want_logits: bool) -> Result<Option<Vec<f32>>> {
        let t = rows.len() / D;
        assert!(t >= 1 && t <= self.tm, "chunk of {t} rows exceeds the prefill chunk size {}", self.tm);
        if pos0 + t > self.ctx {
            bail!("context length {} exceeded", self.ctx);
        }
        gpu.queue.write_buffer(&self.acts.next.raw, 0, f32_bytes(rows));
        let mut rec = gpu.recorder();
        self.rmsnorm(gpu, &mut rec, &self.acts.next, &self.embed_norm, &self.acts.cur, t as u32, D as u32, 0);
        gpu.submit(rec);
        for (r, feat) in inject {
            gpu.queue.write_buffer(&self.acts.cur.raw, (r * D * 4) as u64, f32_bytes(feat));
        }
        self.run_stack(gpu, t as u32, pos0 as u32, want_logits)
    }
}
