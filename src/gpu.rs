//! Thin wgpu layer: device setup, buffers, compute kernels and a dispatch recorder.
//!
//! Every kernel follows the same binding convention: `@binding(0)` is a small uniform
//! parameter block, `@binding(1..)` are storage buffers. Parameters for all dispatches of
//! one forward pass are packed into a single uniform buffer (one aligned slot per dispatch)
//! and uploaded with one `write_buffer`, so bind groups never change between forwards and
//! are cached.

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_BUF_ID: AtomicU64 = AtomicU64::new(1);

pub struct Buf {
    pub raw: wgpu::Buffer,
    pub id: u64,
    pub size: u64,
}

impl Buf {
    fn new(raw: wgpu::Buffer, size: u64) -> Self {
        crate::metrics::GPU_BYTES.fetch_add(size as i64, Ordering::Relaxed);
        Buf {
            raw,
            id: NEXT_BUF_ID.fetch_add(1, Ordering::Relaxed),
            size,
        }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        crate::metrics::GPU_BYTES.fetch_sub(self.size as i64, Ordering::Relaxed);
    }
}

/// A (sub)range of a buffer bound to one kernel slot.
#[derive(Clone, Copy)]
pub struct Slice<'a> {
    pub buf: &'a Buf,
    pub offset: u64,
    pub size: u64,
}

impl Buf {
    pub fn all(&self) -> Slice<'_> {
        Slice {
            buf: self,
            offset: 0,
            size: self.size,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum K {
    RmsNorm,
    Gemv,
    Gemm,
    Copy,
    AttnRes,
    ScaledClip,
    Conv,
    ConvHist,
    KvStore,
    Attention,
    Router,
    MoeGemm,
    MoeGemv,
    Swiglu,
    MoeCombine,
    Ew,
    LayerNorm,
    BiasAct,
    Rope2d,
    VitAttn,
    AttnCombine,
}

const MOE_COMMON: &str = include_str!("shaders/moe_common.wgsl");

const KERNELS: &[(K, &str)] = &[
    (K::RmsNorm, include_str!("shaders/rmsnorm.wgsl")),
    (K::Gemv, include_str!("shaders/gemv.wgsl")),
    (K::Gemm, include_str!("shaders/gemm.wgsl")),
    (K::Copy, include_str!("shaders/copy.wgsl")),
    (K::AttnRes, include_str!("shaders/attn_res.wgsl")),
    (K::ScaledClip, include_str!("shaders/scaled_clip.wgsl")),
    (K::Conv, include_str!("shaders/conv.wgsl")),
    (K::ConvHist, include_str!("shaders/conv_hist.wgsl")),
    (K::KvStore, include_str!("shaders/kv_store.wgsl")),
    (K::Attention, include_str!("shaders/attention.wgsl")),
    (K::Router, include_str!("shaders/router.wgsl")),
    (K::MoeGemm, include_str!("shaders/moe_gemm.wgsl")),
    (K::MoeGemv, include_str!("shaders/moe_gemv.wgsl")),
    (K::Swiglu, include_str!("shaders/swiglu.wgsl")),
    (K::MoeCombine, include_str!("shaders/moe_combine.wgsl")),
    (K::Ew, include_str!("shaders/ew.wgsl")),
    (K::LayerNorm, include_str!("shaders/layernorm.wgsl")),
    (K::BiasAct, include_str!("shaders/bias_act.wgsl")),
    (K::Rope2d, include_str!("shaders/rope2d.wgsl")),
    (K::VitAttn, include_str!("shaders/vit_attn.wgsl")),
    (K::AttnCombine, include_str!("shaders/attn_combine.wgsl")),
];

const COMMON: &str = include_str!("shaders/common.wgsl");

/// Words of parameters per dispatch (64 bytes).
pub const PARAM_WORDS: usize = 16;
const MAX_DISPATCHES: usize = 16384;
/// Max workgroups per dispatch dimension guaranteed by WebGPU.
pub const MAX_WG_DIM: u32 = 65535;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pipelines: HashMap<K, (wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    params_buf: Buf,
    param_align: usize,
    bind_cache: HashMap<(K, Vec<(u64, u64, u64)>), wgpu::BindGroup>,
}

enum Op {
    Dispatch { kernel: K, bind_group: wgpu::BindGroup, wg: [u32; 3] },
    Copy { src: wgpu::Buffer, src_off: u64, dst: wgpu::Buffer, dst_off: u64, size: u64 },
}

/// Records the dispatches of one submission.
pub struct Recorder {
    params: Vec<u8>,
    ops: Vec<Op>,
    copies: Vec<(wgpu::Buffer, u64, wgpu::Buffer, u64, u64)>,
    /// Dispatches already submitted by `flush`.
    flushed: usize,
    n_dispatch: usize,
}

/// Split a 1-D workgroup count into a 2-D grid within the per-dimension limit.
/// Shaders recover the linear index as `wid.x + wid.y * nwg.x`.
pub fn grid(n: u32) -> [u32; 3] {
    if n <= MAX_WG_DIM {
        [n.max(1), 1, 1]
    } else {
        let y = n.div_ceil(MAX_WG_DIM);
        [n.div_ceil(y), y, 1]
    }
}

impl Gpu {
    fn instance(backends: wgpu::Backends) -> wgpu::Instance {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        wgpu::Instance::new(desc)
    }

    pub fn new(adapter_index: Option<usize>) -> Result<Self> {
        // WGPU_BACKEND (vulkan, dx12, metal, gl) overrides wgpu's default backend choice.
        let candidates = [wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY)];

        let adapter = match adapter_index {
            Some(i) => {
                let instance = Self::instance(wgpu::Backends::all());
                let all = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
                all.into_iter()
                    .nth(i)
                    .with_context(|| format!("no adapter with index {i}"))?
            }
            None => {
                let mut found = None;
                for b in candidates {
                    let instance = Self::instance(b);
                    if let Ok(a) =
                        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::HighPerformance,
                            force_fallback_adapter: false,
                            compatible_surface: None,
                            ..Default::default()
                        }))
                    {
                        found = Some(a);
                        break;
                    }
                }
                found.context("no suitable GPU adapter found")?
            }
        };
        let info = adapter.get_info();
        let alimits = adapter.limits();

        // Ask for everything the adapter offers on the axes that matter for big weight buffers.
        let mut limits = wgpu::Limits::default();
        limits.max_buffer_size = alimits.max_buffer_size;
        limits.max_storage_buffer_binding_size = alimits.max_storage_buffer_binding_size;
        limits.max_storage_buffers_per_shader_stage =
            alimits.max_storage_buffers_per_shader_stage.min(16);
        limits.max_compute_workgroup_storage_size = alimits.max_compute_workgroup_storage_size;
        limits.max_compute_invocations_per_workgroup =
            alimits.max_compute_invocations_per_workgroup;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("inkling"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("requesting GPU device")?;

        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            tracing::error!("wgpu error: {e}");
            panic!("wgpu error: {e}");
        }));

        let param_align =
            (limits.min_uniform_buffer_offset_alignment as usize).max(PARAM_WORDS * 4);
        let params_buf = Self::make_buffer(
            &device,
            "params",
            (param_align * MAX_DISPATCHES) as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let mut pipelines = HashMap::new();
        for (k, src) in KERNELS {
            let t0 = std::time::Instant::now();
            let moe = if matches!(k, K::MoeGemm | K::MoeGemv) { MOE_COMMON } else { "" };
            let code = format!("{COMMON}\n{moe}\n{src}");
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{k:?}")),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{k:?}")),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                // Every kernel writes its workgroup arrays before reading them. Skipping the
                // implicit zero-fill also avoids ~30 s compiles of large tiles on DX12 (FXC).
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            });
            let bgl = pipeline.get_bind_group_layout(0);
            pipelines.insert(*k, (pipeline, bgl));
            let dt = t0.elapsed().as_secs_f64();
            if dt > 0.5 {
                tracing::warn!("compiling kernel {k:?} took {dt:.1} s");
            }
        }

        Ok(Self {
            device,
            queue,
            info,
            limits,
            pipelines,
            params_buf,
            param_align,
            bind_cache: HashMap::new(),
        })
    }

    fn make_buffer(
        device: &wgpu::Device,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Buf {
        let size = size.max(16).next_multiple_of(16);
        let raw = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Buf::new(raw, size)
    }

    /// A zero-initialised storage buffer.
    pub fn storage(&self, label: &str, size: u64) -> Buf {
        Self::make_buffer(
            &self.device,
            label,
            size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    }

    /// A storage buffer initialised with `data`.
    pub fn storage_init(&self, label: &str, data: &[u8]) -> Buf {
        let size = (data.len() as u64).max(16).next_multiple_of(16);
        let raw = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        raw.slice(..)
            .get_mapped_range_mut()
            .expect("mapped buffer")
            .slice(..data.len())
            .copy_from_slice(data);
        raw.unmap();
        Buf::new(raw, size)
    }

    pub fn readback_buffer(&self, size: u64) -> Buf {
        Self::make_buffer(
            &self.device,
            "readback",
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )
    }

    pub fn max_binding(&self) -> u64 {
        self.limits
            .max_storage_buffer_binding_size
            .min(self.limits.max_buffer_size)
    }

    pub fn recorder(&self) -> Recorder {
        Recorder {
            params: Vec::with_capacity(self.param_align * 512),
            ops: Vec::with_capacity(512),
            copies: Vec::new(),
            flushed: 0,
            n_dispatch: 0,
        }
    }

    /// Queue one dispatch. `params` are up to 16 words (u32 or f32 bit patterns).
    pub fn dispatch(
        &mut self,
        rec: &mut Recorder,
        kernel: K,
        params: &[u32],
        bufs: &[Slice],
        wg: [u32; 3],
    ) {
        assert!(params.len() <= PARAM_WORDS);
        let slot = rec.flushed + rec.n_dispatch;
        rec.n_dispatch += 1;
        assert!(
            slot < MAX_DISPATCHES,
            "too many dispatches in one submission"
        );
        let off = slot * self.param_align;
        rec.params.resize(off + self.param_align, 0);
        for (i, w) in params.iter().enumerate() {
            rec.params[off + i * 4..off + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }

        let mut key_bufs = Vec::with_capacity(bufs.len() + 1);
        key_bufs.push((self.params_buf.id, off as u64, (PARAM_WORDS * 4) as u64));
        for s in bufs {
            key_bufs.push((s.buf.id, s.offset, s.size));
        }
        let key = (kernel, key_bufs);
        let bind_group = if let Some(bg) = self.bind_cache.get(&key) {
            bg.clone()
        } else {
            let (_, bgl) = &self.pipelines[&kernel];
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.params_buf.raw,
                    offset: off as u64,
                    size: std::num::NonZeroU64::new((PARAM_WORDS * 4) as u64),
                }),
            }];
            for (i, s) in bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: (i + 1) as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &s.buf.raw,
                        offset: s.offset,
                        size: std::num::NonZeroU64::new(s.size),
                    }),
                });
            }
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: bgl,
                entries: &entries,
            });
            self.bind_cache.insert(key, bg.clone());
            bg
        };
        rec.ops.push(Op::Dispatch { kernel, bind_group, wg });
    }

    /// Queue a buffer copy in order with the surrounding dispatches.
    pub fn copy_inline(&self, rec: &mut Recorder, src: &Buf, src_off: u64, dst: &Buf, dst_off: u64, size: u64) {
        rec.ops.push(Op::Copy { src: src.raw.clone(), src_off, dst: dst.raw.clone(), dst_off, size });
    }

    /// Queue a buffer copy executed after all dispatches recorded so far... and before later ones.
    /// (Copies are encoded at the end of the submission; use only for outputs.)
    pub fn copy_at_end(
        &self,
        rec: &mut Recorder,
        src: &Buf,
        src_off: u64,
        dst: &Buf,
        dst_off: u64,
        size: u64,
    ) {
        rec.copies
            .push((src.raw.clone(), src_off, dst.raw.clone(), dst_off, size));
    }

    /// Submit the dispatches recorded since the last flush and keep recording into `rec`
    /// (parameter slots keep counting up, so in-flight work is never overwritten).
    /// Keeps individual submissions short on slow GPUs (OS watchdogs reset long ones).
    pub fn flush(&self, rec: &mut Recorder) {
        let ops = std::mem::take(&mut rec.ops);
        let start = rec.flushed;
        rec.flushed = start + rec.n_dispatch;
        rec.n_dispatch = 0;
        self.encode_and_submit(&rec.params, start, &ops, &[]);
        rec.ops = Vec::with_capacity(ops.len());
    }

    pub fn submit(&self, rec: Recorder) -> wgpu::SubmissionIndex {
        self.encode_and_submit(&rec.params, rec.flushed, &rec.ops, &rec.copies)
    }

    fn encode_and_submit(
        &self,
        params: &[u8],
        first_slot: usize,
        ops: &[Op],
        copies: &[(wgpu::Buffer, u64, wgpu::Buffer, u64, u64)],
    ) -> wgpu::SubmissionIndex {
        let lo = first_slot * self.param_align;
        if params.len() > lo {
            self.queue
                .write_buffer(&self.params_buf.raw, lo as u64, &params[lo..]);
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut i = 0;
        while i < ops.len() {
            if let Op::Copy { src, src_off, dst, dst_off, size } = &ops[i] {
                enc.copy_buffer_to_buffer(src, *src_off, dst, *dst_off, *size);
                i += 1;
                continue;
            }
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            let mut current: Option<K> = None;
            while i < ops.len() {
                let Op::Dispatch { kernel, bind_group, wg } = &ops[i] else { break };
                if current != Some(*kernel) {
                    pass.set_pipeline(&self.pipelines[kernel].0);
                    current = Some(*kernel);
                }
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
                i += 1;
            }
        }
        for (src, so, dst, d_off, size) in copies {
            enc.copy_buffer_to_buffer(src, *so, dst, *d_off, *size);
        }
        self.queue.submit([enc.finish()])
    }

    /// Copy `size` bytes of `src` to the host (blocking).
    pub fn read(&self, staging: &Buf, size: u64) -> Result<Vec<u8>> {
        let slice = staging.raw.slice(0..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow!("device poll failed: {e}"))?;
        rx.recv()?.map_err(|e| anyhow!("buffer map failed: {e}"))?;
        let out = slice
            .get_mapped_range()
            .map_err(|e| anyhow!("buffer view failed: {e:?}"))?
            .to_vec();
        staging.raw.unmap();
        Ok(out)
    }

    /// Copy a storage buffer region to the host as f32.
    pub fn read_f32(&self, src: &Buf, offset: u64, count: usize) -> Result<Vec<f32>> {
        let size = (count * 4) as u64;
        let staging = self.readback_buffer(size);
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&src.raw, offset, &staging.raw, 0, size.next_multiple_of(4));
        self.queue.submit([enc.finish()]);
        let bytes = self.read(&staging, size)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    pub fn clear(&self, buf: &Buf) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&buf.raw, 0, None);
        self.queue.submit([enc.finish()]);
    }

    pub fn copy(&self, src: &Buf, dst: &Buf) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&src.raw, 0, &dst.raw, 0, src.size.min(dst.size));
        self.queue.submit([enc.finish()]);
    }
}
