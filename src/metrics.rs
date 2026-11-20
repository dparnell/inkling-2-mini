//! Process-wide runtime metrics for the dashboard (`serve --tui`).
//!
//! The engine thread reports request progress (queued → prefill → generating → done) and every
//! generated slot; the server reports rejected HTTP requests; a sampler thread adds GPU and
//! memory readings once a second and turns the counters into per-second rates.
//!
//! GPU utilisation and device memory come from NVML on NVIDIA GPUs and from sysfs on Linux AMD
//! GPUs. Elsewhere the dashboard falls back to "engine busy" (the share of wall time the engine
//! spent running requests, which on this GPU-bound runtime tracks GPU load) and to the bytes of
//! GPU buffers this process allocated.

use crate::engine::Usage;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Bytes currently held in GPU buffers created through `gpu.rs`.
pub static GPU_BYTES: AtomicI64 = AtomicI64::new(0);

/// MoE expert cache counters (cumulative), published by the engine.
pub static EXPERT_HITS: AtomicU64 = AtomicU64::new(0);
pub static EXPERT_MISSES: AtomicU64 = AtomicU64::new(0);
pub static EXPERT_UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);

/// Set when the dashboard asks the server to stop.
pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);
pub static SHUTDOWN_NOTIFY: LazyLock<tokio::sync::Notify> = LazyLock::new(tokio::sync::Notify::new);

pub fn request_shutdown() {
    SHUTDOWN.store(true, Ordering::SeqCst);
    SHUTDOWN_NOTIFY.notify_waiters();
    SHUTDOWN_NOTIFY.notify_one();
}

const MAX_SAMPLES: usize = 3600;
const MAX_REQUESTS: usize = 500;
const MAX_LOGS: usize = 1000;

static METRICS: LazyLock<Metrics> = LazyLock::new(|| Metrics {
    s: Mutex::new(State {
        info: Info {
            status: "starting".into(),
            ..Default::default()
        },
        started: Instant::now(),
        last_tick: Instant::now(),
        acc_gen: 0,
        acc_audio: 0,
        acc_prompt: 0,
        busy_accum: Duration::ZERO,
        busy_since: None,
        ttft_acc: Vec::new(),
        samples: VecDeque::new(),
        requests: VecDeque::new(),
        next_id: 1,
        totals: Totals::default(),
        logs: VecDeque::new(),
        last_expert: (0, 0, 0),
    }),
});

pub fn metrics() -> &'static Metrics {
    &METRICS
}

#[derive(Default, Clone)]
pub struct Info {
    pub model: String,
    pub gpu: String,
    pub addr: String,
    pub status: String,
    /// Source of the utilisation/VRAM readings ("NVML", "sysfs") if any.
    pub gpu_source: Option<String>,
}

/// One per-second sample.
#[derive(Clone, Default)]
pub struct Sample {
    /// generated slots (text tokens + speech frames) per second
    pub gen_tps: f64,
    /// of which speech frames per second
    pub audio_fps: f64,
    /// prompt tokens prefilled (excluding cached) per second
    pub prompt_tps: f64,
    /// average time to first token of requests whose first token arrived this second
    pub ttft_ms: Option<f64>,
    /// share of the second the engine spent running requests, 0..1
    pub busy: f64,
    /// device utilisation in percent (NVML / sysfs)
    pub gpu_util: Option<f64>,
    /// device memory used / total in bytes (NVML / sysfs)
    pub vram: Option<(u64, u64)>,
    /// bytes in GPU buffers this process allocated
    pub gpu_alloc: u64,
    /// resident memory of this process in bytes
    pub rss: u64,
    /// share of expert lookups served from the GPU cache this second
    pub expert_hit: Option<f64>,
    /// bytes of expert weights streamed to the GPU per second
    pub upload_bps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReqState {
    Queued,
    Prefill,
    Generating,
    Done(String),
    Failed(String),
    Rejected(u16, String),
}

#[derive(Clone)]
pub struct ReqRow {
    pub id: u64,
    pub endpoint: String,
    /// input modalities, e.g. "text+image"
    pub input: String,
    /// output modality, e.g. "text", "audio"
    pub output: &'static str,
    pub stream: bool,
    pub state: ReqState,
    /// seconds since process start when the request arrived
    pub at: f64,
    pub queued_at: Instant,
    pub prompt: usize,
    pub cached: usize,
    pub generated: usize,
    pub audio_frames: usize,
    pub prefill_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub decode_tps: Option<f64>,
    pub total_ms: Option<f64>,
    /// prefill progress (rows done, rows to do)
    pub progress: Option<(usize, usize)>,
    started: Option<Instant>,
    gen_started: Option<Instant>,
}

impl ReqRow {
    pub fn active(&self) -> bool {
        matches!(
            self.state,
            ReqState::Queued | ReqState::Prefill | ReqState::Generating
        )
    }
}

#[derive(Clone, Default)]
pub struct Totals {
    pub requests: u64,
    pub completed: u64,
    pub failed: u64,
    pub rejected: u64,
    pub cancelled: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub generated: u64,
    pub audio_frames: u64,
}

struct State {
    info: Info,
    started: Instant,
    last_tick: Instant,
    acc_gen: u64,
    acc_audio: u64,
    acc_prompt: u64,
    busy_accum: Duration,
    busy_since: Option<Instant>,
    ttft_acc: Vec<f64>,
    samples: VecDeque<Sample>,
    requests: VecDeque<ReqRow>,
    next_id: u64,
    totals: Totals,
    logs: VecDeque<String>,
    last_expert: (u64, u64, u64),
}

impl State {
    fn row(&mut self, id: u64) -> Option<&mut ReqRow> {
        self.requests.iter_mut().rev().find(|r| r.id == id)
    }

    fn push_row(&mut self, r: ReqRow) {
        self.requests.push_back(r);
        while self.requests.len() > MAX_REQUESTS {
            // drop the oldest finished row; keep active ones
            match self.requests.iter().position(|r| !r.active()) {
                Some(i) => {
                    self.requests.remove(i);
                }
                None => break,
            }
        }
    }

    fn busy_end(&mut self, now: Instant) {
        if let Some(s) = self.busy_since.take() {
            self.busy_accum += now - s;
        }
    }
}

/// A consistent copy of everything the dashboard draws.
pub struct Snapshot {
    pub info: Info,
    pub uptime: Duration,
    pub samples: Vec<Sample>,
    pub requests: Vec<ReqRow>,
    pub totals: Totals,
    pub logs: Vec<String>,
}

pub struct Metrics {
    s: Mutex<State>,
}

impl Metrics {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.s.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn set_info(&self, f: impl FnOnce(&mut Info)) {
        f(&mut self.lock().info);
    }

    pub fn log(&self, line: String) {
        let mut s = self.lock();
        s.logs.push_back(line);
        while s.logs.len() > MAX_LOGS {
            s.logs.pop_front();
        }
    }

    /// A request entered the engine queue; returns its id.
    pub fn queued(
        &self,
        endpoint: &str,
        input: String,
        output: &'static str,
        stream: bool,
        prompt: usize,
    ) -> u64 {
        let mut s = self.lock();
        let id = s.next_id;
        s.next_id += 1;
        s.totals.requests += 1;
        let at = s.started.elapsed().as_secs_f64();
        s.push_row(ReqRow {
            id,
            endpoint: endpoint.to_string(),
            input,
            output,
            stream,
            state: ReqState::Queued,
            at,
            queued_at: Instant::now(),
            prompt,
            cached: 0,
            generated: 0,
            audio_frames: 0,
            prefill_ms: None,
            ttft_ms: None,
            decode_tps: None,
            total_ms: None,
            progress: None,
            started: None,
            gen_started: None,
        });
        id
    }

    /// An HTTP request failed before reaching the engine (or with an error status).
    pub fn rejected(&self, endpoint: &str, status: u16, message: String) {
        let mut s = self.lock();
        s.totals.requests += 1;
        s.totals.rejected += 1;
        let at = s.started.elapsed().as_secs_f64();
        let id = s.next_id;
        s.next_id += 1;
        s.push_row(ReqRow {
            id,
            endpoint: endpoint.to_string(),
            input: String::new(),
            output: "",
            stream: false,
            state: ReqState::Rejected(status, message),
            at,
            queued_at: Instant::now(),
            prompt: 0,
            cached: 0,
            generated: 0,
            audio_frames: 0,
            prefill_ms: None,
            ttft_ms: None,
            decode_tps: None,
            total_ms: Some(0.0),
            progress: None,
            started: None,
            gen_started: None,
        });
    }

    /// Prefill progress for the dashboard.
    pub fn progress(&self, id: u64, done: usize, total: usize) {
        if let Some(r) = self.lock().row(id) {
            r.progress = Some((done, total));
        }
    }

    /// The client went away mid-generation.
    pub fn cancelled(&self, id: u64) {
        let now = Instant::now();
        let mut s = self.lock();
        s.busy_end(now);
        s.totals.cancelled += 1;
        if let Some(r) = s.row(id) {
            r.state = ReqState::Done("cancelled".into());
            r.total_ms = Some((now - r.queued_at).as_secs_f64() * 1e3);
        }
    }

    /// The engine picked the request up.
    pub fn started(&self, id: u64) {
        let now = Instant::now();
        let mut s = self.lock();
        s.busy_end(now);
        s.busy_since = Some(now);
        if let Some(r) = s.row(id) {
            r.state = ReqState::Prefill;
            r.started = Some(now);
        }
    }

    /// Prefill finished: `cached` prompt tokens were reused, `computed` were processed.
    pub fn prefilled(&self, id: u64, cached: usize, computed: usize) {
        let now = Instant::now();
        let mut s = self.lock();
        s.acc_prompt += computed as u64;
        s.totals.prompt_tokens += (cached + computed) as u64;
        s.totals.cached_tokens += cached as u64;
        if let Some(r) = s.row(id) {
            r.state = ReqState::Generating;
            r.cached = cached;
            r.prompt = cached + computed;
            r.prefill_ms = r.started.map(|t| (now - t).as_secs_f64() * 1e3);
            r.gen_started = Some(now);
        }
    }

    /// One generated slot (a text token, or a speech frame when `audio`).
    pub fn slot(&self, id: u64, audio: bool) {
        let now = Instant::now();
        let mut s = self.lock();
        s.acc_gen += 1;
        s.totals.generated += 1;
        if audio {
            s.acc_audio += 1;
            s.totals.audio_frames += 1;
        }
        let mut ttft = None;
        if let Some(r) = s.row(id) {
            r.generated += 1;
            if audio {
                r.audio_frames += 1;
            }
            if r.ttft_ms.is_none() {
                let t = (now - r.queued_at).as_secs_f64() * 1e3;
                r.ttft_ms = Some(t);
                ttft = Some(t);
            }
            if let Some(g) = r.gen_started {
                let dt = (now - g).as_secs_f64();
                if dt > 0.0 {
                    r.decode_tps = Some(r.generated as f64 / dt);
                }
            }
        }
        if let Some(t) = ttft {
            s.ttft_acc.push(t);
        }
    }

    /// The request finished normally with `finish` ("stop", "length", ...).
    pub fn finished(&self, id: u64, finish: &str, usage: &Usage) {
        let now = Instant::now();
        let mut s = self.lock();
        s.busy_end(now);
        s.totals.completed += 1;
        if let Some(r) = s.row(id) {
            r.state = ReqState::Done(finish.to_string());
            r.generated = usage.completion_tokens;
            r.audio_frames = usage.audio_frames;
            r.total_ms = Some((now - r.queued_at).as_secs_f64() * 1e3);
            if let Some(g) = r.gen_started {
                let dt = (now - g).as_secs_f64();
                if dt > 0.0 && r.generated > 0 {
                    r.decode_tps = Some(r.generated as f64 / dt);
                }
            }
        }
    }

    pub fn failed(&self, id: u64, err: String) {
        let now = Instant::now();
        let mut s = self.lock();
        s.busy_end(now);
        s.totals.failed += 1;
        if let Some(r) = s.row(id) {
            r.state = ReqState::Failed(err);
            r.total_ms = Some((now - r.queued_at).as_secs_f64() * 1e3);
        }
    }

    /// The engine is done with the request; marks it cancelled if it did not finish.
    pub fn close(&self, id: u64) {
        let now = Instant::now();
        let mut s = self.lock();
        s.busy_end(now);
        let mut cancelled = false;
        if let Some(r) = s.row(id)
            && r.active()
        {
            r.state = ReqState::Done("cancelled".into());
            r.total_ms = Some((now - r.queued_at).as_secs_f64() * 1e3);
            cancelled = true;
        }
        if cancelled {
            s.totals.cancelled += 1;
        }
    }

    /// Close the current one-second window with external readings.
    fn tick(&self, gpu_util: Option<f64>, vram: Option<(u64, u64)>, rss: u64) {
        let now = Instant::now();
        let mut s = self.lock();
        let dt = (now - s.last_tick).as_secs_f64().max(1e-3);
        s.last_tick = now;
        let mut busy = s.busy_accum;
        if let Some(b) = s.busy_since {
            busy += now - b;
            s.busy_since = Some(now);
        }
        s.busy_accum = Duration::ZERO;
        let ttft_ms = (!s.ttft_acc.is_empty())
            .then(|| s.ttft_acc.iter().sum::<f64>() / s.ttft_acc.len() as f64);
        s.ttft_acc.clear();
        let ex = (EXPERT_HITS.load(Ordering::Relaxed), EXPERT_MISSES.load(Ordering::Relaxed), EXPERT_UPLOAD_BYTES.load(Ordering::Relaxed));
        let (dh, dm, db) = (ex.0.saturating_sub(s.last_expert.0), ex.1.saturating_sub(s.last_expert.1), ex.2.saturating_sub(s.last_expert.2));
        s.last_expert = ex;
        let expert_hit = (dh + dm > 0).then(|| dh as f64 / (dh + dm) as f64);
        let upload_bps = db as f64 / dt;
        let sample = Sample {
            expert_hit,
            upload_bps,
            gen_tps: s.acc_gen as f64 / dt,
            audio_fps: s.acc_audio as f64 / dt,
            prompt_tps: s.acc_prompt as f64 / dt,
            ttft_ms,
            busy: (busy.as_secs_f64() / dt).min(1.0),
            gpu_util,
            vram,
            gpu_alloc: GPU_BYTES.load(Ordering::Relaxed).max(0) as u64,
            rss,
        };
        s.acc_gen = 0;
        s.acc_audio = 0;
        s.acc_prompt = 0;
        s.samples.push_back(sample);
        while s.samples.len() > MAX_SAMPLES {
            s.samples.pop_front();
        }
    }

    pub fn snapshot(&self, max_samples: usize, max_logs: usize) -> Snapshot {
        let s = self.lock();
        let ns = s.samples.len();
        let nl = s.logs.len();
        Snapshot {
            info: s.info.clone(),
            uptime: s.started.elapsed(),
            samples: s
                .samples
                .iter()
                .skip(ns.saturating_sub(max_samples))
                .cloned()
                .collect(),
            requests: s.requests.iter().cloned().collect(),
            totals: s.totals.clone(),
            logs: s
                .logs
                .iter()
                .skip(nl.saturating_sub(max_logs))
                .cloned()
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GPU and memory sampling

trait GpuProbe: Send {
    /// (utilisation %, (vram used, vram total))
    fn read(&mut self) -> (Option<f64>, Option<(u64, u64)>);
}

struct NvmlProbe {
    nvml: nvml_wrapper::Nvml,
    index: u32,
}

impl GpuProbe for NvmlProbe {
    fn read(&mut self) -> (Option<f64>, Option<(u64, u64)>) {
        let Ok(d) = self.nvml.device_by_index(self.index) else {
            return (None, None);
        };
        let util = d.utilization_rates().ok().map(|u| u.gpu as f64);
        let mem = d.memory_info().ok().map(|m| (m.used, m.total));
        (util, mem)
    }
}

/// Find the NVML device whose name matches the wgpu adapter name.
fn nvml_probe(adapter: &str) -> Option<NvmlProbe> {
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let n = nvml.device_count().ok()?;
    let want = adapter.to_ascii_lowercase();
    let mut pick = None;
    for i in 0..n {
        let name = nvml
            .device_by_index(i)
            .ok()?
            .name()
            .ok()?
            .to_ascii_lowercase();
        if !want.is_empty() && (want.contains(&name) || name.contains(&want)) {
            pick = Some(i);
            break;
        }
    }
    let index = pick?;
    Some(NvmlProbe { nvml, index })
}

#[cfg(target_os = "linux")]
struct SysfsProbe {
    dir: std::path::PathBuf,
}

#[cfg(target_os = "linux")]
impl GpuProbe for SysfsProbe {
    fn read(&mut self) -> (Option<f64>, Option<(u64, u64)>) {
        let num = |f: &str| -> Option<u64> {
            std::fs::read_to_string(self.dir.join(f))
                .ok()?
                .trim()
                .parse()
                .ok()
        };
        let util = num("gpu_busy_percent").map(|v| v as f64);
        let mem = num("mem_info_vram_used").zip(num("mem_info_vram_total"));
        (util, mem)
    }
}

/// AMD GPUs expose utilisation and VRAM in sysfs; take the first card that has them.
#[cfg(target_os = "linux")]
fn sysfs_probe() -> Option<SysfsProbe> {
    let mut cards: Vec<_> = std::fs::read_dir("/sys/class/drm")
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("device"))
        .filter(|p| p.join("gpu_busy_percent").exists())
        .collect();
    cards.sort();
    cards.dedup();
    cards.into_iter().next().map(|dir| SysfsProbe { dir })
}

fn find_probe(adapter: &str) -> Option<(Box<dyn GpuProbe>, &'static str)> {
    if let Some(p) = nvml_probe(adapter) {
        return Some((Box::new(p), "NVML"));
    }
    #[cfg(target_os = "linux")]
    if let Some(p) = sysfs_probe() {
        return Some((Box::new(p), "sysfs"));
    }
    None
}

/// Start the once-a-second sampler for the adapter named `adapter`.
pub fn spawn_sampler(adapter: String) {
    std::thread::Builder::new()
        .name("lfm3-metrics".into())
        .spawn(move || {
            let mut probe = find_probe(&adapter);
            metrics().set_info(|i| i.gpu_source = probe.as_ref().map(|p| p.1.to_string()));
            let pid = sysinfo::get_current_pid().ok();
            let mut sys = sysinfo::System::new().ok();
            loop {
                std::thread::sleep(Duration::from_secs(1));
                let (util, vram) = match probe.as_mut() {
                    Some((p, _)) => p.read(),
                    None => (None, None),
                };
                let rss = match (sys.as_mut(), pid) {
                    (Some(sys), Some(pid)) => {
                        sys.refresh_processes_specifics(
                            sysinfo::ProcessesToUpdate::Some(&[pid]),
                            false,
                            sysinfo::ProcessRefreshKind::nothing().with_memory(),
                        );
                        sys.process(pid).map(|p| p.memory()).unwrap_or(0)
                    }
                    _ => 0,
                };
                metrics().tick(util, vram, rss);
            }
        })
        .expect("spawning metrics thread");
}

// ---------------------------------------------------------------------------------------------
// Log capture

/// A `tracing` writer that appends formatted log lines to the dashboard's log panel.
pub struct LogWriter(Vec<u8>);

impl LogWriter {
    pub fn new() -> Self {
        LogWriter(Vec::new())
    }
}

impl std::io::Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for LogWriter {
    fn drop(&mut self) {
        let text = String::from_utf8_lossy(&self.0);
        for line in text.lines() {
            if !line.trim().is_empty() {
                metrics().log(line.to_string());
            }
        }
    }
}

/// (used, total) VRAM bytes of the adapter, when NVML or sysfs can tell.
pub fn vram_info(adapter: &wgpu::AdapterInfo) -> Option<(u64, u64)> {
    let (mut p, _) = find_probe(&adapter.name)?;
    p.read().1
}

pub fn vram_total(adapter: &wgpu::AdapterInfo) -> Option<u64> {
    vram_info(adapter).map(|x| x.1)
}

pub fn vram_used(adapter: &wgpu::AdapterInfo) -> Option<u64> {
    vram_info(adapter).map(|x| x.0)
}

fn state_str(s: &ReqState) -> (String, String) {
    match s {
        ReqState::Queued => ("queued".into(), String::new()),
        ReqState::Prefill => ("prefill".into(), String::new()),
        ReqState::Generating => ("generating".into(), String::new()),
        ReqState::Done(f) => ("done".into(), f.clone()),
        ReqState::Failed(e) => ("failed".into(), e.clone()),
        ReqState::Rejected(c, e) => ("rejected".into(), format!("{c}: {e}")),
    }
}

/// The dashboard snapshot as JSON for the browser dashboard.
pub fn snapshot_json(max_samples: usize) -> serde_json::Value {
    use serde_json::json;
    let s = metrics().snapshot(max_samples, 200);
    let samples: Vec<_> = s
        .samples
        .iter()
        .map(|x| {
            json!({
                "gen_tps": x.gen_tps, "prompt_tps": x.prompt_tps, "ttft_ms": x.ttft_ms, "busy": x.busy,
                "gpu_util": x.gpu_util, "vram_used": x.vram.map(|v| v.0), "vram_total": x.vram.map(|v| v.1),
                "gpu_alloc": x.gpu_alloc, "rss": x.rss, "expert_hit": x.expert_hit, "upload_bps": x.upload_bps,
            })
        })
        .collect();
    let requests: Vec<_> = s
        .requests
        .iter()
        .rev()
        .take(100)
        .map(|r| {
            let (state, detail) = state_str(&r.state);
            json!({
                "id": r.id, "endpoint": r.endpoint, "input": r.input, "stream": r.stream, "state": state,
                "detail": detail, "at": r.at, "prompt": r.prompt, "cached": r.cached, "generated": r.generated,
                "prefill_ms": r.prefill_ms, "ttft_ms": r.ttft_ms, "decode_tps": r.decode_tps, "total_ms": r.total_ms,
                "progress": r.progress,
            })
        })
        .collect();
    let t = &s.totals;
    json!({
        "info": {"model": s.info.model, "gpu": s.info.gpu, "addr": s.info.addr, "status": s.info.status, "gpu_source": s.info.gpu_source},
        "uptime_s": s.uptime.as_secs_f64(),
        "samples": samples,
        "requests": requests,
        "totals": {"requests": t.requests, "completed": t.completed, "failed": t.failed, "rejected": t.rejected,
                   "cancelled": t.cancelled, "prompt_tokens": t.prompt_tokens, "cached_tokens": t.cached_tokens, "generated": t.generated},
        "experts": {"hits": EXPERT_HITS.load(Ordering::Relaxed), "misses": EXPERT_MISSES.load(Ordering::Relaxed),
                    "uploaded_bytes": EXPERT_UPLOAD_BYTES.load(Ordering::Relaxed)},
        "logs": s.logs,
    })
}
