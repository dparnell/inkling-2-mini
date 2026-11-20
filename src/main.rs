mod cli_chat;
mod config;
mod download;
mod engine;
mod experts;
mod gpu;
mod metrics;
mod model;
mod reference;
mod sampler;
mod server;
mod text;
mod tui;
mod vision;
mod weights;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "inkling", about = "Inkling-2-Mini inference engine (Rust + wgpu)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct ModelArgs {
    /// Skip loading the vision tower (text only, saves ~0.9 GB VRAM).
    #[arg(long)]
    no_vision: bool,
    /// Largest image in LM tokens (40x40 px each); bigger images are downscaled.
    #[arg(long, default_value_t = 1024)]
    max_image_tokens: usize,
    /// Directory with config.json, tokenizer and safetensors.
    #[arg(long, default_value = "models/Inkling-2-Mini", env = "INKLING_MODEL_DIR")]
    model_dir: PathBuf,
    /// Maximum context length (tokens).
    #[arg(long, default_value_t = 32768)]
    ctx: usize,
    /// Prefill chunk size (tokens per forward pass).
    #[arg(long, default_value_t = 512)]
    chunk: usize,
    /// VRAM for the expert cache in GB (default: as much as fits).
    #[arg(long)]
    expert_cache_gb: Option<f64>,
    /// GPU adapter index (see `inkling devices`).
    #[arg(long)]
    adapter: Option<usize>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download model weights from Hugging Face.
    Download {
        #[arg(long, default_value = download::DEFAULT_REPO)]
        repo: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// Also fetch the MTP (speculative decoding) head.
        #[arg(long)]
        mtp: bool,
    },
    /// Render the chat template for a JSON file {messages, tools?, reasoning_effort?} (debug).
    Template {
        #[arg(long, default_value = "models/Inkling-2-Mini")]
        model_dir: PathBuf,
        messages: PathBuf,
    },
    /// Compute next-token logits for a raw prompt on the GPU (validation).
    Logits {
        #[command(flatten)]
        m: ModelArgs,
        /// Prompt text (special tokens allowed).
        #[arg(long)]
        prompt: String,
        /// Write logits (f32 LE) here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Prefill the prompt in chunks of this many tokens (validation of chunking).
        #[arg(long)]
        step: Option<usize>,
        /// Dump each layer application's hidden state into this directory.
        #[arg(long)]
        dump: Option<PathBuf>,
        /// Keep only the first N prompt tokens.
        #[arg(long)]
        truncate: Option<usize>,
    },
    /// Same as `logits` on the CPU reference implementation (slow).
    Reference {
        #[arg(long, default_value = "models/Inkling-2-Mini")]
        model_dir: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        dump: Option<PathBuf>,
    },
    /// Compare two logit files.
    Compare { a: PathBuf, b: PathBuf },
    /// List GPU adapters.
    Devices,
    /// Serve the OpenAI-compatible API, the web chat UI and the browser dashboard.
    Serve {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Name reported in /v1/models.
        #[arg(long, default_value = "inkling-2-mini")]
        served_model_name: String,
        /// Require `Authorization: Bearer <key>`.
        #[arg(long, env = "INKLING_API_KEY")]
        api_key: Option<String>,
        /// Allow http(s) image URLs (the server fetches them).
        #[arg(long)]
        allow_remote_media: bool,
        /// Show the live terminal performance dashboard.
        #[arg(long)]
        tui: bool,
        /// Do not serve the web chat UI and dashboard.
        #[arg(long)]
        no_ui: bool,
        /// Default reasoning effort when a request sets none (none|minimal|low|medium|high|xhigh).
        #[arg(long)]
        reasoning_effort: Option<String>,
        /// Download the weights first if the model directory is missing or incomplete.
        #[arg(long)]
        download: bool,
    },
    /// Interactive chat in the terminal.
    Chat {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(long, default_value = "low")]
        reasoning_effort: String,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long, default_value_t = 2048)]
        max_tokens: usize,
    },
    /// Measure prefill and decode throughput.
    Bench {
        #[command(flatten)]
        m: ModelArgs,
        /// Prompt lengths to test.
        #[arg(long, value_delimiter = ',', default_value = "128,1024")]
        prompt: Vec<usize>,
        /// Tokens to decode after each prompt.
        #[arg(long = "gen", default_value_t = 64)]
        n_gen: usize,
    },
}

fn write_f32(p: &Path, v: &[f32]) -> Result<()> {
    std::fs::write(p, bytemuck::cast_slice(v)).with_context(|| format!("writing {}", p.display()))
}

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(bytemuck::cast_slice(&b).to_vec())
}

fn top_k(l: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..l.len()).collect();
    idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
    idx.into_iter().take(k).map(|i| (i, l[i])).collect()
}

fn show_top(tp: &text::TextProcessor, l: &[f32]) {
    for (i, v) in top_k(l, 8) {
        println!("  {i:>7} {v:>9.4} {:?}", tp.decode(&[i as u32]));
    }
}

fn compare(a: &[f32], b: &[f32]) -> String {
    let n = a.len().min(b.len());
    let mut max = 0f32;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        max = max.max((a[i] - b[i]).abs());
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    format!("max_abs {max:.5} cos {:.7}", dot / (na.sqrt() * nb.sqrt()).max(1e-30))
}

fn logit_report(a: &[f32], b: &[f32]) -> String {
    let ta: Vec<usize> = top_k(a, 5).into_iter().map(|x| x.0).collect();
    let tb: Vec<usize> = top_k(b, 5).into_iter().map(|x| x.0).collect();
    let lse = |x: &[f32]| {
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(a), lse(b));
    let kl: f64 = a.iter().zip(b).map(|(&x, &y)| {
        let pa = (x as f64 - la).exp();
        pa * ((x as f64 - la) - (y as f64 - lb))
    }).sum();
    format!("{} | top5 {} | argmax {} | KL {kl:.2e}", compare(a, b), if ta == tb { "same" } else { "DIFFERENT" }, if ta[0] == tb[0] { "same" } else { "DIFFERENT" })
}

pub struct Loaded {
    pub gpu: gpu::Gpu,
    pub model: model::Model,
    pub tp: Arc<text::TextProcessor>,
    pub cfg: config::ModelConfig,
}

pub(crate) fn load_model(m: &ModelArgs) -> Result<Loaded> {
    let cfg = config::ModelConfig::load(&m.model_dir)?;
    let tp = Arc::new(text::TextProcessor::load(&m.model_dir, cfg.text_config.unpadded_vocab_size)?);
    let st = Arc::new(weights::SafeTensors::open(&m.model_dir)?);
    let gpu = gpu::Gpu::new(m.adapter)?;
    tracing::info!("GPU: {} ({:?})", gpu.info.name, gpu.info.backend);
    let budget = match m.expert_cache_gb {
        Some(g) => (g * 1e9) as u64,
        None => auto_expert_budget(&gpu, &cfg.text_config, m),
    };
    let opts = model::ModelOptions { ctx: m.ctx, chunk: m.chunk, expert_budget: budget, rewind_slack: 2048 };
    let model = model::Model::load(&gpu, st, &cfg.text_config, &opts)?;
    Ok(Loaded { gpu, model, tp, cfg })
}

/// VRAM left for experts after dense weights, caches and activations (NVML when available).
fn auto_expert_budget(gpu: &gpu::Gpu, c: &config::TextConfig, m: &ModelArgs) -> u64 {
    let total = metrics::vram_total(&gpu.info).unwrap_or(8 << 30);
    let used_other = metrics::vram_used(&gpu.info).unwrap_or(0);
    let dense = 1_400_000_000u64 + (c.unpadded_vocab_size * c.hidden_size * 2) as u64;
    let n_apps = model::Model::app_count(c) as u64;
    let full_apps = 8u64;
    let kv = full_apps * m.ctx as u64 * 4096 + (n_apps - full_apps) * (c.sliding_window_size + m.chunk + 2048) as u64 * 4096;
    let acts = m.chunk as u64 * 2304 * 4 * 60 + m.chunk as u64 * 8 * (2304 * 2 + 1152) * 4;
    let vision = if m.no_vision { 0 } else { 900_000_000u64 + m.max_image_tokens as u64 * 4 * 4 * (4096 + 9 * 1024) };
    let reserve = 1_500_000_000u64 + vision;
    let free = total.saturating_sub(used_other).saturating_sub(dense + kv + acts + reserve);
    tracing::info!("VRAM {:.1} GB total, {:.1} GB used elsewhere; expert cache budget {:.1} GB", total as f64 / 1e9, used_other as f64 / 1e9, free as f64 / 1e9);
    free.max(1_200_000_000)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let filter = || tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,wgpu_core=warn,wgpu_hal=warn,naga=warn".into());
    if matches!(cli.cmd, Cmd::Serve { tui: true, .. }) {
        // logs go to the dashboard's log panel instead of the terminal
        tracing_subscriber::fmt()
            .with_env_filter(filter())
            .with_ansi(false)
            .with_target(false)
            .with_timer(tracing_subscriber::fmt::time::Uptime::default())
            .with_writer(metrics::LogWriter::new)
            .init();
        let dash = tui::Tui::start().context("starting the terminal dashboard")?;
        let res = run(cli);
        dash.stop();
        res
    } else {
        // terminal, plus a copy for the browser dashboard's log panel
        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry()
            .with(filter())
            .with(tracing_subscriber::fmt::layer())
            .with(tracing_subscriber::fmt::layer().with_ansi(false).with_target(false).with_timer(tracing_subscriber::fmt::time::Uptime::default()).with_writer(metrics::LogWriter::new))
            .init();
        run(cli)
    }
}

fn ensure_model(m: &ModelArgs, download_flag: bool) -> Result<()> {
    let ok = m.model_dir.join("config.json").exists() && m.model_dir.join("model.safetensors.index.json").exists();
    if ok && (download::is_complete(&m.model_dir) || !download_flag) {
        return Ok(());
    }
    if !download_flag {
        anyhow::bail!(
            "no model found in {}.\nRun `inkling download` or pass --download to fetch {} from Hugging Face",
            m.model_dir.display(),
            download::DEFAULT_REPO
        );
    }
    tokio::runtime::Runtime::new()?.block_on(download::download(download::DEFAULT_REPO, "main", &m.model_dir, false))
}

fn load_vision(m: &ModelArgs, l: &Loaded) -> Result<Option<vision::Vision>> {
    if m.no_vision || l.cfg.vision_config.is_none() {
        return Ok(None);
    }
    let st = weights::SafeTensors::open(&m.model_dir)?;
    Ok(Some(vision::Vision::load(&l.gpu, &st, m.max_image_tokens)?))
}

fn run(cli: Cli) -> Result<()> {
    match cli.cmd {
        Cmd::Download { repo, revision, model_dir, mtp } => {
            let dir = model_dir.unwrap_or_else(|| download::default_dir(&repo));
            tokio::runtime::Runtime::new()?.block_on(download::download(&repo, &revision, &dir, mtp))
        }
        Cmd::Template { model_dir, messages } => {
            let cfg = config::ModelConfig::load(&model_dir)?;
            let tp = text::TextProcessor::load(&model_dir, cfg.text_config.unpadded_vocab_size)?;
            let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(messages)?)?;
            let s = tp.render(&j["messages"], j.get("tools"), j.get("reasoning_effort"))?;
            println!("{s}");
            println!("{:?}", tp.encode(&s)?);
            Ok(())
        }
        Cmd::Logits { m, prompt, out, step, dump, truncate } => {
            let mut l = load_model(&m)?;
            let mut ids = l.tp.encode(&prompt)?;
            if let Some(n) = truncate {
                ids.truncate(n);
            }
            println!("{} tokens", ids.len());
            if dump.is_some() {
                l.model.dump = Some(Vec::new());
            }
            let step = step.unwrap_or(m.chunk).min(m.chunk);
            let t0 = std::time::Instant::now();
            let mut logits = None;
            let mut pos = 0;
            while pos < ids.len() {
                let n = step.min(ids.len() - pos);
                let rows = l.model.embed_rows(&ids[pos..pos + n])?;
                logits = l.model.forward(&mut l.gpu, &rows, &[], pos, pos + n == ids.len())?;
                pos += n;
            }
            let dt = t0.elapsed().as_secs_f64();
            let (h, mi, b) = l.model.experts.stats();
            println!("forward {:.2} s ({:.1} tok/s); route syncs {} ({:.2} s); host expert copies {:.2} s; experts hit {h} miss {mi} uploaded {:.2} GB", dt, ids.len() as f64 / dt, l.model.timings.route_syncs, l.model.timings.route_wait_s, l.model.timings.ensure_s, b as f64 / 1e9);
            let logits = logits.unwrap();
            show_top(&l.tp, &logits);
            if let Some(o) = out {
                write_f32(&o, &logits)?;
            }
            if let (Some(dir), Some(d)) = (dump, &l.model.dump) {
                std::fs::create_dir_all(&dir)?;
                for (i, v) in d.iter().enumerate() {
                    write_f32(&dir.join(format!("app{i:02}.bin")), v)?;
                }
            }
            Ok(())
        }
        Cmd::Reference { model_dir, prompt, out, dump } => {
            let cfg = config::ModelConfig::load(&model_dir)?;
            let tp = text::TextProcessor::load(&model_dir, cfg.text_config.unpadded_vocab_size)?;
            let st = weights::SafeTensors::open(&model_dir)?;
            let ids = tp.encode(&prompt)?;
            println!("{} tokens", ids.len());
            let mut r = reference::Reference::new(&st, &cfg.text_config);
            let t0 = std::time::Instant::now();
            let logits = r.forward(&ids)?;
            println!("reference forward {:.1} s", t0.elapsed().as_secs_f64());
            show_top(&tp, &logits);
            if let Some(o) = out {
                write_f32(&o, &logits)?;
            }
            if let Some(dir) = dump {
                std::fs::create_dir_all(&dir)?;
                for (i, v) in r.dump.iter().enumerate() {
                    write_f32(&dir.join(format!("app{i:02}.bin")), v)?;
                }
            }
            Ok(())
        }
        Cmd::Compare { a, b } => {
            let (x, y) = (read_f32(&a)?, read_f32(&b)?);
            if a.is_dir() {
                unreachable!()
            }
            println!("{}", logit_report(&x, &y));
            Ok(())
        }
        Cmd::Serve { m, host, port, served_model_name, api_key, allow_remote_media, tui: _, no_ui, reasoning_effort, download } => {
            ensure_model(&m, download)?;
            metrics::metrics().set_info(|i| {
                i.model = served_model_name.clone();
                i.status = "loading".into();
            });
            let l = load_model(&m)?;
            metrics::metrics().set_info(|i| i.gpu = format!("{} ({:?})", l.gpu.info.name, l.gpu.info.backend));
            metrics::spawn_sampler(l.gpu.info.name.clone());
            let vision = load_vision(&m, &l)?;
            let engine = engine::spawn(l.gpu, l.model, vision, l.tp);
            server::run(
                engine,
                server::ServerOptions {
                    model_name: served_model_name,
                    host,
                    port,
                    api_key,
                    allow_remote_media,
                    ui: !no_ui,
                    max_image_tokens: m.max_image_tokens,
                    default_reasoning_effort: reasoning_effort,
                },
            )
        }
        Cmd::Chat { m, reasoning_effort, temperature, max_tokens } => {
            ensure_model(&m, false)?;
            let l = load_model(&m)?;
            let tp = l.tp.clone();
            let engine = engine::spawn(l.gpu, l.model, None, l.tp);
            cli_chat::repl(engine, tp, &reasoning_effort, temperature, max_tokens)
        }
        Cmd::Bench { m, prompt, n_gen } => {
            ensure_model(&m, false)?;
            let mut l = load_model(&m)?;
            cli_chat::bench(&mut l, &prompt, n_gen)
        }
        Cmd::Devices => {
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
            desc.backends = wgpu::Backends::all();
            let inst = wgpu::Instance::new(desc);
            for (i, a) in pollster::block_on(inst.enumerate_adapters(wgpu::Backends::all())).iter().enumerate() {
                let info = a.get_info();
                println!("{i}: {} ({:?}, {:?})", info.name, info.backend, info.device_type);
            }
            Ok(())
        }
    }
}
