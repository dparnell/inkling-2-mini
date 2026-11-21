//! Generation engine: owns the GPU and the model on a dedicated thread and serves requests one at
//! a time.
//!
//! The state of the previous request is kept: a prompt that extends the previous sequence only
//! prefills the new suffix; one that shares a prefix up to the previous *prompt* rewinds to a
//! snapshot taken right before that prompt's last token. Image rows are keyed by a content hash so
//! a different image under the same placeholder tokens never reuses stale state.

use crate::metrics::metrics;
use crate::model::Model;
use crate::sampler::{Sampler, SamplingParams};
use crate::text::{ParseEvent, ResponseParser, TextProcessor, tok};
use crate::vision::{ImageInput, Vision};
use anyhow::{Context, bail};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub reasoning_tokens: usize,
    /// kept for the dashboard's shared metrics code
    pub audio_frames: usize,
}

#[derive(Debug)]
pub enum GenEvent {
    Reasoning(String),
    Content(String),
    ToolCall { name: String, arguments: String },
    /// Running token counts: once after prefill, then after every generated token.
    Progress { prompt_tokens: usize, cached_tokens: usize, completion_tokens: usize },
    Done { finish: FinishReason, usage: Usage },
    Error(String),
}

pub struct GenRequest {
    pub id: u64,
    /// Prompt tokens with every image placeholder expanded to its full run.
    pub prompt: Vec<u32>,
    /// One per image placeholder run, in prompt order.
    pub images: Vec<ImageInput>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop: Vec<String>,
    /// Raw completion: stream every token as content (no chat structure parsing).
    pub raw: bool,
    pub events: UnboundedSender<GenEvent>,
}

#[derive(Clone)]
pub struct Capabilities {
    pub vision: bool,
}

#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<GenRequest>,
    pub tp: Arc<TextProcessor>,
    pub max_ctx: usize,
    pub caps: Capabilities,
}

impl EngineHandle {
    pub fn submit(&self, req: GenRequest) -> anyhow::Result<()> {
        self.tx.send(req).map_err(|_| anyhow::anyhow!("engine thread has stopped"))
    }
}

pub fn spawn(gpu: crate::gpu::Gpu, model: Model, vision: Option<Vision>, tp: Arc<TextProcessor>) -> EngineHandle {
    let (tx, rx) = mpsc::channel::<GenRequest>();
    let max_ctx = model.ctx;
    let caps = Capabilities { vision: vision.is_some() };
    let tpc = tp.clone();
    std::thread::Builder::new()
        .name("inkling-engine".into())
        .spawn(move || {
            let mut engine = Engine { gpu, model, vision, tp: tpc, history: Vec::new() };
            while let Ok(req) = rx.recv() {
                let events = req.events.clone();
                let id = req.id;
                metrics().started(id);
                let res = engine.run(req);
                if let Err(e) = &res {
                    metrics().failed(id, format!("{e:#}"));
                }
                metrics().close(id);
                engine.publish_stats();
                if let Err(e) = res {
                    tracing::error!("generation failed: {e:#}");
                    engine.model.reset(&engine.gpu);
                    engine.history.clear();
                    let _ = events.send(GenEvent::Error(format!("{e:#}")));
                }
            }
        })
        .expect("spawning engine thread");
    EngineHandle { tx, tp, max_ctx, caps }
}

struct Engine {
    gpu: crate::gpu::Gpu,
    model: Model,
    vision: Option<Vision>,
    tp: Arc<TextProcessor>,
    /// Keys of the rows the model state holds: token ids, or tagged hashes for image rows.
    history: Vec<u64>,
}

const MEDIA_KEY: u64 = 1 << 63;

/// Tokens the model must never emit (unused / placeholder specials).
fn suppressed() -> Vec<u32> {
    let mut v = vec![199998, tok::IMAGE_PLACEHOLDER, tok::BEGIN_OF_TEXT, tok::CONTENT_IMAGE, 200007, 200009, 200023];
    v.extend(200011..=200021);
    v.extend(200025..=200027);
    v.extend(200029..=200048);
    v.extend(200050..=200056);
    v
}

impl Engine {
    fn publish_stats(&self) {
        let (h, m, b) = self.model.experts.stats();
        crate::metrics::EXPERT_HITS.store(h, std::sync::atomic::Ordering::Relaxed);
        crate::metrics::EXPERT_MISSES.store(m, std::sync::atomic::Ordering::Relaxed);
        crate::metrics::EXPERT_UPLOAD_BYTES.store(b, std::sync::atomic::Ordering::Relaxed);
    }

    /// Bring the model state to `keys[..n]` for the largest reusable `n < keys.len()`; returns `n`.
    fn reuse_prefix(&mut self, keys: &[u64]) -> usize {
        let h = self.history.len();
        if h > 0 && h < keys.len() && keys[..h] == self.history[..] {
            return h;
        }
        if let Some(sp) = self.model.snap_pos {
            if sp > 0 && sp < keys.len() && sp <= h && keys[..sp] == self.history[..sp] {
                if let Some(sp) = self.model.restore(&self.gpu, h) {
                    self.history.truncate(sp);
                    return sp;
                }
            }
        }
        self.model.reset(&self.gpu);
        self.history.clear();
        0
    }

    fn forward_rows(&mut self, ids: &[u32], keys_start: usize, feats: &[(usize, Vec<f32>)], want_logits: bool) -> anyhow::Result<Option<Vec<f32>>> {
        let rows = self.model.embed_rows(ids)?;
        let inject: Vec<(usize, Vec<f32>)> = feats.iter().filter(|(r, _)| *r >= keys_start && *r < keys_start + ids.len()).map(|(r, f)| (r - keys_start, f.clone())).collect();
        self.model.forward(&mut self.gpu, &rows, &inject, keys_start, want_logits)
    }

    fn run(&mut self, req: GenRequest) -> anyhow::Result<()> {
        let prompt = &req.prompt;
        if prompt.is_empty() {
            bail!("empty prompt");
        }
        if prompt.len() >= self.model.ctx {
            bail!("prompt has {} tokens, which exceeds the context window of {}", prompt.len(), self.model.ctx);
        }
        let t0 = Instant::now();

        // ---- key rows; map image placeholder runs to images
        let mut keys: Vec<u64> = prompt.iter().map(|&t| t as u64).collect();
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        let mut i = 0;
        let mut next = 0;
        while i < prompt.len() {
            if prompt[i] == tok::IMAGE_PLACEHOLDER {
                let img = req.images.get(next).context("more image placeholders than images")?;
                let n = img.n_tokens();
                if i + n > prompt.len() || prompt[i..i + n].iter().any(|&x| x != tok::IMAGE_PLACEHOLDER) {
                    bail!("image placeholder run {next} does not have {n} tokens");
                }
                let h = img.content_hash();
                for j in 0..n {
                    keys[i + j] = MEDIA_KEY | ((h << 20) ^ j as u64);
                }
                runs.push((i, n, next));
                next += 1;
                i += n;
            } else {
                i += 1;
            }
        }
        if next != req.images.len() {
            bail!("{} images but {next} placeholder runs", req.images.len());
        }

        let cached = self.reuse_prefix(&keys);

        // ---- encode images overlapping the uncached suffix
        let d = self.model.cfg.hidden_size;
        let mut feats: Vec<(usize, Vec<f32>)> = Vec::new();
        for &(start, n, mi) in &runs {
            if start + n <= cached {
                continue;
            }
            let te = Instant::now();
            let v = self.vision.as_mut().context("this model was loaded without vision support")?;
            let f = v.encode(&mut self.gpu, &req.images[mi])?;
            if f.len() != n * d {
                bail!("vision encoder produced {} rows for a run of {n}", f.len() / d);
            }
            tracing::info!("encoded image {mi} ({n} tokens) in {:.0} ms", te.elapsed().as_secs_f64() * 1e3);
            for j in 0..n {
                feats.push((start + j, f[j * d..(j + 1) * d].to_vec()));
            }
        }

        // ---- prefill the uncached suffix in chunks; snapshot one row before the end
        let tm = self.model.tm;
        let end = prompt.len() - 1;
        let mut pos = cached;
        while pos < end {
            let n = tm.min(end - pos);
            self.forward_rows(&prompt[pos..pos + n].to_vec(), pos, &feats, false)?;
            self.history.extend_from_slice(&keys[pos..pos + n]);
            pos += n;
            metrics().progress(req.id, pos - cached, prompt.len() - cached);
        }
        self.model.snapshot(&self.gpu, pos);
        let logits = self.forward_rows(&prompt[end..], end, &feats, true)?.context("no logits")?;
        self.history.push(keys[end]);
        let prefill_s = t0.elapsed().as_secs_f64();
        metrics().prefilled(req.id, cached, prompt.len() - cached);
        tracing::info!(
            "prefill {} tokens ({} cached) in {:.2} s ({:.0} tok/s)",
            prompt.len() - cached,
            cached,
            prefill_s,
            (prompt.len() - cached) as f64 / prefill_s.max(1e-9)
        );
        self.publish_stats();
        self.generate(&req, logits, cached, prefill_s)
    }

    fn generate(&mut self, req: &GenRequest, mut logits: Vec<f32>, cached: usize, prefill_s: f64) -> anyhow::Result<()> {
        let mut sampler = Sampler::new(req.sampling.clone());
        for &t in &req.prompt {
            sampler.observe(t);
        }
        let suppress = suppressed();
        let mut usage = Usage { prompt_tokens: req.prompt.len(), cached_tokens: cached, ..Default::default() };
        let mut parser = ResponseParser::new();
        let mut content = String::new();
        let mut emitted = 0usize;
        let max_stop = req.stop.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut had_tool = false;
        let t1 = Instant::now();
        let mut n = 0usize;
        let mut finish = FinishReason::Length;
        let mut raw_text_ids: Vec<u32> = Vec::new();
        let mut raw_emitted = 0usize;
        let progress = |n: usize| GenEvent::Progress { prompt_tokens: req.prompt.len(), cached_tokens: cached, completion_tokens: n };
        if req.events.send(progress(0)).is_err() {
            metrics().cancelled(req.id);
            return Ok(());
        }
        loop {
            for &s in &suppress {
                if let Some(l) = logits.get_mut(s as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
            let id = sampler.sample(&mut logits);
            sampler.observe(id);
            n += 1;
            usage.completion_tokens = n;
            metrics().slot(req.id, false);
            let _ = req.events.send(progress(n));

            let mut stop_now = false;
            if self.tp.eos.contains(&id) {
                finish = if had_tool { FinishReason::ToolCalls } else { FinishReason::Stop };
                stop_now = true;
            } else if !req.raw && matches!(id, tok::MESSAGE_USER | tok::MESSAGE_SYSTEM | tok::MESSAGE_TOOL | tok::ENDOFTEXT) {
                // the model started another participant's turn: the reply is over
                finish = if had_tool { FinishReason::ToolCalls } else { FinishReason::Stop };
                stop_now = true;
            }
            let mut events = Vec::new();
            if req.raw {
                if !stop_now {
                    raw_text_ids.push(id);
                    let s = self.tp.decode(&raw_text_ids);
                    let safe = s.trim_end_matches('\u{FFFD}').len();
                    if safe > raw_emitted {
                        events.push(ParseEvent::Content(s[raw_emitted..safe].to_string()));
                        raw_emitted = safe;
                    }
                }
            } else if stop_now {
                events = parser.finish(&self.tp);
            } else {
                events = parser.push(&self.tp, id);
            }
            for ev in events {
                let msg = match ev {
                    ParseEvent::Reasoning(s) => {
                        usage.reasoning_tokens = n;
                        GenEvent::Reasoning(s)
                    }
                    ParseEvent::ToolCall(name, arguments) => {
                        had_tool = true;
                        GenEvent::ToolCall { name, arguments }
                    }
                    ParseEvent::Content(s) => {
                        content.push_str(&s);
                        // stop strings: cut at the first match, hold back a possible partial match
                        let mut hit = None;
                        if max_stop > 0 {
                            for st in &req.stop {
                                if let Some(i) = content.find(st.as_str()) {
                                    hit = Some(hit.map_or(i, |x: usize| x.min(i)));
                                }
                            }
                        }
                        let upto = match hit {
                            Some(at) => at,
                            None if max_stop > 0 => {
                                let mut keep = content.len();
                                for k in (1..max_stop).rev() {
                                    let start = content.len().saturating_sub(k);
                                    if content.is_char_boundary(start) && req.stop.iter().any(|s| s.starts_with(&content[start..])) {
                                        keep = start;
                                        break;
                                    }
                                }
                                keep.max(emitted)
                            }
                            None => content.len(),
                        };
                        if hit.is_some() {
                            stop_now = true;
                            finish = FinishReason::Stop;
                        }
                        if upto > emitted {
                            let piece = content[emitted..upto].to_string();
                            emitted = upto;
                            GenEvent::Content(piece)
                        } else {
                            continue;
                        }
                    }
                };
                if req.events.send(msg).is_err() {
                    tracing::info!("client went away; stopping generation");
                    metrics().cancelled(req.id);
                    return Ok(());
                }
            }
            if stop_now {
                break;
            }
            if n >= req.max_tokens || self.history.len() + 1 >= self.model.ctx {
                finish = FinishReason::Length;
                for ev in parser.finish(&self.tp) {
                    let m = match ev {
                        ParseEvent::Reasoning(s) => GenEvent::Reasoning(s),
                        ParseEvent::Content(s) => GenEvent::Content(s),
                        ParseEvent::ToolCall(name, arguments) => GenEvent::ToolCall { name, arguments },
                    };
                    let _ = req.events.send(m);
                }
                break;
            }
            let pos = self.history.len();
            logits = self.forward_rows(&[id], pos, &[], true)?.context("no logits")?;
            self.history.push(id as u64);
            if n % 16 == 0 {
                self.publish_stats();
            }
        }
        if finish != FinishReason::Stop || emitted < content.len() {
            // flush held-back text when no stop string matched
            if emitted < content.len() && !req.stop.iter().any(|s| content[emitted..].contains(s.as_str())) {
                let _ = req.events.send(GenEvent::Content(content[emitted..].to_string()));
            }
        }
        let decode_s = t1.elapsed().as_secs_f64();
        tracing::info!(
            "prompt {} tok ({} cached) in {:.2} s | generated {} tok in {:.2} s ({:.1} tok/s) | {}",
            usage.prompt_tokens,
            usage.cached_tokens,
            prefill_s,
            usage.completion_tokens,
            decode_s,
            usage.completion_tokens as f64 / decode_s.max(1e-9),
            finish.as_str()
        );
        metrics().finished(req.id, finish.as_str(), &usage);
        let _ = req.events.send(GenEvent::Done { finish, usage });
        Ok(())
    }
}
