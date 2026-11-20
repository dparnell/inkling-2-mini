//! Terminal chat REPL and the throughput benchmark.

use crate::{Loaded, engine, metrics, sampler, text};
use anyhow::Result;
use serde_json::{Value, json};
use std::io::{BufRead, Write};
use std::sync::Arc;

pub fn repl(engine: engine::EngineHandle, tp: Arc<text::TextProcessor>, effort: &str, temperature: f32, max_tokens: usize) -> Result<()> {
    let mut messages: Vec<Value> = Vec::new();
    println!("Inkling-2-Mini chat. Commands: /reset, /effort <none|minimal|low|medium|high|xhigh>, /quit");
    let mut effort = effort.to_string();
    let stdin = std::io::stdin();
    loop {
        print!("\n> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            return Ok(());
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            return Ok(());
        }
        if line == "/reset" {
            messages.clear();
            println!("(conversation cleared)");
            continue;
        }
        if let Some(e) = line.strip_prefix("/effort ") {
            effort = e.trim().to_string();
            println!("(reasoning effort: {effort})");
            continue;
        }
        messages.push(json!({"role": "user", "content": line}));
        let rendered = tp.render(&Value::Array(messages.clone()), None, Some(&json!(effort)))?;
        let prompt = tp.encode(&rendered)?;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let id = metrics::metrics().queued("cli", "text".into(), "text", true, prompt.len());
        let sampling = sampler::SamplingParams { temperature, ..Default::default() };
        engine.submit(engine::GenRequest { id, prompt, images: vec![], max_tokens, sampling, stop: vec![], raw: false, events: tx })?;
        let (mut reasoning, mut content) = (String::new(), String::new());
        let mut in_reasoning = false;
        while let Some(ev) = rx.blocking_recv() {
            match ev {
                engine::GenEvent::Reasoning(s) => {
                    if !in_reasoning {
                        print!("\x1b[2m[thinking] ");
                        in_reasoning = true;
                    }
                    print!("{s}");
                    reasoning.push_str(&s);
                }
                engine::GenEvent::Content(s) => {
                    if in_reasoning {
                        print!("\x1b[0m\n\n");
                        in_reasoning = false;
                    }
                    print!("{s}");
                    content.push_str(&s);
                }
                engine::GenEvent::ToolCall { name, arguments } => println!("[tool call {name}({arguments})]"),
                engine::GenEvent::Done { usage, finish } => {
                    if in_reasoning {
                        print!("\x1b[0m");
                    }
                    println!(
                        "\n\x1b[2m[{} prompt tokens ({} cached), {} generated, {}]\x1b[0m",
                        usage.prompt_tokens,
                        usage.cached_tokens,
                        usage.completion_tokens,
                        finish.as_str()
                    );
                    break;
                }
                engine::GenEvent::Error(e) => {
                    println!("\nerror: {e}");
                    break;
                }
            }
            std::io::stdout().flush()?;
        }
        let mut msg = json!({"role": "assistant", "content": content});
        if !reasoning.is_empty() {
            msg["reasoning_content"] = json!(reasoning);
        }
        messages.push(msg);
    }
}

pub fn bench(l: &mut Loaded, lens: &[usize], n_gen: usize) -> Result<()> {
    let text = std::fs::read_to_string("README.md").unwrap_or_default().repeat(64);
    let mut corpus = l.tp.encode(&text)?;
    if corpus.len() < 4096 {
        corpus = (0..65536u32).map(|i| 1000 + (i * 7919) % 50000).collect();
    }
    for &n in lens {
        l.model.reset(&l.gpu);
        let ids = &corpus[..n.min(corpus.len())];
        let t0 = std::time::Instant::now();
        let mut pos = 0;
        let mut logits = None;
        while pos < ids.len() {
            let c = l.model.tm.min(ids.len() - pos);
            let rows = l.model.embed_rows(&ids[pos..pos + c])?;
            logits = l.model.forward(&mut l.gpu, &rows, &[], pos, pos + c == ids.len())?;
            pos += c;
        }
        let pf = t0.elapsed().as_secs_f64();
        let mut next = sampler::argmax(&logits.unwrap());
        let (h0, m0, b0) = l.model.experts.stats();
        let t1 = std::time::Instant::now();
        for i in 0..n_gen {
            let rows = l.model.embed_rows(&[next])?;
            let lg = l.model.forward(&mut l.gpu, &rows, &[], ids.len() + i, true)?.unwrap();
            next = sampler::argmax(&lg);
        }
        let dt = t1.elapsed().as_secs_f64();
        let (h1, m1, b1) = l.model.experts.stats();
        let hit = (h1 - h0) as f64 / ((h1 - h0) + (m1 - m0)).max(1) as f64;
        println!(
            "prompt {n:>6}: prefill {:>7.1} tok/s ({pf:.2} s) | decode {:>6.2} tok/s | decode expert hit {:.1}%, streamed {:.2} GB",
            ids.len() as f64 / pf,
            n_gen as f64 / dt,
            hit * 100.0,
            (b1 - b0) as f64 / 1e9
        );
    }
    Ok(())
}
