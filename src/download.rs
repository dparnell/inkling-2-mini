//! Hugging Face model download: resolves the repo revision, downloads the files the runtime
//! needs with resume + SHA-256 verification, and writes a completion marker.

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

pub const DEFAULT_REPO: &str = "thinkingmachines/Inkling-2-Mini";
const MARKER: &str = ".inkling-download-complete";

#[derive(Deserialize)]
struct RepoInfo {
    sha: String,
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    sha256: String,
    size: u64,
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into()).trim_end_matches('/').to_string()
}

fn token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.trim().is_empty() {
            return Some(t.trim().to_string());
        }
    }
    let home = std::env::var("HF_HOME").map(PathBuf::from).ok().or_else(|| {
        std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok().map(|h| PathBuf::from(h).join(".cache").join("huggingface"))
    })?;
    std::fs::read_to_string(home.join("token")).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Files the runtime reads. Everything else (README, images, python) is skipped.
fn wanted(path: &str, include_mtp: bool) -> bool {
    if path == "mtp.safetensors" {
        return include_mtp;
    }
    path.ends_with(".json") || path.ends_with(".jinja") || path.ends_with(".safetensors") || path.starts_with("tiktoken/") || path == "LICENSE"
}

fn safe_rel(path: &str) -> Result<PathBuf> {
    let p = Path::new(path);
    for c in p.components() {
        match c {
            Component::Normal(_) => {}
            _ => bail!("refusing unsafe repository path {path:?}"),
        }
    }
    Ok(p.to_path_buf())
}

pub fn is_complete(dir: &Path) -> bool {
    dir.join(MARKER).exists()
}

/// Default local directory for a repo: `./models/<name>`.
pub fn default_dir(repo: &str) -> PathBuf {
    PathBuf::from("models").join(repo.rsplit('/').next().unwrap_or(repo))
}

pub async fn download(repo: &str, revision: &str, dir: &Path, include_mtp: bool) -> Result<()> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .user_agent("inkling-engine/0.1")
        .build()?;
    let ep = endpoint();
    let tok = token();
    let auth = |rb: reqwest::RequestBuilder| match &tok {
        Some(t) => rb.bearer_auth(t),
        None => rb,
    };
    let url = format!("{ep}/api/models/{repo}/revision/{revision}?blobs=true");
    let resp = auth(client.get(&url)).send().await.context("contacting Hugging Face")?;
    if resp.status() == 401 || resp.status() == 403 {
        bail!("access denied ({}) for {repo}: the repo may be gated; accept its terms on huggingface.co and set HF_TOKEN", resp.status());
    }
    let info: RepoInfo = resp.error_for_status()?.json().await.context("parsing repo info")?;
    tracing::info!("{repo}@{} ({} files listed)", &info.sha[..12.min(info.sha.len())], info.siblings.len());
    std::fs::create_dir_all(dir)?;

    let files: Vec<&Sibling> = info.siblings.iter().filter(|s| wanted(&s.rfilename, include_mtp)).collect();
    let total: u64 = files.iter().map(|s| s.lfs.as_ref().map(|l| l.size).or(s.size).unwrap_or(0)).sum();
    println!("Downloading {} files ({:.2} GB) from {repo} into {}", files.len(), total as f64 / 1e9, dir.display());

    for s in files {
        let rel = safe_rel(&s.rfilename)?;
        let dest = dir.join(&rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let size = s.lfs.as_ref().map(|l| l.size).or(s.size);
        let sha = s.lfs.as_ref().map(|l| l.sha256.clone());
        if let (Ok(m), Some(sz)) = (std::fs::metadata(&dest), size) {
            if m.len() == sz {
                println!("  ok   {}", s.rfilename);
                continue;
            }
        }
        let url = format!("{ep}/{repo}/resolve/{}/{}", info.sha, s.rfilename);
        download_file(&client, &auth, &url, &dest, size, sha.as_deref(), &s.rfilename).await?;
    }
    std::fs::write(dir.join(MARKER), format!("{repo}@{}\n", info.sha))?;
    println!("Download complete.");
    Ok(())
}

async fn download_file(
    client: &reqwest::Client,
    auth: &impl Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder,
    url: &str,
    dest: &Path,
    size: Option<u64>,
    sha256: Option<&str>,
    name: &str,
) -> Result<()> {
    let part = dest.with_extension(format!("{}part", dest.extension().map(|e| format!("{}.", e.to_string_lossy())).unwrap_or_default()));
    let mut attempt = 0;
    loop {
        attempt += 1;
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let mut rb = auth(client.get(url));
        if have > 0 {
            rb = rb.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let res: Result<()> = async {
            let resp = rb.send().await?;
            let status = resp.status();
            if status == 401 || status == 403 {
                bail!("access denied ({status}) downloading {name}; set HF_TOKEN if the repo is gated");
            }
            let resp = resp.error_for_status()?;
            let append = status == reqwest::StatusCode::PARTIAL_CONTENT;
            let mut f = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(&part)?;
            let mut done = if append { have } else { 0 };
            let total = size.unwrap_or(done + resp.content_length().unwrap_or(0));
            let mut stream = resp.bytes_stream();
            let t0 = Instant::now();
            let start = done;
            let mut last = Instant::now();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                f.write_all(&chunk)?;
                done += chunk.len() as u64;
                if last.elapsed() > Duration::from_millis(500) {
                    last = Instant::now();
                    let rate = (done - start) as f64 / t0.elapsed().as_secs_f64().max(1e-3);
                    print!("\r  get  {name}: {:.1}% {:.2}/{:.2} GB @ {:.1} MB/s   ", 100.0 * done as f64 / total.max(1) as f64, done as f64 / 1e9, total as f64 / 1e9, rate / 1e6);
                    let _ = std::io::stdout().flush();
                }
            }
            f.flush()?;
            Ok(())
        }
        .await;
        match res {
            Ok(()) => break,
            Err(e) if attempt < 8 => {
                eprintln!("\n  retry {name} after error: {e:#}");
                tokio::time::sleep(Duration::from_secs(2 * attempt)).await;
            }
            Err(e) => return Err(e),
        }
    }
    println!();
    let got = std::fs::metadata(&part)?.len();
    if let Some(sz) = size {
        if got != sz {
            bail!("{name}: size mismatch ({got} != {sz})");
        }
    }
    if let Some(want) = sha256 {
        print!("  sha  {name} ... ");
        let _ = std::io::stdout().flush();
        let p = part.clone();
        let got = tokio::task::spawn_blocking(move || sha256_file(&p)).await??;
        if !got.eq_ignore_ascii_case(want) {
            let _ = std::fs::remove_file(&part);
            bail!("{name}: SHA-256 mismatch (got {got}, want {want}); partial file removed");
        }
        println!("ok");
    }
    std::fs::rename(&part, dest)?;
    Ok(())
}

fn sha256_file(p: &Path) -> Result<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}
