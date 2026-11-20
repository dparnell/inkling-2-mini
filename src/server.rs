//! OpenAI-compatible HTTP API plus the built-in web UI and dashboard:
//! `POST /v1/chat/completions` (text + images, tools, reasoning), `POST /v1/completions`,
//! `GET /v1/models`, `GET /health`, `GET /` (chat UI), `GET /dashboard`, `GET /api/stats`.

use crate::engine::{EngineHandle, FinishReason, GenEvent, GenRequest, Usage};
use crate::metrics::metrics;
use crate::sampler::SamplingParams;
use crate::text::tok;
use crate::vision::ImageInput;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

pub struct ServerOptions {
    pub model_name: String,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub allow_remote_media: bool,
    pub ui: bool,
    pub max_image_tokens: usize,
    pub default_reasoning_effort: Option<String>,
}

struct AppState {
    engine: EngineHandle,
    opts: ServerOptions,
    http: reqwest::Client,
}

type St = Arc<AppState>;

pub struct ApiError(StatusCode, String);

#[derive(Clone)]
struct ErrorMessage(String);

async fn track(req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    let resp = next.run(req).await;
    let status = resp.status();
    if path.starts_with("/v1/") && status.is_client_error() {
        let msg = resp.extensions().get::<ErrorMessage>().map(|m| m.0.clone()).unwrap_or_else(|| status.canonical_reason().unwrap_or("").to_string());
        metrics().rejected(path.trim_start_matches("/v1/"), status.as_u16(), msg);
    }
    resp
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = if self.0.is_client_error() { "invalid_request_error" } else { "server_error" };
        let mut resp = (self.0, Json(json!({"error": {"message": self.1, "type": kind, "param": null, "code": null}}))).into_response();
        resp.extensions_mut().insert(ErrorMessage(self.1));
        resp
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

pub fn run(engine: EngineHandle, opts: ServerOptions) -> anyhow::Result<()> {
    let addr = format!("{}:{}", opts.host, opts.port);
    let state = Arc::new(AppState { engine, opts, http: reqwest::Client::builder().timeout(std::time::Duration::from_secs(30)).build()? });
    let api = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{id}", get(get_model))
        .route("/api/stats", get(stats))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .route("/health", get(|| async { "ok" }));
    let app = if state.opts.ui {
        api.route("/", get(ui_page))
            .route("/dashboard", get(dashboard_page))
            .route("/favicon.svg", get(favicon))
            .route("/favicon.ico", get(favicon))
    } else {
        api
    };
    let app = app.layer(middleware::from_fn(track)).layer(tower_http::cors::CorsLayer::permissive()).with_state(state.clone());

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let ui = state.opts.ui;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        let shown = addr.replace("0.0.0.0:", "127.0.0.1:").replace("[::]:", "[::1]:");
        tracing::info!("OpenAI API at http://{shown}/v1");
        if ui {
            tracing::info!("chat UI at http://{shown}/  |  dashboard at http://{shown}/dashboard");
        }
        metrics().set_info(|i| {
            i.addr = format!("http://{shown}");
            i.status = "serving".into();
        });
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = crate::metrics::SHUTDOWN_NOTIFY.notified() => {}
                }
                tracing::info!("shutting down");
            })
            .await?;
        Ok(())
    })
}

const UI_HTML: &str = include_str!("ui/index.html");
const DASH_HTML: &str = include_str!("ui/dashboard.html");
const FAVICON_SVG: &str = include_str!("ui/favicon.svg");

async fn favicon() -> Response {
    ([(header::CONTENT_TYPE, "image/svg+xml"), (header::CACHE_CONTROL, "max-age=86400")], FAVICON_SVG).into_response()
}

async fn ui_page() -> Response {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], UI_HTML).into_response()
}

async fn dashboard_page() -> Response {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], DASH_HTML).into_response()
}

async fn stats() -> Json<Value> {
    Json(crate::metrics::snapshot_json(300))
}

async fn auth(State(st): State<St>, req: Request, next: Next) -> Result<Response, ApiError> {
    if let Some(key) = &st.opts.api_key {
        let ok = req.headers().get("authorization").and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).is_some_and(|v| v == key);
        if !ok {
            return Err(ApiError(StatusCode::UNAUTHORIZED, "invalid API key".into()));
        }
    }
    Ok(next.run(req).await)
}

fn model_obj(st: &AppState) -> Value {
    let mut input = vec!["text"];
    if st.engine.caps.vision {
        input.push("image");
    }
    json!({
        "id": st.opts.model_name, "object": "model", "created": 0, "owned_by": "thinkingmachines",
        "max_model_len": st.engine.max_ctx, "modalities": {"input": input, "output": ["text"]},
        "capabilities": {"tools": true, "reasoning": true, "vision": st.engine.caps.vision},
    })
}

async fn list_models(State(st): State<St>) -> Json<Value> {
    Json(json!({"object": "list", "data": [model_obj(&st)]}))
}

async fn get_model(State(st): State<St>, axum::extract::Path(id): axum::extract::Path<String>) -> Result<Json<Value>, ApiError> {
    if id == st.opts.model_name { Ok(Json(model_obj(&st))) } else { Err(ApiError(StatusCode::NOT_FOUND, format!("model `{id}` not found"))) }
}

// -------------------------------------------------------------------------------------------
// request parsing

struct Common {
    sampling: SamplingParams,
    max_tokens: Option<usize>,
    stop: Vec<String>,
    stream: bool,
    include_usage: bool,
}

fn f32_field(body: &Value, k: &str) -> Result<Option<f32>, ApiError> {
    match body.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().map(|x| Some(x as f32)).ok_or_else(|| bad(format!("`{k}` must be a number"))),
    }
}

fn usize_field(body: &Value, k: &str) -> Result<Option<usize>, ApiError> {
    match body.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_u64().map(|x| Some(x as usize)).ok_or_else(|| bad(format!("`{k}` must be a non-negative integer"))),
    }
}

fn parse_common(body: &Value) -> Result<Common, ApiError> {
    if usize_field(body, "n")?.unwrap_or(1) != 1 {
        return Err(bad("only n=1 is supported"));
    }
    let mut s = SamplingParams::default();
    if let Some(t) = f32_field(body, "temperature")? {
        if !(0.0..=2.0).contains(&t) {
            return Err(bad("temperature must be between 0 and 2"));
        }
        s.temperature = t;
    }
    if let Some(v) = f32_field(body, "top_p")? {
        s.top_p = v;
    }
    if let Some(v) = usize_field(body, "top_k")? {
        s.top_k = v;
    }
    if let Some(v) = f32_field(body, "min_p")? {
        s.min_p = v;
    }
    if let Some(v) = f32_field(body, "repetition_penalty")? {
        s.repetition_penalty = v;
    }
    if let Some(v) = f32_field(body, "presence_penalty")? {
        s.presence_penalty = v;
    }
    if let Some(v) = f32_field(body, "frequency_penalty")? {
        s.frequency_penalty = v;
    }
    s.seed = body.get("seed").and_then(Value::as_u64);
    if let Some(lb) = body.get("logit_bias").and_then(Value::as_object) {
        let mut m = HashMap::new();
        for (k, v) in lb {
            let id: u32 = k.parse().map_err(|_| bad("logit_bias keys must be token ids"))?;
            m.insert(id, v.as_f64().unwrap_or(0.0) as f32);
        }
        s.logit_bias = m;
    }
    let max_tokens = match usize_field(body, "max_completion_tokens")? {
        Some(v) => Some(v),
        None => usize_field(body, "max_tokens")?,
    };
    let stop = match body.get("stop") {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => return Err(bad("`stop` must be a string or an array of strings")),
    };
    Ok(Common {
        sampling: s,
        max_tokens,
        stop: stop.into_iter().filter(|s| !s.is_empty()).collect(),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        include_usage: body.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false),
    })
}

async fn fetch_url(st: &AppState, url: &str) -> Result<Vec<u8>, ApiError> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').ok_or_else(|| bad("malformed data URL"))?;
        if !meta.ends_with(";base64") {
            return Err(bad("data URLs must be base64-encoded"));
        }
        return B64.decode(data.trim()).map_err(|_| bad("invalid base64 in data URL"));
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        if !st.opts.allow_remote_media {
            return Err(bad("remote image URLs are disabled; send a base64 data URL or start the server with --allow-remote-media"));
        }
        let resp = st.http.get(url).send().await.map_err(|e| bad(format!("fetching {url}: {e}")))?;
        if !resp.status().is_success() {
            return Err(bad(format!("fetching {url}: HTTP {}", resp.status())));
        }
        let b = resp.bytes().await.map_err(|e| bad(format!("fetching {url}: {e}")))?;
        if b.len() > 64 * 1024 * 1024 {
            return Err(bad("remote image is larger than 64 MiB"));
        }
        return Ok(b.to_vec());
    }
    Err(bad("image URLs must be data: URLs or http(s) URLs"))
}

/// Convert OpenAI messages to the template's shape; images are decoded in order.
async fn normalize_messages(st: &AppState, messages: &Value) -> Result<(Vec<Value>, Vec<ImageInput>), ApiError> {
    let arr = messages.as_array().ok_or_else(|| bad("`messages` must be an array"))?;
    if arr.is_empty() {
        return Err(bad("`messages` must not be empty"));
    }
    let mut out = Vec::with_capacity(arr.len());
    let mut raw_images: Vec<Vec<u8>> = Vec::new();
    for m in arr {
        let mut m = m.as_object().cloned().ok_or_else(|| bad("each message must be an object"))?;
        let role = m.get("role").and_then(Value::as_str).unwrap_or("").to_string();
        if !["system", "developer", "user", "assistant", "tool"].contains(&role.as_str()) {
            return Err(bad(format!("unsupported message role `{role}`")));
        }
        if role == "developer" {
            m.insert("role".into(), json!("system"));
        }
        // OpenAI-style reasoning field names
        if role == "assistant" && !m.contains_key("reasoning_content") {
            if let Some(r) = m.get("reasoning").cloned() {
                m.insert("reasoning_content".into(), r);
            }
        }
        match m.get("content").cloned() {
            Some(Value::Array(parts)) => {
                let mut new_parts = Vec::with_capacity(parts.len());
                for p in parts {
                    match p.get("type").and_then(Value::as_str) {
                        Some("text") | Some("input_text") | None => new_parts.push(json!({"type": "text", "text": p.get("text").and_then(Value::as_str).unwrap_or("")})),
                        Some("image_url") | Some("image") | Some("input_image") => {
                            if !st.engine.caps.vision {
                                return Err(bad("this model was loaded without image support"));
                            }
                            if !["user", "tool"].contains(&role.as_str()) {
                                return Err(bad("images are only supported in user and tool messages"));
                            }
                            let url = p
                                .pointer("/image_url/url")
                                .or_else(|| p.get("image_url"))
                                .or_else(|| p.get("image"))
                                .and_then(Value::as_str)
                                .ok_or_else(|| bad("image part needs image_url.url"))?;
                            raw_images.push(fetch_url(st, url).await?);
                            new_parts.push(json!({"type": "image"}));
                        }
                        Some(other) => return Err(bad(format!("content part type `{other}` is not supported"))),
                    }
                }
                m.insert("content".into(), Value::Array(new_parts));
            }
            Some(Value::Null) | None => {
                m.insert("content".into(), json!(""));
            }
            _ => {}
        }
        // the template wants tool-call arguments as objects; OpenAI sends JSON strings
        if let Some(Value::Array(calls)) = m.get_mut("tool_calls") {
            for c in calls.iter_mut() {
                if let Some(args) = c.pointer_mut("/function/arguments") {
                    if let Value::String(s) = args {
                        *args = if s.trim().is_empty() {
                            json!({})
                        } else {
                            serde_json::from_str(s).map_err(|_| bad("tool_calls[].function.arguments must be a JSON object"))?
                        };
                    }
                }
            }
        }
        out.push(Value::Object(m));
    }
    let max_tokens = st.opts.max_image_tokens;
    let images = tokio::task::spawn_blocking(move || raw_images.iter().map(|b| ImageInput::from_bytes(b, max_tokens)).collect::<anyhow::Result<Vec<_>>>())
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| bad(format!("{e:#}")))?;
    Ok((out, images))
}

/// Render the chat template, tokenize and expand each image placeholder to its token run.
fn build_prompt(st: &AppState, messages: &Value, tools: Option<&Value>, effort: Option<&Value>, images: &[ImageInput]) -> Result<Vec<u32>, ApiError> {
    let text = st.engine.tp.render(messages, tools, effort).map_err(|e| bad(format!("{e:#}")))?;
    let ids = st.engine.tp.encode(&text).map_err(|e| bad(format!("{e:#}")))?;
    let mut out = Vec::with_capacity(ids.len());
    let mut next = 0;
    for id in ids {
        if id == tok::IMAGE_PLACEHOLDER {
            let img = images.get(next).ok_or_else(|| bad("more image placeholders than images"))?;
            out.extend(std::iter::repeat_n(tok::IMAGE_PLACEHOLDER, img.n_tokens()));
            next += 1;
        } else {
            out.push(id);
        }
    }
    if next != images.len() {
        return Err(bad("image count does not match the rendered prompt"));
    }
    Ok(out)
}

fn start(st: &AppState, prompt: Vec<u32>, images: Vec<ImageInput>, c: &Common, endpoint: &str, raw: bool) -> Result<mpsc::UnboundedReceiver<GenEvent>, ApiError> {
    let room = st.engine.max_ctx.saturating_sub(prompt.len());
    if room == 0 {
        return Err(bad(format!("prompt is {} tokens, exceeding the context window of {}", prompt.len(), st.engine.max_ctx)));
    }
    let (tx, rx) = mpsc::unbounded_channel();
    let input = match images.len() {
        0 => "text".to_string(),
        1 => "text+image".to_string(),
        n => format!("text+{n}image"),
    };
    let id = metrics().queued(endpoint, input, "text", c.stream, prompt.len());
    st.engine
        .submit(GenRequest { id, prompt, images, max_tokens: c.max_tokens.unwrap_or(room).min(room), sampling: c.sampling.clone(), stop: c.stop.clone(), raw, events: tx })
        .map_err(|e| {
            metrics().failed(id, e.to_string());
            ApiError(StatusCode::SERVICE_UNAVAILABLE, e.to_string())
        })?;
    Ok(rx)
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": u.cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": u.reasoning_tokens},
    })
}

fn tool_call_json(i: usize, name: &str, arguments: &str) -> Value {
    json!({"index": i, "id": format!("call_{}", uuid::Uuid::new_v4().simple()), "type": "function",
           "function": {"name": name, "arguments": arguments}})
}

// -------------------------------------------------------------------------------------------
// /v1/chat/completions

async fn chat_completions(State(st): State<St>, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let c = parse_common(&body)?;
    let (messages, images) = normalize_messages(&st, body.get("messages").unwrap_or(&Value::Null)).await?;
    let tool_choice_none = body.get("tool_choice").and_then(Value::as_str) == Some("none");
    let tools = body.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty()) && !tool_choice_none).cloned();
    let effort = body
        .get("reasoning_effort")
        .or_else(|| body.pointer("/reasoning/effort"))
        .or_else(|| body.pointer("/chat_template_kwargs/reasoning_effort"))
        .filter(|v| !v.is_null())
        .cloned()
        .or_else(|| st.opts.default_reasoning_effort.clone().map(Value::String));
    let prompt = build_prompt(&st, &Value::Array(messages), tools.as_ref(), effort.as_ref(), &images)?;
    let rx = start(&st, prompt, images, &c, "chat", false)?;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now();
    let model = st.opts.model_name.clone();

    if !c.stream {
        let mut rx = rx;
        let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
        loop {
            match rx.recv().await {
                Some(GenEvent::Content(s)) => content.push_str(&s),
                Some(GenEvent::Reasoning(s)) => reasoning.push_str(&s),
                Some(GenEvent::ToolCall { name, arguments }) => calls.push(tool_call_json(calls.len(), &name, &arguments)),
                Some(GenEvent::Done { finish, usage }) => {
                    let mut message = json!({"role": "assistant", "content": content});
                    if !reasoning.is_empty() {
                        message["reasoning_content"] = json!(reasoning);
                    }
                    let mut fr = finish.as_str();
                    if !calls.is_empty() {
                        message["tool_calls"] = Value::Array(calls);
                        if content.is_empty() {
                            message["content"] = Value::Null;
                        }
                        fr = FinishReason::ToolCalls.as_str();
                    }
                    return Ok(Json(json!({
                        "id": id, "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0, "message": message, "finish_reason": fr, "logprobs": null}],
                        "usage": usage_json(&usage),
                    }))
                    .into_response());
                }
                Some(GenEvent::Error(e)) => return Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, e)),
                None => return Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, "generation aborted".into())),
            }
        }
    }

    let (tx, out) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let chunk = move |delta: Value, finish: Option<&str>| -> Event {
        Event::default().data(
            json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}]})
            .to_string(),
        )
    };
    tokio::spawn(async move {
        let mut rx = rx;
        let send = |e: Event| tx.send(Ok(e)).is_ok();
        if !send(chunk(json!({"role": "assistant", "content": ""}), None)) {
            return;
        }
        let mut n_calls = 0;
        while let Some(ev) = rx.recv().await {
            let ok = match ev {
                GenEvent::Content(s) => send(chunk(json!({"content": s}), None)),
                GenEvent::Reasoning(s) => send(chunk(json!({"reasoning_content": s}), None)),
                GenEvent::ToolCall { name, arguments } => {
                    let d = json!({"tool_calls": [tool_call_json(n_calls, &name, &arguments)]});
                    n_calls += 1;
                    send(chunk(d, None))
                }
                GenEvent::Done { finish, usage } => {
                    let fr = if n_calls > 0 { "tool_calls" } else { finish.as_str() };
                    send(chunk(json!({}), Some(fr)));
                    if c.include_usage {
                        let u = json!({"id": "chatcmpl-usage", "object": "chat.completion.chunk", "created": created, "choices": [], "usage": usage_json(&usage)});
                        send(Event::default().data(u.to_string()));
                    }
                    send(Event::default().data("[DONE]"));
                    return;
                }
                GenEvent::Error(e) => {
                    send(Event::default().data(json!({"error": {"message": e, "type": "server_error"}}).to_string()));
                    return;
                }
            };
            if !ok {
                return;
            }
        }
    });
    Ok(Sse::new(tokio_stream::wrappers::UnboundedReceiverStream::new(out)).keep_alive(KeepAlive::default()).into_response())
}

// -------------------------------------------------------------------------------------------
// /v1/completions (raw text continuation)

async fn completions(State(st): State<St>, Json(body): Json<Value>) -> Result<Response, ApiError> {
    let c = parse_common(&body)?;
    let tp = &st.engine.tp;
    let prompt = match body.get("prompt") {
        Some(Value::String(s)) => tp.encode(s).map_err(|e| bad(e.to_string()))?,
        Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => tp.encode(a[0].as_str().unwrap()).map_err(|e| bad(e.to_string()))?,
        Some(Value::Array(a)) if !a.is_empty() && a.iter().all(Value::is_u64) => a.iter().map(|x| x.as_u64().unwrap() as u32).collect(),
        _ => return Err(bad("`prompt` must be a string (batched prompts are not supported)")),
    };
    if prompt.is_empty() {
        return Err(bad("`prompt` must not be empty"));
    }
    if prompt.contains(&tok::IMAGE_PLACEHOLDER) {
        return Err(bad("image placeholder tokens are not allowed in raw completions"));
    }
    let c = Common { max_tokens: Some(c.max_tokens.unwrap_or(16)), ..c };
    let rx = start(&st, prompt, vec![], &c, "completions", true)?;
    let id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now();
    let model = st.opts.model_name.clone();
    if !c.stream {
        let mut rx = rx;
        let mut text = String::new();
        loop {
            match rx.recv().await {
                Some(GenEvent::Content(s)) | Some(GenEvent::Reasoning(s)) => text.push_str(&s),
                Some(GenEvent::ToolCall { .. }) => {}
                Some(GenEvent::Done { finish, usage }) => {
                    return Ok(Json(json!({
                        "id": id, "object": "text_completion", "created": created, "model": model,
                        "choices": [{"index": 0, "text": text, "finish_reason": finish.as_str(), "logprobs": null}],
                        "usage": usage_json(&usage),
                    }))
                    .into_response());
                }
                Some(GenEvent::Error(e)) => return Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, e)),
                None => return Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, "generation aborted".into())),
            }
        }
    }
    let (tx, out) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    tokio::spawn(async move {
        let mut rx = rx;
        let chunk = |text: &str, finish: Option<&str>| {
            Event::default().data(json!({"id": id, "object": "text_completion", "created": created, "model": model,
                                         "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": null}]}).to_string())
        };
        while let Some(ev) = rx.recv().await {
            match ev {
                GenEvent::Content(s) | GenEvent::Reasoning(s) => {
                    if tx.send(Ok(chunk(&s, None))).is_err() {
                        return;
                    }
                }
                GenEvent::ToolCall { .. } => {}
                GenEvent::Done { finish, usage } => {
                    let _ = tx.send(Ok(chunk("", Some(finish.as_str()))));
                    if c.include_usage {
                        let u = json!({"id": id, "object": "text_completion", "created": created, "choices": [], "usage": usage_json(&usage)});
                        let _ = tx.send(Ok(Event::default().data(u.to_string())));
                    }
                    let _ = tx.send(Ok(Event::default().data("[DONE]")));
                    return;
                }
                GenEvent::Error(e) => {
                    let _ = tx.send(Ok(Event::default().data(json!({"error": {"message": e, "type": "server_error"}}).to_string())));
                    return;
                }
            }
        }
    });
    Ok(Sse::new(tokio_stream::wrappers::UnboundedReceiverStream::new(out)).keep_alive(KeepAlive::default()).into_response())
}
