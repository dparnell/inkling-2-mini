//! Tokenizer, chat-template rendering and the streaming response parser.

use anyhow::{Context, Result, anyhow};
use minijinja::value::{Kwargs, Value};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::Value as Json;
use std::path::Path;

pub mod tok {
    pub const ENDOFTEXT: u32 = 199999;
    pub const MESSAGE_USER: u32 = 200000;
    pub const MESSAGE_MODEL: u32 = 200001;
    pub const MESSAGE_SYSTEM: u32 = 200002;
    pub const MESSAGE_TOOL: u32 = 200003;
    pub const CONTENT_TEXT: u32 = 200004;
    pub const CONTENT_IMAGE: u32 = 200005;
    pub const END_SAMPLING: u32 = 200006;
    pub const CONTENT_THINKING: u32 = 200008;
    pub const END_MESSAGE: u32 = 200010;
    pub const CONTENT_TOOL_ERROR: u32 = 200022;
    pub const BEGIN_OF_TEXT: u32 = 200028;
    pub const INVOKE_TOOL_JSON: u32 = 200049;
    pub const IMAGE_PLACEHOLDER: u32 = 200054;
    pub const INVOKE_TOOL_TEXT: u32 = 200057;
}

pub struct TextProcessor {
    pub tokenizer: tokenizers::Tokenizer,
    env: Environment<'static>,
    pub eos: Vec<u32>,
    pub unpadded_vocab: usize,
}

impl TextProcessor {
    pub fn load(dir: &Path, unpadded_vocab: usize) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        let src = std::fs::read_to_string(dir.join("chat_template.jinja")).context("reading chat_template.jinja")?;
        Ok(Self { tokenizer, env: make_env(&src)?, eos: vec![tok::END_SAMPLING], unpadded_vocab })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_default()
    }

    /// Raw bytes of a single token (byte-level BPE); used for incremental UTF-8 decoding.
    pub fn token_str(&self, id: u32) -> String {
        self.tokenizer.decode(&[id], false).unwrap_or_default()
    }

    /// Render the checkpoint's chat template.
    pub fn render(&self, messages: &Json, tools: Option<&Json>, reasoning_effort: Option<&Json>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        let ctx = minijinja::context! {
            messages => Value::from(minijinja::value::Serde(messages)),
            tools => tools.map(|t| Value::from(minijinja::value::Serde(t))),
            reasoning_effort => reasoning_effort.map(|t| Value::from(minijinja::value::Serde(t))),
        };
        tmpl.render(ctx).map_err(|e| anyhow!("chat template: {e:#}"))
    }
}

/// The template uses the HF-only `{% generation %}` block; strip it (it only marks spans).
fn preprocess(src: &str) -> String {
    let mut s = src.to_string();
    for pat in ["{%- generation -%}", "{%- endgeneration -%}", "{% generation %}", "{% endgeneration %}"] {
        s = s.replace(pat, "");
    }
    s
}

fn make_env(src: &str) -> Result<Environment<'static>> {
    let mut env = Environment::new();
    env.set_syntax(minijinja::syntax::SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build()?);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", tojson_filter);
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> { Err(Error::new(ErrorKind::InvalidOperation, msg)) });
    env.add_template_owned("chat", preprocess(src))?;
    Ok(env)
}

/// `tojson` with Python `json.dumps` semantics as configured by transformers
/// (`ensure_ascii=False`, default separators `", "`/`": "`, optional `sort_keys`).
fn tojson_filter(v: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let sort_keys: bool = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let seps: Option<Value> = kwargs.get("separators")?;
    let _indent: Option<Value> = kwargs.get("indent")?;
    let _ensure: Option<bool> = kwargs.get("ensure_ascii")?;
    kwargs.assert_all_used()?;
    let (item, kv) = match seps {
        Some(s) => {
            let a: Vec<String> = s.try_iter()?.map(|x| x.as_str().unwrap_or("").to_string()).collect();
            (a.first().cloned().unwrap_or(", ".into()), a.get(1).cloned().unwrap_or(": ".into()))
        }
        None => (", ".into(), ": ".into()),
    };
    let j: Json = serde_json::to_value(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let mut out = String::new();
    py_dumps(&j, sort_keys, &item, &kv, &mut out);
    Ok(Value::from_safe_string(out))
}

pub fn py_dumps(j: &Json, sort_keys: bool, item: &str, kv: &str, out: &mut String) {
    match j {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Number(n) => {
            if let Some(f) = n.as_f64().filter(|_| n.is_f64()) {
                out.push_str(&py_float(f));
            } else {
                out.push_str(&n.to_string());
            }
        }
        Json::String(s) => out.push_str(&serde_json::to_string(s).unwrap()),
        Json::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                py_dumps(x, sort_keys, item, kv, out);
            }
            out.push(']');
        }
        Json::Object(m) => {
            out.push('{');
            let mut keys: Vec<&String> = m.keys().collect();
            if sort_keys {
                keys.sort();
            }
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push_str(kv);
                py_dumps(&m[k.as_str()], sort_keys, item, kv, out);
            }
            out.push('}');
        }
    }
}

fn py_float(f: f64) -> String {
    if f.is_finite() && f == f.trunc() && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// Events produced while parsing generated tokens.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseEvent {
    Reasoning(String),
    Content(String),
    /// A complete tool call: (name, arguments JSON string).
    ToolCall(String, String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Section {
    None,
    Header,
    Thinking,
    Text,
    ToolJson,
}

/// Token-level state machine over the model's output format:
/// `<|message_model|>[name]<|content_X|>...<|end_message|>` repeated, then
/// `<|content_model_end_sampling|>`.
pub struct ResponseParser {
    section: Section,
    pending: Vec<u32>,
    tool_buf: Vec<u32>,
    emitted_len: usize,
}

impl Default for ResponseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseParser {
    pub fn new() -> Self {
        Self { section: Section::None, pending: vec![], tool_buf: vec![], emitted_len: 0 }
    }

    fn flush_text(&mut self, tp: &TextProcessor, final_: bool, out: &mut Vec<ParseEvent>) {
        if self.pending.is_empty() {
            return;
        }
        let s = tp.decode(&self.pending);
        // Hold back an incomplete UTF-8 sequence (decoder emits U+FFFD) until more tokens arrive.
        let safe = if final_ { s.len() } else { s.trim_end_matches('\u{FFFD}').len() };
        if safe > self.emitted_len {
            let piece = s[self.emitted_len..safe].to_string();
            self.emitted_len = safe;
            match self.section {
                Section::Thinking => out.push(ParseEvent::Reasoning(piece)),
                _ => out.push(ParseEvent::Content(piece)),
            }
        }
        if final_ {
            self.pending.clear();
            self.emitted_len = 0;
        }
    }

    pub fn push(&mut self, tp: &TextProcessor, id: u32) -> Vec<ParseEvent> {
        use tok::*;
        let mut out = vec![];
        match id {
            MESSAGE_MODEL | MESSAGE_USER | MESSAGE_SYSTEM | MESSAGE_TOOL => {
                self.finish_section(tp, &mut out);
                self.section = Section::Header;
            }
            CONTENT_THINKING => {
                self.finish_section(tp, &mut out);
                self.section = Section::Thinking;
            }
            CONTENT_TEXT | CONTENT_TOOL_ERROR | INVOKE_TOOL_TEXT => {
                self.finish_section(tp, &mut out);
                self.section = Section::Text;
            }
            INVOKE_TOOL_JSON => {
                self.finish_section(tp, &mut out);
                self.section = Section::ToolJson;
                self.tool_buf.clear();
            }
            END_MESSAGE | END_SAMPLING => {
                self.finish_section(tp, &mut out);
                self.section = Section::None;
            }
            _ => match self.section {
                Section::Thinking | Section::Text => {
                    self.pending.push(id);
                    self.flush_text(tp, false, &mut out);
                }
                Section::ToolJson => self.tool_buf.push(id),
                Section::Header => {}
                // Text outside any section (model skipped the header): treat as content.
                Section::None => {
                    self.section = Section::Text;
                    self.pending.push(id);
                    self.flush_text(tp, false, &mut out);
                }
            },
        }
        out
    }

    fn finish_section(&mut self, tp: &TextProcessor, out: &mut Vec<ParseEvent>) {
        match self.section {
            Section::Thinking | Section::Text => self.flush_text(tp, true, out),
            Section::ToolJson => {
                let s = tp.decode(&self.tool_buf);
                self.tool_buf.clear();
                match serde_json::from_str::<Json>(&s) {
                    Ok(j) => {
                        let name = j.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                        let args = j.get("args").cloned().unwrap_or(Json::Object(Default::default()));
                        out.push(ParseEvent::ToolCall(name, serde_json::to_string(&args).unwrap()));
                    }
                    Err(_) => out.push(ParseEvent::Content(s)),
                }
            }
            _ => {}
        }
    }

    /// Call at end of generation (e.g. max_tokens hit) to flush anything pending.
    pub fn finish(&mut self, tp: &TextProcessor) -> Vec<ParseEvent> {
        let mut out = vec![];
        self.finish_section(tp, &mut out);
        self.section = Section::None;
        out
    }
}
