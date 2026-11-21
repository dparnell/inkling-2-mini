# inkling — Inkling-2-Mini inference engine (Rust + wgpu)

A from-scratch inference engine for [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
It is written in Rust, and every kernel is WGSL running on wgpu (Vulkan, DX12 or Metal). It provides:

- an **OpenAI-compatible HTTP API**, covering chat completions (streaming, reasoning, tool calls and images), completions and models;
- a **built-in web chat UI** at `/` and a **live performance dashboard** at `/dashboard`;
- an optional **terminal performance dashboard** (`serve --tui`);
- `download` to fetch the weights (resumable, SHA-256 verified);
- `chat` (a terminal REPL), `bench`, and validation tools (`logits`, `reference`, `compare`).

## The model

Inkling-2-Mini has 26.4B total and 3.4B active parameters. It is a looped MoE transformer, and its operators
were transcribed from transformers PR #51050 (`modeling_inkling2_loop.py`).

- **Layers.** It has 24 weight layers. Layers 6–17 run **three times** with shared weights, giving 48 layer
  applications per token. Each later pass restarts from the pre-loop state and gets "messages" from the
  previous pass through tanh-gated mixing (the `controllers`).
- **Block attention residuals.** Instead of plain residual sums, each layer's attention and MLP inputs are a
  learned softmax mix over RMS-normalised block states.
- **Attention.**
  - 20 sliding-window layers (window 1024) and 4 global layers.
  - No RoPE. Position comes from a hidden-state-conditioned relative bias (16-dim states × a 1024-entry
    profile bank), from causal short convs on K/V (dynamic per-token taps in the global layers), and from a
    log-length scale on Q in the global layers.
  - L2-clipped Q/K, RMS-normalised V, **per-token attention sinks**, and a sigmoid output gate.
- **MoE.** 128 experts with top-8 sigmoid routing plus a correction bias, ×8 route scale, and a softcapped
  SwiGLU. Experts are stored as **NVFP4** (layers 0–19) and **MXFP8** (layers 20–23).
- **Vision.** A native-resolution ViT: 20 px patches merged 2×2 into one LM token per 40×40 px, with 2-D axial
  RoPE and a bilinearly interpolated position grid.

## How it runs on a 16 GB GPU

The checkpoint is 19.5 GB, and 15.7 GB of that is experts. Here is how memory is split:

- Dense weights (bf16) are kept on the GPU.
- Experts stay **in their quantized formats** and are dequantized inside the matmul kernels.
- An **expert slot cache in VRAM** uses LFU-with-aging eviction. It is sized automatically from free VRAM
  (via NVML) or set with `--expert-cache-gb`.
- Missing experts stream from the memory-mapped checkpoint. A background thread pages it into RAM at startup.

Routing needs one host round trip per layer application, which decides what to upload.

Activations are f32, the KV cache is f16, and each of the 48 layer applications has its own KV cache and conv
state. The decode attention is split-K. The prefix cache works like this:

- a request that extends the previous conversation only prefills the new tokens;
- one that diverges after the previous prompt rewinds to a snapshot (conv states are snapshotted; KV is
  addressed by position);
- images are keyed by content hash.

## Quick start

```powershell
cargo build --release
.\target\release\inkling.exe download                 # ~20 GB into models/Inkling-2-Mini
.\target\release\inkling.exe serve                    # http://127.0.0.1:8080/  (UI) and /v1 (API)
.\target\release\inkling.exe serve --tui              # same, with the terminal dashboard
.\target\release\inkling.exe serve --download         # download on first start
.\target\release\inkling.exe chat --reasoning-effort low
```

Useful `serve` options:

| Option | Meaning |
| --- | --- |
| `--ctx 32768` | max context; global-layer KV memory grows with it |
| `--chunk 512` | prefill chunk size |
| `--expert-cache-gb N` | VRAM for experts (default: what is left after everything else) |
| `--no-vision` | text only (frees ~0.9 GB for experts) |
| `--max-image-tokens 1024` | larger images are downscaled to fit (40×40 px per token) |
| `--reasoning-effort low` | default when a request sets none (the template default is `high` = 0.9) |
| `--api-key KEY` | require `Authorization: Bearer KEY` |
| `--host 0.0.0.0 --port 8080` | listen address |
| `--allow-remote-media` | allow http(s) image URLs |

Use `inkling devices` to list adapters, then `--adapter N` to pick one. `WGPU_BACKEND=dx12|vulkan` overrides
the backend.

## API

`POST /v1/chat/completions` follows the OpenAI format:

- **Reasoning:** set `reasoning_effort` (`none|minimal|low|medium|high|xhigh` or a number 0–0.99). The
  thinking comes back as `reasoning_content` in messages and in stream deltas. Pass it back on assistant
  messages to keep it in context.
- **Tools:** `tools` / `tool_choice: "none"`. The model emits a structured tool call, returned as `tool_calls`
  with `finish_reason: "tool_calls"`.
- **Images:** `image_url` parts as `data:` URLs (or http(s) with `--allow-remote-media`).
- **Sampling:** `temperature`, `top_p`, `top_k`, `min_p`, penalties, `logit_bias`, `seed`, `stop`,
  `max_tokens` / `max_completion_tokens`.
- **Streaming:** `stream` and `stream_options.include_usage`. Add
  `stream_options.continuous_usage_stats: true` (as in vLLM) to get running `usage` on every chunk,
  starting with a usage-only chunk once the prompt is prefilled. The web UI uses it for its
  context meter.

`POST /v1/completions` continues raw text (no chat template). `GET /v1/models`, `GET /health` and
`GET /api/stats` (dashboard JSON) are also available.

```powershell
curl.exe http://127.0.0.1:8080/v1/chat/completions -H "content-type: application/json" -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"reasoning_effort\":\"low\"}'
```

## Validation

`inkling reference` is an independent CPU implementation of the reference forward pass (`src/reference.rs`;
it shares no code with the GPU path). `inkling compare` reports max-abs, cosine, top-5 agreement and KL.
Results on an RTX 5080 (Vulkan):

| Check | Result |
| --- | --- |
| GPU vs CPU reference, 20-token chat prompt, token-by-token decode path | max abs 0.003 on logits, same top-5, KL 2.5e-15 |
| GPU vs CPU reference, same prompt, batched prefill | identical except one row after a top-8 routing near-tie (margin 1.8e-5) at layer 16 / pass 3; KL 2.8e-12 |
| Chunked prefill (step 7 / 300) vs single-shot prefill | bit-identical |
| Split-K decode attention vs chunked prefill, 2,049-token context (ring wrap) | max abs 3e-4 |
| End to end | correct answers; tool calls; image description (red square left of a blue circle); recall of a code word at the start of a 4,120-token prompt |

To compare against the official transformers implementation (requires torch and ~60 GB RAM), use
`scripts/hf_reference.py`.

## Performance (RTX 5080 16 GB, Vulkan, ~8 GB expert cache = ~50% of experts)

From `inkling bench --prompt 512,4096,512 --gen 128 --no-vision`:

| | |
| --- | --- |
| decode | ~25–28 tok/s (512–4K context) |
| prefill, warm experts | ~350–420 tok/s |
| prefill, first request after start | much slower while experts are read from disk the first time |

Decode is bound by the 48 per-layer routing round trips. Prefill is bound by streaming experts that do not fit
in VRAM. A GPU with ≥ 24 GB holds every expert, which removes the streaming.

## Layout

| Path | Contents |
| --- | --- |
| `src/model.rs` | looped text model, KV/conv state, snapshots, forward |
| `src/experts.rs` | quantized expert slot cache |
| `src/vision.rs` | image preprocessing and ViT |
| `src/shaders/*.wgsl` | kernels: GEMM/GEMV, attention (+ split-K combine), attention residuals, short convs, router, NVFP4/MXFP8 MoE GEMM/GEMV, ViT ops |
| `src/engine.rs` | generation thread, prefix cache, output parsing |
| `src/text.rs` | tokenizer, chat template (minijinja), response parser |
| `src/server.rs` | OpenAI API, `src/ui/` web UI and dashboard |
| `src/tui.rs`, `src/metrics.rs` | dashboards |
| `src/reference.rs` | CPU reference |
| `reference/` | the third-party reference implementations this was transcribed from (see `reference/SOURCES.md`) |

## License

The code is Apache-2.0. The model is Apache-2.0, subject to Thinking Machines' acceptable use policy.
