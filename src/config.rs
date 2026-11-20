//! Model configuration (`config.json`) for Inkling-2 (`inkling2_loop`).

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    #[serde(default)]
    pub is_varloop: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub model_max_length: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: f64,
    pub log_scaling_alpha: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub use_value_norm_no_gamma: bool,
    pub qk_l2_clip_threshold: f64,
    pub rel_l2_clip_threshold: f64,
    pub local_layer_ids: Vec<usize>,
    pub sconv_kernel_size: usize,
    pub unpadded_vocab_size: usize,
    pub final_logit_softcapping: f64,
    pub sliding_window_size: usize,
    #[serde(default)]
    pub swa_head_dim: Option<usize>,
    pub n_layers_per_attn_res_block: usize,
    #[serde(default)]
    pub attn_res_normalize_values: bool,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    pub intermediate_size: usize,
    pub mlp_gate_softcap: f64,
    pub mlp_up_softcap: f64,
    pub route_scale: f64,
    #[serde(default)]
    pub use_gate_bias: bool,
    pub loop_config: LoopConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub text_hidden_size: usize,
    pub temporal_patch_size: usize,
    pub native_patch_size: usize,
    pub adapter_merge_size: usize,
    pub projection_hidden_dim: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub mlp_ratio: f64,
    pub posemb_grid_size: usize,
    pub layer_norm_eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub eos_token_id: u32,
    pub text_config: TextConfig,
    pub vision_config: Option<VisionConfig>,
}

impl ModelConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        serde_json::from_str(&s).with_context(|| format!("parsing {}", p.display()))
    }
}

impl TextConfig {
    pub fn swa_head_dim(&self) -> usize {
        self.swa_head_dim.unwrap_or(self.head_dim)
    }

    pub fn is_local(&self, layer: usize) -> bool {
        self.local_layer_ids.contains(&layer)
    }
}
