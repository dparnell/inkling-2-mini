//! Token sampling on the host from the full logit vector.

use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub logit_bias: HashMap<u32, f32>,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        // the model card recommends greedy decoding for text
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: HashMap::new(),
            seed: None,
        }
    }
}

pub struct Sampler {
    p: SamplingParams,
    rng: StdRng,
    counts: HashMap<u32, u32>,
}

impl Sampler {
    pub fn new(p: SamplingParams) -> Self {
        let rng = match p.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        Self {
            p,
            rng,
            counts: HashMap::new(),
        }
    }

    /// Tokens that count for the repetition penalty (prompt + generated, like transformers).
    pub fn observe(&mut self, id: u32) {
        *self.counts.entry(id).or_insert(0) += 1;
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let p = &self.p;
        for (&id, &b) in &p.logit_bias {
            if let Some(l) = logits.get_mut(id as usize) {
                *l += b;
            }
        }
        if p.repetition_penalty != 1.0 || p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 {
            for (&id, &c) in &self.counts {
                let Some(l) = logits.get_mut(id as usize) else {
                    continue;
                };
                if p.repetition_penalty != 1.0 {
                    *l = if *l > 0.0 {
                        *l / p.repetition_penalty
                    } else {
                        *l * p.repetition_penalty
                    };
                }
                *l -= p.presence_penalty + p.frequency_penalty * c as f32;
            }
        }

        if p.temperature <= 0.0 || p.top_k == 1 {
            return argmax(logits);
        }

        // candidates sorted by logit
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        let k = if p.top_k > 0 {
            p.top_k.min(idx.len())
        } else {
            idx.len()
        };
        // partial sort for efficiency when top_k is set; otherwise keep a generous candidate set
        let keep = k.min(1024).max(1);
        idx.select_nth_unstable_by(keep - 1, |&a, &b| {
            logits[b as usize].total_cmp(&logits[a as usize])
        });
        idx.truncate(keep);
        idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));

        let inv_t = 1.0 / p.temperature;
        let max = logits[idx[0] as usize];
        let mut probs: Vec<f32> = idx
            .iter()
            .map(|&i| ((logits[i as usize] - max) * inv_t).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= sum);

        let mut n = probs.len();
        if p.min_p > 0.0 {
            let thr = probs[0] * p.min_p;
            n = probs.iter().take_while(|&&x| x >= thr).count().max(1);
        }
        if p.top_p < 1.0 {
            let mut c = 0.0;
            let mut m = 0;
            for &x in &probs[..n] {
                c += x;
                m += 1;
                if c >= p.top_p {
                    break;
                }
            }
            n = m.max(1);
        }
        let total: f32 = probs[..n].iter().sum();
        let mut r = self.rng.random::<f32>() * total;
        for i in 0..n {
            r -= probs[i];
            if r <= 0.0 {
                return idx[i];
            }
        }
        idx[n - 1]
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}
