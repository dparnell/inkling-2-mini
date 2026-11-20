"""Dump next-token logits from the official transformers implementation for comparison with
`inkling logits` / `inkling reference`.

Requirements (CPU is fine, ~60 GB RAM because experts are dequantized to bf16):
    pip install torch numpy "git+https://github.com/huggingface/transformers@6173a2f679ff4ed393afc499aefc569da7e68bc1"

Usage:
    python scripts/hf_reference.py --model models/Inkling-2-Mini --prompt "<|message_user|><|content_text|>Hi<|end_message|>" --out hf.bin
    inkling compare hf.bin gpu.bin
"""

import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, Inkling2LoopForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Inkling-2-Mini")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Inkling2LoopForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype))
    model.eval()
    ids = tok(a.prompt, add_special_tokens=False, return_tensors="pt").input_ids
    print(f"{ids.shape[1]} tokens")
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1].float().numpy()
    logits.astype(np.float32).tofile(a.out)
    top = np.argsort(-logits)[:8]
    for i in top:
        print(f"{i:>7} {logits[i]:9.4f} {tok.decode([int(i)])!r}")


if __name__ == "__main__":
    main()
