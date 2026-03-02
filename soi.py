"""soi.py – Safety-Oriented Initialization (SOI) for variable-rank LoRA fine-tuning.

Pipeline
--------
1. Extract first 300 samples from two datasets and write to soiData/
     • alpaca_cleaned_no_safety_train.csv  → soiData/task.csv   (utility)
     • advbench.txt                        → soiData/safe.csv   (safety)
2. Load llama-2-7b-chat-hf and mount forward hooks on every *up_proj* layer.
3. Forward-pass both datasets through the model to collect input activations:
     H_S  (task / utility data)
     H_T  (safe / adversarial data)
4. For each layer l:
     SVD( H_S^(l) ) → take top-r_max right singular vectors → U_S^(l)
     SVD( H_T^(l) ) → take top-r_max right singular vectors → U_T^(l)
     SOI^(l) = ‖ U_S^(l)ᵀ · U_T^(l) ‖_F
5. Allocate per-layer LoRA rank:
     r^(l) = r_min + ⌊ (r_max - r_min) · SOI^(l) / max_l(SOI^(l)) ⌋
6. Attach variable-rank LoRA adapters (up_proj only) with LoRA scaling = 1
   (lora_alpha = r per layer so that alpha / r = 1).
7. Fine-tune on the full yahma/alpaca-cleaned dataset with the Hugging Face Trainer.
8. Evaluate the fine-tuned model with run_eval() from quick_low_rank_diff_lora.py
   (wikitext PPL + harmful-rate via Llama-Guard-3-8B).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import types
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ── repo-root plumbing (mirrors quick_low_rank_diff_lora.py) ──────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "lowrank_prune"))

# Import shared utilities from the existing pipeline script.
# (module-level code in that file only sets up constants / sys.path; main() is
#  guarded by `if __name__ == "__main__":` so importing is safe.)
from quick_low_rank_diff_lora import (  # noqa: E402
    group_texts,
    load_alpaca_prompts,
    run_eval,
)

# ── paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR    = REPO_ROOT / "lowrank_prune" / "data"
ALPACA_SRC   = _DATA_DIR / "alpaca_cleaned_no_safety_train.csv"
ADVBENCH_SRC = _DATA_DIR / "advbench.txt"
SOI_DATA_DIR = REPO_ROOT / "soiData"
TASK_CSV     = SOI_DATA_DIR / "task.csv"
SAFE_CSV     = SOI_DATA_DIR / "safe.csv"

# ── default model path (same as quick_low_rank_diff_lora.py) ─────────────────
_DEFAULT_MODEL = (
    "/home/users/zhoukang/.cache/huggingface/hub/"
    "models--meta-llama--llama-2-7b-chat-hf/snapshots/"
    "f5db02db724555f92da89c216ac04704f23d4590"
)

# ── SOI hyper-parameters ──────────────────────────────────────────────────────
N_SAMPLES           = 300    # rows extracted from each source dataset
R_MAX               = 8      # maximum LoRA rank
R_MIN               = 1      # minimum LoRA rank
MAX_TOKENS_PER_LAYER = 2048  # token budget per layer for activation collection
ACT_BATCH_SIZE      = 4      # batch size during forward-hook collection
ACT_SEQ_LEN         = 128    # max sequence length during collection


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 · Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data() -> tuple[list[str], list[str]]:
    """Extract the first N_SAMPLES rows from each source dataset.

    Returns
    -------
    task_texts : list[str]  – alpaca instruction prompts (utility domain)
    safe_texts : list[str]  – advbench harmful prompts   (safety domain)
    """
    SOI_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── task.csv: first N_SAMPLES rows of alpaca utility data ────────────────
    df_alpaca = pd.read_csv(ALPACA_SRC, nrows=N_SAMPLES)
    df_alpaca.to_csv(TASK_CSV, index=False)
    print(f"[Data] Saved {len(df_alpaca)} utility samples → {TASK_CSV}")

    # ── safe.csv: first N_SAMPLES lines of advbench harmful prompts ──────────
    with open(ADVBENCH_SRC, "r", encoding="utf-8") as fh:
        adv_lines = [ln.strip() for ln in fh if ln.strip()][:N_SAMPLES]
    pd.DataFrame({"prompt": adv_lines}).to_csv(SAFE_CSV, index=False)
    print(f"[Data] Saved {len(adv_lines)} safety samples → {SAFE_CSV}")

    # Build plain-text lists for activation collection.
    # Alpaca CSV stores pre-formatted "[INST] … [/INST]" strings in "prompt".
    task_texts = df_alpaca["prompt"].astype(str).tolist()

    # Wrap advbench lines in the same Llama-2 chat template for consistency.
    safe_texts = [f"[INST] {ln} [/INST]" for ln in adv_lines]

    return task_texts, safe_texts


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 · Activation collection
# ─────────────────────────────────────────────────────────────────────────────

def _get_up_proj_names(model: AutoModelForCausalLM) -> list[str]:
    """Return all *up_proj* module names, sorted by layer index."""
    names = [
        n for n, m in model.named_modules()
        if n.endswith("up_proj") and isinstance(m, nn.Linear)
    ]
    # Sort numerically on the transformer-layer index embedded in the name
    # e.g. "model.layers.11.mlp.up_proj" → key 11
    def _layer_idx(name: str) -> int:
        try:
            return int(name.split(".layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            return -1

    return sorted(names, key=_layer_idx)


def collect_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer_names: list[str],
    device: torch.device,
    batch_size: int = ACT_BATCH_SIZE,
    max_length: int = ACT_SEQ_LEN,
    max_tokens_per_layer: int = MAX_TOKENS_PER_LAYER,
) -> Dict[str, torch.Tensor]:
    """Forward-pass *texts* through *model* with hooks on every layer in
    *layer_names* to collect input activations (stored on CPU).

    A per-layer token budget (``max_tokens_per_layer``) prevents OOM when
    there are many layers.  Early-stops once every layer is saturated.

    Parameters
    ----------
    layer_names : names returned by ``_get_up_proj_names``

    Returns
    -------
    Dict mapping layer name → FloatTensor of shape (N_tokens, d_in)
    """
    act_lists: Dict[str, list[torch.Tensor]] = {n: [] for n in layer_names}
    token_counts: Dict[str, int] = {n: 0 for n in layer_names}
    hooks: list = []

    def _make_hook(name: str):
        def _fn(module, inp, _out):
            if token_counts[name] >= max_tokens_per_layer:
                return
            x = inp[0].detach()                        # (B, T, d) or (B·T, d)
            x_flat = x.reshape(-1, x.shape[-1]).cpu()  # → (B·T, d)
            remaining = max_tokens_per_layer - token_counts[name]
            chunk = x_flat[:remaining]
            act_lists[name].append(chunk)
            token_counts[name] += chunk.shape[0]
        return _fn

    for name in layer_names:
        module = model.get_submodule(name)
        hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    # Determine which device to send inputs to (first parameter's device)
    inp_device = next(model.parameters()).device

    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="  Collecting activations",
            leave=False,
        ):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(inp_device)
            model(**enc)

            # Early-stop once every layer has enough tokens
            if all(c >= max_tokens_per_layer for c in token_counts.values()):
                break

    for h in hooks:
        h.remove()

    return {
        name: torch.cat(chunks, dim=0)
        for name, chunks in act_lists.items()
        if chunks
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 · SOI computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_soi(
    H_S: Dict[str, torch.Tensor],
    H_T: Dict[str, torch.Tensor],
    layer_names: list[str],
    r_max: int = R_MAX,
) -> Dict[str, float]:
    """Compute SOI^(l) = ‖ U_S^(l)ᵀ · U_T^(l) ‖_F for every layer.

    U_S^(l) and U_T^(l) are the top-r_max *right* singular vectors of H_S^(l)
    and H_T^(l) respectively (i.e. columns of V in the full SVD H = U·Σ·Vᵀ).

    ``torch.svd_lowrank`` is used for efficiency — it computes only the top-r
    singular triplets without materialising a full d×d matrix.
    """
    soi: Dict[str, float] = {}

    for name in tqdm(layer_names, desc="Computing SOI"):
        hs = H_S.get(name)
        ht = H_T.get(name)
        if hs is None or ht is None:
            print(f"  WARNING: missing activations for {name}, skipping.")
            soi[name] = 0.0
            continue

        hs = hs.float()  # (N_s, d_in)
        ht = ht.float()  # (N_t, d_in)

        # Clamp r to what is actually achievable given the data shapes
        r = min(r_max, hs.shape[0], hs.shape[1], ht.shape[0], ht.shape[1])

        # svd_lowrank returns (U, S, V) where V has shape (d_in, r)
        # The *right* singular vectors are the columns of V.
        _, _, V_s = torch.svd_lowrank(hs, q=r, niter=4)  # (d_in, r)
        _, _, V_t = torch.svd_lowrank(ht, q=r, niter=4)  # (d_in, r)

        # SOI = ‖ V_s^T · V_t ‖_F   shape: (r, r)
        overlap = V_s.T @ V_t
        soi_val = torch.linalg.matrix_norm(overlap, ord="fro").item()
        soi[name] = soi_val
        print(f"  {name}: SOI = {soi_val:.4f}")

    return soi


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 · Per-layer rank allocation
# ─────────────────────────────────────────────────────────────────────────────

def compute_layer_ranks(
    soi: Dict[str, float],
    layer_names: list[str],
    r_min: int = R_MIN,
    r_max: int = R_MAX,
) -> Dict[str, int]:
    """Apply the SOI-driven rank formula:

    r^(l) = r_min + ⌊ (r_max − r_min) · SOI^(l) / max_l(SOI^(l)) ⌋
    """
    max_soi = max(soi.values()) if soi else 1.0
    if max_soi == 0.0:
        max_soi = 1.0  # avoid division by zero

    ranks: Dict[str, int] = {}
    for name in layer_names:
        r = r_min + int((r_max - r_min) * soi.get(name, 0.0) / max_soi)
        ranks[name] = r

    print("\n[Ranks] Per-layer LoRA rank allocation (up_proj only):")
    for name, r in ranks.items():
        print(f"  {name}: rank = {r:2d}  (SOI = {soi.get(name, 0.0):.4f})")

    return ranks


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 · LoRA attachment with variable ranks
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_model(
    model: AutoModelForCausalLM,
    layer_ranks: Dict[str, int],
    r_default: int = R_MAX,
) -> AutoModelForCausalLM:
    """Attach variable-rank LoRA adapters on *up_proj* layers.

    LoRA scaling coefficient = lora_alpha / r = 1 per layer
    → achieved by setting alpha_pattern[name] = r (same as rank_pattern).

    PEFT's ``rank_pattern`` and ``alpha_pattern`` dicts map full module names
    to per-layer overrides; the default ``r`` / ``lora_alpha`` applies to any
    layer not explicitly listed (none in our case since we enumerate all).
    """
    rank_pattern  = dict(layer_ranks)                    # name → r
    alpha_pattern = {k: v for k, v in layer_ranks.items()}  # alpha = r → scale = 1

    lora_config = LoraConfig(
        r=r_default,
        lora_alpha=r_default,
        lora_dropout=0.0,
        target_modules=["up_proj"],
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        inference_mode=False,
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 · Fine-tuning on alpaca-cleaned
# ─────────────────────────────────────────────────────────────────────────────

def finetune(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: str,
    train_samples: int = 0,
    batch_size: int = 8,
    epochs: int = 1,
    lr: float = 2e-4,
) -> AutoModelForCausalLM:
    """Standard Hugging Face Trainer fine-tuning on alpaca-cleaned.

    ``load_alpaca_prompts`` is imported from quick_low_rank_diff_lora.py and
    uses yahma/alpaca-cleaned from Hugging Face Hub.
    """
    print("\n[Train] Loading alpaca-cleaned training prompts …")
    prompts = load_alpaca_prompts(train_samples)
    dataset = Dataset.from_dict({"text": prompts})

    def _tokenize(batch):
        return tokenizer(batch["text"])

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.map(group_texts, batched=True, batch_size=1024)

    n_blocks = len(tokenized)
    steps_per_epoch = max(1, n_blocks // batch_size)
    print(
        f"[Train] {n_blocks} token-blocks of 128 | "
        f"batch={batch_size} | epochs={epochs} | "
        f"~{steps_per_epoch * epochs} gradient steps"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.0,
        logging_steps=10,
        save_steps=500,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )
    trainer.train()

    lora_dir = os.path.join(output_dir, "LoRA")
    model.save_pretrained(lora_dir)
    print(f"[Train] LoRA adapter saved to {lora_dir}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOI-based variable-rank LoRA fine-tuning for Llama-2-7b-chat-hf"
    )
    p.add_argument("--model", type=str, default=_DEFAULT_MODEL,
                   help="Path or HF hub ID of the base model.")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Primary CUDA device (used by eval helpers).")
    p.add_argument("--output_dir", type=str, default="out/soi_lora",
                   help="Root output directory.")
    p.add_argument("--train_samples", type=int, default=0,
                   help="Alpaca-cleaned training samples (0 = full dataset).")
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--eval_batch_size", type=int, default=16,
                   help="Batch size for generation during evaluation.")
    p.add_argument("--act_batch_size", type=int, default=ACT_BATCH_SIZE,
                   help="Batch size for activation-collection forward passes.")
    p.add_argument("--act_seq_len", type=int, default=ACT_SEQ_LEN,
                   help="Max sequence length during activation collection.")
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS_PER_LAYER,
                   help="Max token budget per layer during activation collection.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 1 · Preparing SOI datasets")
    print("=" * 65)
    task_texts, safe_texts = prepare_data()

    # ── Step 2: Load model ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 2 · Loading model & tokenizer")
    print("=" * 65)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for generation consistency

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.seqlen = 4096   # used by eval_ppl inside run_eval

    layer_names = _get_up_proj_names(model)
    print(f"  Found {len(layer_names)} up_proj layers.")

    # ── Step 3: Collect activations ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 3 · Collecting input activations for up_proj layers")
    print("=" * 65)

    print("  → H_S  from task data (alpaca utility prompts) …")
    H_S = collect_activations(
        model, tokenizer, task_texts, layer_names, device,
        batch_size=args.act_batch_size,
        max_length=args.act_seq_len,
        max_tokens_per_layer=args.max_tokens,
    )

    print("  → H_T  from safe data (advbench harmful prompts) …")
    H_T = collect_activations(
        model, tokenizer, safe_texts, layer_names, device,
        batch_size=args.act_batch_size,
        max_length=args.act_seq_len,
        max_tokens_per_layer=args.max_tokens,
    )

    # Log collected shapes
    for name in layer_names:
        hs_shape = H_S[name].shape if name in H_S else "missing"
        ht_shape = H_T[name].shape if name in H_T else "missing"
        print(f"  {name}: H_S={hs_shape}, H_T={ht_shape}")

    # ── Step 4: Compute SOI ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 4 · Computing SOI per layer")
    print("=" * 65)
    soi = compute_soi(H_S, H_T, layer_names, r_max=R_MAX)

    # Release activation tensors — no longer needed
    del H_S, H_T
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 5: Rank allocation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 5 · Allocating per-layer LoRA ranks")
    print("=" * 65)
    layer_ranks = compute_layer_ranks(soi, layer_names, r_min=R_MIN, r_max=R_MAX)

    # ── Step 6: Attach LoRA ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 6 · Attaching variable-rank LoRA adapters (up_proj only)")
    print("=" * 65)
    model = build_lora_model(model, layer_ranks, r_default=R_MAX)

    # ── Step 7: Fine-tune ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 7 · Fine-tuning on yahma/alpaca-cleaned")
    print("=" * 65)
    model = finetune(
        model, tokenizer,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        batch_size=args.train_batch_size,
        epochs=args.train_epochs,
        lr=args.learning_rate,
    )

    # ── Step 8: Evaluate ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 8 · Evaluating (wikitext PPL + Llama-Guard harmful rate)")
    print("=" * 65)
    # run_eval expects an args namespace with at least eval_batch_size set.
    # model.seqlen is already set; eval_ppl reads it from the model object.
    run_eval(args, model, tokenizer, device)

    print("\n[Done] SOI-based LoRA fine-tuning complete.")
    print(f"       Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
