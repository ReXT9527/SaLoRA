# Analysis: Harmless Fine-tuning Safety Alignment Anomaly (Revised)

## 1. Experiment Results Summary

### Harmful Fine-tuning (Behaves as Expected)

| Lambda | Refusal Rate | LG Safety | LG Harmful | Total ‖DW‖ |
|--------|-------------|-----------|------------|------------|
| 0.0    | 4.4%        | 3.7%      | **96.4%**  | 41.27      |
| 0.1    | 10.2%       | 6.6%      | 93.4%      | 18.89      |
| 1.0    | 21.2%       | 21.9%     | 78.1%      | 11.03      |
| 10.0   | **97.1%**   | **97.1%** | 2.9%       | 6.85       |

Higher lambda = stronger safety preservation. Consistent with expectations.

### Harmless Fine-tuning (Anomalous)

| Lambda | Refusal Rate | LG Safety | LG Harmful | Total ‖DW‖ |
|--------|-------------|-----------|------------|------------|
| 0.0    | 97.81%      | 97.81%    | 2.19%      | 195.84     |
| 0.001  | 97.81%      | 97.81%    | 2.19%      | 119.98     |
| 0.01   | **98.54%**  | **98.54%**| **1.46%**  | 86.52      |
| 0.1    | **98.54%**  | **98.54%**| **1.46%**  | 58.56      |
| 1.0    | 97.08%      | 97.81%    | 2.19%      | 34.61      |
| 10.0   | 93.43%      | 96.35%    | 3.65%      | 14.07      |
| 100.0  | 95.62%      | 96.35%    | 3.65%      | 5.70       |

Other papers: Standard LoRA on Alpaca -> ~23-28% harmful rate (Llama-2-chat-7B, rank 16).
This experiment: Standard LoRA (lambda=0) -> **2.19% harmful rate**.

**Three core anomalies:**
1. lambda=0 maintains safety (2.19% vs other papers' ~24%)
2. Increasing lambda makes safety WORSE (from 1.46% to 3.65%)
3. Harmless fine-tuning penalty is much larger than harmful fine-tuning penalty — this is unreasonable

---

## 2. Confirmed: lambda=0 IS Standard LoRA

From `quick_low_rank_diff_lora.py:297-309`:

```python
def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    self._activation_cache.clear()
    outputs = model(**inputs)          # Standard forward pass
    loss = outputs.loss                # Standard CE loss
    if self.safety_lambda != 0:        # <-- When lambda=0, this entire block is SKIPPED
        penalty = self._safety_penalty(model, ...)
        loss = loss + self.safety_lambda * penalty
    return (loss, outputs) if return_outputs else loss
```

When `safety_lambda=0`, the penalty is **never computed**. The loss is purely `outputs.loss` (standard cross-entropy). The activation hooks are registered but never consumed — they have zero effect on gradients.

**This is identical to standard LoRA fine-tuning with the HuggingFace Trainer.** There is no hidden mechanism affecting training.

Training configuration (matches `lora_train_act.py` exactly):
- Dataset: `yahma/alpaca-cleaned` (51,760 samples)
- Format: `[INST]{instruction} {input}[/INST]{output}`
- block_size=128, batch_size=16, epochs=1, lr=1e-4
- LoRA: rank=16, alpha=16, dropout=0, target_modules=q_proj,v_proj
- Init: Default (A=Kaiming, B=0, so initial DW=0)

---

## 3. Root Cause Analysis

### 3.1 Why lambda=0 Maintains Safety (~2% vs Literature's ~24%)

The training configuration is identical to `lora_train_act.py` (same data, same format, same block_size=128, same hyperparameters). The model DID learn — ‖DW‖=195.84, training loss dropped from 2.20 to 1.32, PPL=6.228. Yet safety barely degraded (2.19% harmful vs base model's ~0%).

**Possible explanations:**

1. **Evaluation pipeline difference**: This experiment uses HuggingFace `model.generate()` with transformers, while the paper likely evaluates with vLLM (`lora_test_eval.py`). Even with identical parameters (temperature=0, same prompt format), different inference frameworks can produce different outputs due to:
   - Different KV-cache implementations
   - Different attention computation precision
   - Different tokenizer handling edge cases
   - vLLM's PagedAttention vs HuggingFace's standard attention

2. **Evaluation set difference**: This experiment uses 137 prompts from `harm_test.csv`. The papers typically use AdvBench's full test subset (520 prompts). Different prompt distributions may have different difficulty levels.

3. **The LoRA weight change avoids safety-critical directions**: ‖DW‖=195.84 is large in magnitude, but the weight change is distributed across 64 layers. Standard LoRA with B=0 initialization learns directions driven by Alpaca's gradient signal. Alpaca is entirely benign instruction-following data — its gradient signal points toward better benign task performance, NOT toward overriding safety refusals. The safety-critical directions (those that encode "refuse harmful requests") may simply not overlap with the directions useful for Alpaca. This would explain why a large ‖DW‖ has minimal safety impact.

4. **This is an open question**: The exact mechanism by which some LoRA fine-tuning setups break safety (24%) while seemingly identical setups don't (2%) warrants further investigation. The reproducibility of the paper's 24% baseline should be verified with the same evaluation pipeline.

### 3.2 Why Harmless Penalty >> Harmful Penalty (BUG: Missing Batch Normalization)

**This is the most critical finding.** The penalty values from the logs:

| Step | Harmful (lambda=0.1, batch=4) | Harmless (lambda=0.1, batch=16) | Ratio |
|------|-------------------------------|----------------------------------|-------|
| 2    | 0.3353                        | **1.8560**                       | 5.5x  |
| 5    | 0.1347                        | 0.2879                           | 2.1x  |
| 10   | 0.0742                        | 0.1589                           | 2.1x  |
| 20   | 0.0906                        | 0.1234                           | 1.4x  |

The harmless penalty is consistently 2-6x larger than the harmful penalty at the same lambda. This is **unreasonable** — harmless data should produce a SMALLER safety penalty than harmful data.

**Root cause: The penalty is not normalized by batch size (N = batch_size × seq_len).**

From `quick_low_rank_diff_lora.py:376`:
```python
proj_norm_sq = (proj ** 2).sum()    # <-- SUM, not MEAN
layer_penalty = proj_norm_sq
```

Where `proj` has shape `(r_s, N)` with:
- Harmless: N = 16 × 128 = **2048**
- Harmful: N = 4 × 128 = **512**

The penalty sums over all N columns, so it scales **linearly with batch size**:

```
penalty_harmless / penalty_harmful ≈ N_harmless / N_harmful = 2048 / 512 = 4x
```

This means the **effective lambda for harmless training is ~4x larger than for harmful training**, even when the nominal lambda values are identical.

**Consequences:**
- lambda=0.1 for harmless training acts like lambda≈0.4 for harmful training
- lambda=10 for harmless acts like lambda≈40 for harmful
- This explains why the harmless penalty is so disproportionately large
- This makes cross-configuration comparison (harmful vs harmless) invalid

**Fix: Normalize by N:**
```python
# Current (buggy):
proj_norm_sq = (proj ** 2).sum()

# Fixed:
N = proj.shape[1]  # batch_size * seq_len
proj_norm_sq = (proj ** 2).sum() / N
```

### 3.3 Why Increasing Lambda WORSENS Safety in Harmless Fine-tuning

Combined effect of the batch normalization bug + over-regularization:

```
Lambda=0:   ‖DW‖ = 195.84, grad_norm ≈ 0.33  -> Stable learning, safety intact
Lambda=10:  ‖DW‖ = 14.07,  grad_norm ≈ 6.0   -> Severely constrained (effective ~40x)
Lambda=100: ‖DW‖ = 5.70,   grad_norm ≈ 238    -> Training destroyed (effective ~400x)
```

**The mechanism:**

1. **Over-regularization prevents learning**: At lambda=100, the first-10-step loss is **63.0** (vs 2.14 at lambda=0). The penalty dominates the loss. The model's LoRA barely changes (‖DW‖=5.70 vs 195.84).

2. **Training instability from gradient explosion**: Gradient norms go from 0.33 (lambda=0) to 238 (lambda=100). The optimizer makes erratic updates rather than converging.

3. **Degraded output quality looks like reduced safety**:
   - A well-trained model produces clear refusals: "I'm sorry, I cannot help with that."
   - An over-constrained model produces degraded/incomplete outputs
   - Degraded outputs may not contain refusal prefixes ("I'm sorry", "I cannot", etc.)
   - Llama-Guard classifies incoherent outputs as "unsafe" more readily than clear refusals

4. **The fundamental paradox**: The model's safety comes from its BASE weights, not the LoRA update. When lambda=0 already has 97.81% safety, the penalty has nothing to protect. Instead, it prevents the model from learning benign task performance, degrading overall output quality, which paradoxically reduces safety metrics.

| Lambda | Train Loss | PPL    | Refusal | LG Harmful | Interpretation |
|--------|-----------|--------|---------|------------|----------------|
| 0.0    | 1.320     | 6.228  | 97.81%  | 2.19%      | Good learning, full safety |
| 0.01   | 1.354     | **6.071** | **98.54%** | **1.46%** | Mild regularizer, slight benefit |
| 1.0    | 1.425     | 6.214  | 97.08%  | 2.19%      | Starting to constrain |
| 10.0   | 1.536     | 6.313  | 93.43%  | 3.65%      | Over-regularized, safety drops |
| 100.0  | 2.033     | 6.207  | 95.62%  | 3.65%      | Training nearly broken |

---

## 4. Summary

| Question | Answer |
|----------|--------|
| Is lambda=0 standard LoRA? | **Yes.** When `safety_lambda=0`, penalty is never computed. Loss = pure CE. Identical to HuggingFace Trainer standard LoRA. |
| Why is lambda=0 so safe (2% vs 24%)? | Training config matches the paper, but evaluation differs (transformers vs vLLM, 137 vs 520 prompts). The LoRA weight change (‖DW‖=196) may avoid safety-critical directions because Alpaca gradients don't point toward those directions. Needs further investigation. |
| Why is harmless penalty > harmful penalty? | **Bug: penalty uses `.sum()` not `.mean()`.** Harmless batch=16 (N=2048) vs harmful batch=4 (N=512) → harmless penalty is ~4x larger. Effective lambda for harmless is ~4x that of harmful. |
| Why does increasing lambda worsen safety? | Over-regularization (amplified by the batch normalization bug) prevents learning → degraded outputs → fewer refusal prefixes matched → worse safety scores. |

## 5. Key Actions

1. **Fix the batch normalization bug**: Divide penalty by N to make it batch-size-invariant
2. **Investigate the evaluation pipeline discrepancy**: Run baseline evaluation with vLLM and AdvBench's full test subset to determine if the 2% vs 24% gap is due to evaluation methodology
3. **After fixing**: Re-run harmless experiments to see if the penalty behaves correctly (smaller for harmless, larger for harmful)
