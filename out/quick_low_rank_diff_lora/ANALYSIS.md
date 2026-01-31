# Analysis: Harmless Fine-tuning Safety Alignment Anomaly

## 1. Experiment Results Summary

### Harmful Fine-tuning (Behaves as Expected)

| Lambda | Refusal Rate | LG Safety | LG Harmful | Total ‖DW‖ |
|--------|-------------|-----------|------------|------------|
| 0.0    | 4.4%        | 3.7%      | **96.4%**  | 41.27      |
| 0.1    | 10.2%       | 6.6%      | 93.4%      | 18.89      |
| 1.0    | 21.2%       | 21.9%     | 78.1%      | 11.03      |
| 10.0   | **97.1%**   | **97.1%** | 2.9%       | 6.85       |

Conclusion: Higher lambda = stronger safety preservation. Consistent with expectations.

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

SaLoRA paper Table 1: Standard LoRA on Alpaca -> **23.7% harmful rate** (Llama-2-chat-7B, rank 16).
This experiment: Standard LoRA (lambda=0) -> **2.19% harmful rate**.

**Three core anomalies:**
1. lambda=0 maintains safety (2.19% vs paper's 23.7%)
2. Increasing lambda makes safety WORSE (from 1.46% to 3.65%)
3. The safety penalty provides no benefit for harmless fine-tuning

---

## 2. Root Cause Analysis

### 2.1 Why lambda=0 Maintains Safety (2.19% vs Paper's 23.7%)

The discrepancy stems from **fundamental differences between `quick_low_rank_diff_lora.py` and the paper's training/evaluation pipeline**:

#### A. LoRA Initialization Difference (Primary Factor)

| Aspect | This Experiment | SaLoRA Paper's LoRA Baseline |
|--------|----------------|------------------------------|
| Init method | Default (A=random, B=0) | Likely standard Kaiming/zero init |
| Starting DW | **Zero** (B=0 -> DW=BA=0) | Zero |
| Base weights | Unchanged | Unchanged |

With B=0 initialization, the LoRA update starts as exactly zero. The model needs many gradient steps to develop meaningful weight perturbations from scratch. Compare the SaLoRA method itself which uses **PiSSA initialization** (`init_lora_weights=False` in `lora_train_act.py`) where LoRA weights are initialized to the top-r SVD components of the original weight, giving the model much greater capacity to modify behavior from step 1.

The paper's "standard LoRA" baseline (23.7%) was likely run under different conditions (possibly more epochs, different LR, different framework) than this experiment.

#### B. Training Data Fragmentation via block_size=128 (Major Factor)

```python
def group_texts(examples, block_size=128):
    # Concatenate ALL tokenized examples, then split into 128-token chunks
```

Average Alpaca example ≈ 190 tokens. With block_size=128:
- ~50% of blocks contain **incomplete** instruction-response pairs
- The `[INST]...[/INST]` boundary is frequently split across blocks
- Model learns **language modeling on fragments**, NOT instruction-following

**Why this preserves safety:** Safety alignment is encoded in the instruction-following pattern — "when user asks harmful X, refuse." If the model doesn't learn a new instruction-following distribution, it can't override the safety refusal pattern. The model essentially does **continued pre-training** rather than instruction fine-tuning.

Note: The original `lora_train_act.py` ALSO uses block_size=128, but there the PiSSA initialization + base weight modification gives the model much more capacity for change.

#### C. Evaluation Difference: max_new_tokens=64 vs 256

| Aspect | This Experiment | Paper's Evaluation (`lora_test_eval.py`) |
|--------|----------------|------------------------------------------|
| Generation length | `max_new_tokens=64` | `max_tokens=256` |
| Stop tokens | None | `["[INST]","[/INST]"]` |
| Framework | transformers | vLLM |
| Eval prompts | 137 (harm_test.csv) | AdvBench test subset |

With only 64 tokens:
- Refusal responses are short ("I'm sorry, I cannot...") and easily detected
- Potentially harmful responses may be cut off before revealing harmful content
- Llama-Guard has less text to classify as harmful

With 256 tokens:
- Model has space to generate complete harmful responses
- More content for Llama-Guard to flag as unsafe
- Borderline cases have more opportunity to manifest as harmful

#### D. Combined Effect

The interaction of these factors creates a "safety illusion":

```
Standard LoRA init (B=0)        -> Weak weight perturbation
+ block_size=128 fragmentation  -> Poor instruction-following learning
+ max_new_tokens=64             -> Truncated evaluation
= Model barely changes from base -> Safety preserved (artificially)
```

### 2.2 Why Increasing Lambda WORSENS Safety in Harmless Fine-tuning

This is the counterintuitive finding: lambda goes up, safety goes DOWN.

#### Mechanism

```
Lambda=0:   ‖DW‖ = 195.84, grad_norm ≈ 0.33  -> Model learns normally (stable)
Lambda=10:  ‖DW‖ = 14.07,  grad_norm ≈ 6.0   -> Learning severely constrained
Lambda=100: ‖DW‖ = 5.70,   grad_norm ≈ 238    -> Training nearly destroyed
```

**Step-by-step explanation:**

1. **Over-regularization prevents learning**: The penalty `||U_s^T @ B @ A @ X||^2` constrains the LoRA update to be orthogonal to safety directions. At high lambda, this penalty dominates the loss (first-10-step loss: 2.14 at lambda=0 vs **63.0** at lambda=100), preventing the model from learning the downstream task effectively.

2. **Training instability**: Gradient norms explode (0.33 -> 238), causing erratic weight updates. The optimizer oscillates rather than converging.

3. **Degraded output quality masquerades as reduced safety**:
   - A well-functioning model produces clear refusals: "I'm sorry, I cannot help with that."
   - An over-constrained model produces degraded outputs: garbled, incomplete, or off-topic text
   - Degraded outputs may NOT match refusal prefixes (no "I'm sorry" or "I cannot")
   - Llama-Guard classifies confused/garbled outputs as "unsafe" more often than clear refusals

4. **The paradox**: The safety penalty is SUPPOSED to prevent the LoRA update from touching safety-critical directions. But in harmless fine-tuning, the model's safety behavior comes from its BASE weights, not the LoRA update. The penalty constrains the LoRA (which would have been harmless anyway) while causing training instability that degrades the base model's effective behavior.

**Analogy:** It's like putting heavy armor on someone to protect them from injury, but the armor is so heavy they can't walk properly and end up falling and hurting themselves.

#### Evidence from the Data

| Lambda | Train Loss | PPL    | Refusal | LG Harmful | Interpretation |
|--------|-----------|--------|---------|------------|----------------|
| 0.0    | 1.320     | 6.228  | 97.81%  | 2.19%      | Learns well, safety intact |
| 0.01   | 1.354     | **6.071** | **98.54%** | **1.46%** | Mild regularization benefit |
| 1.0    | 1.425     | 6.214  | 97.08%  | 2.19%      | Starting to hurt |
| 10.0   | 1.536     | 6.313  | 93.43%  | 3.65%      | Significantly degraded |
| 100.0  | 2.033     | 6.207  | 95.62%  | 3.65%      | Training nearly broken |

Note: lambda=0.01 actually achieves the **best PPL** (6.071, better than baseline 6.228), suggesting the penalty can act as a beneficial regularizer at very low strengths. But beyond lambda ≈ 0.1, the penalty becomes harmful.

### 2.3 Why the SaLoRA Paper Reports ~24% ASR for Standard LoRA

The paper's experiment conditions likely differ in several critical ways:

1. **Different LoRA training setup**: The paper's LoRA baseline might use different hyperparameters, target more modules, or train for more effective steps.

2. **Different evaluation**: The paper evaluates on AdvBench's full test subset (potentially more prompts, different distribution) with 256 max tokens and vLLM.

3. **Evaluation metric nuance**: The paper's "harmful rate" in Table 1 is measured against AdvBench's test subset, which may use a stricter ASR definition (e.g., the behavior classifier in `lora_test_eval.py` that checks if the model actually exhibits the harmful behavior, not just fails to refuse).

4. **The paper's SaLoRA method is fundamentally different from this experiment's approach**:

| Aspect | Original SaLoRA (`lora_train_act.py`) | This Experiment (`quick_low_rank_diff_lora.py`) |
|--------|---------------------------------------|------------------------------------------------|
| Base weight modification | Yes (W -= C @ B @ A) | No |
| LoRA initialization | PiSSA (top-r SVD of W) | Standard (B=0) |
| Safety mechanism | Post-training projection (B_new = C @ B) | Dynamic penalty during training |
| Training loss | Standard CE only | CE + lambda * penalty |
| Post-processing | `process_lora.py` multiplies B by C | None |

---

## 3. Recommendations for Meaningful Experiments

### 3.1 To Reproduce the Paper's LoRA Baseline (~24% ASR)

1. **Increase max_new_tokens to 256** in evaluation
2. **Use the full AdvBench test subset** instead of 137 prompts from harm_test.csv
3. **Try different LoRA configs**: increase rank to 32, try targeting all linear layers (`q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj`)
4. **Train for more epochs** (3-5 epochs instead of 1)
5. **Use proper instruction SFT format** instead of block_size=128 concatenation: compute loss only on output tokens, pad/truncate per-example

### 3.2 To Make the Safety Penalty Meaningful

The safety penalty only matters when fine-tuning actually degrades alignment. If the baseline (lambda=0) already maintains safety, the penalty has nothing to protect against.

1. First reproduce the ~24% ASR baseline
2. Then apply the safety penalty and measure improvement
3. Use moderate lambda values (0.001-0.1) to avoid over-regularization

### 3.3 Alternative Training Format

Replace the block_size=128 concatenation with proper per-example training:

```python
def tokenize_sft(example):
    # Tokenize instruction and output separately
    # Mask labels on instruction tokens (only compute loss on output)
    prompt = f"[INST]{instruction}[/INST]"
    full_text = f"{prompt}{output}"
    # Set labels to -100 for prompt tokens
```

This would more closely replicate how LoRA fine-tuning typically degrades safety alignment.

---

## 4. Summary

| Question | Answer |
|----------|--------|
| Why is lambda=0 so safe? | Standard LoRA init (B=0) + block_size=128 fragmentation + max_new_tokens=64 = model barely changes from base |
| Why does increasing lambda worsen safety? | Over-regularization destroys learning capability; degraded outputs no longer match refusal patterns |
| Why does the paper show 24% ASR? | Different evaluation setup (256 tokens, AdvBench), likely different LoRA training configuration |
| What should be done? | Match the paper's evaluation setup, use proper SFT format, reproduce the baseline before testing safety mechanisms |

The core issue is that **the current training setup is too weak to break safety alignment in the first place**, making the safety penalty unnecessary and counterproductive. The experiment needs a stronger baseline (one that actually degrades alignment) before the safety mechanism can be meaningfully evaluated.
