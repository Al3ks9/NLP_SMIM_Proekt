# Gemma 4 E4B student — design

**Date:** 2026-10-02
**Status:** draft, awaiting review
**Depends on:** the Qwen3 SFT pipeline, the GRPO stage
(`2026-09-27-grpo-stage-design.md`) and the GRPO-v2 reward
(`slurm/train_grpo_v2.slurm`).

## 1. Purpose

The thesis question is how good a small, locally runnable model can get at
writing Macedonian poetry in a named poet's style, under this project's data
and reward. Qwen3-4B is the only student so far, and both its open failures —
style collapse onto the corpus's largest authors and weak author
identification after GRPO — could be properties of Qwen rather than of the
method. A second student of matched size from a different family separates
the two:

- If Gemma collapses the same way, the cause is the data/reward, and the
  thesis says so with two models behind it.
- If Gemma does markedly better, the base model matters, and the thesis can
  say by how much.

The only thing that changes is the student. Same synthetic data (the 1,146
Gemma-31B train rows), same task format (`sft_data.build_messages`), same
hyperparameters, same GRPO-v2 reward, same 468 val pairs, same seeds.

## 2. Model choice

`google/gemma-4-E4B-it`: about 4.5B effective parameters, about 8B with
per-layer embeddings, Apache 2.0, instruction-tuned. This is the closest size
match to Qwen3-4B (also a post-trained chat model). Gemma 4 12B is the
fallback if E4B fails the compatibility check (§4), and the optional "does
scale help" point if there is time. It is not part of this spec's scope.

The teacher is the same family (Gemma 4 31B). That's a confound worth one
sentence in the thesis: the student may find the teacher's distribution
easier to imitate. It isn't a reason to choose otherwise, because a
same-family student is also the realistic deployment choice.

## 3. What is Qwen-specific today, and the change for each

Verified against transformers 5.9.0, TRL 1.13.0 and PEFT 0.21.0 (the
`uv.lock` versions, Python 3.11 resolution):

| Today | Gemma 4 fact | Change |
|---|---|---|
| `AutoModelForCausalLM` everywhere | maps `gemma4` → `Gemma4ForConditionalGeneration` (vision + audio towers loaded); `gemma4_text` → `Gemma4ForCausalLM` | load through a family-aware loader (§5); prefer text-only if the checkpoint loads into `Gemma4ForCausalLM` cleanly |
| LoRA targets by suffix `q_proj … down_proj` | the vision/audio towers also have `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`, but as `Gemma4ClippableLinear` (a wrapper around `.linear`, not an `nn.Linear`) — a suffix match would hit them and PEFT would reject or mis-wrap them | regex restricted to the language model: `.*language_model.*\.(q_proj\|k_proj\|v_proj\|o_proj\|gate_proj\|up_proj\|down_proj)$` (or no prefix needed if loaded text-only) |
| — | KV-shared layers (the last `num_kv_shared_layers`) have no `k_proj`/`v_proj` | nothing to do: those layers just get fewer adapters. Logged as a count |
| — | per-layer-input linears `per_layer_input_gate` / `per_layer_projection` | **excluded**, for parity with Qwen's target set (attention + MLP only) |
| `assistant_only_loss=True` (TRL patches `{% generation %}` into Qwen3's template) | TRL 1.13 ships `gemma4.jinja` but **no** `gemma4_training.jinja`: `assistant_only_loss` does not work | Gemma uses TRL's prompt–completion format with `completion_only_loss=True` (§6) |
| `enable_thinking=False` in `render_chat` | Gemma 4's template takes the same `enable_thinking` kwarg, and thinking is off unless it is set | no change |
| end of turn `<|im_end|>` | `<turn|>` | no code depends on the literal; the compatibility check asserts the rendered completion ends with the template's own end token |
| `generate_validation` leaves `top_k=None` → model's `generation_config` (Qwen3 ships `top_k=20`) | Gemma ships its own defaults (expected `top_k=64`, `top_p=0.95`) | pin `top_k=20` explicitly for Gemma runs, so decoding matches the Qwen runs |
| `max_seq_length=1024` | different tokenizer: truncation rate unknown | see §7 |

## 4. Step 0 — compatibility check (GPU, ~20 min, `slurm/gemma_compat.slurm`)

Runs before any training and decides whether this spec proceeds as written,
falls back to 12B, or stops. It prints, and fails on:

1. Load `gemma-4-E4B-it` text-only (`Gemma4ForCausalLM`). Report missing or
   unexpected keys. If text weights are missing, fall back to
   `Gemma4ForConditionalGeneration` and record which loader was used.
2. Count the modules the LoRA regex matches per type (q/k/v/o/gate/up/down).
   Assert the count is non-zero, and that no vision/audio module and no
   `ClippableLinear` matched.
3. Render one training example through `sft_data` both ways (chat with an
   assistant turn; prompt + completion). Assert the prompt is a strict prefix
   of the full chat, and that the completion ends with the template's
   end-of-turn token.
4. Token-length distribution of all 1,146 train examples under Gemma's
   tokenizer (feeds §7).
5. Generate one poem from the untuned model for a fixed val pair. Assert it
   isn't empty and has no thinking channel.
6. Three-step SFT smoke, a merge round-trip (merged vs adapter log-prob of a
   fixed completion agree to within bf16 tolerance), then
   `train_grpo.py --smoke`.

## 5. Model-family registry — `src/model_families.py`

One small module so the family is chosen in one place, not by editing five
scripts:

```python
@dataclass(frozen=True)
class ModelFamily:
    key: str                  # 'qwen3' | 'gemma4'
    hf_name: str              # 'Qwen/Qwen3-4B' | 'google/gemma-4-E4B-it'
    tag: str                  # dir/label prefix: 'qwen3' | 'gemma4-e4b'
    target_modules: list | str   # list (suffixes) or a regex string
    loss_masking: str         # 'assistant_only' | 'completion_only'
    loader: str               # 'auto' | 'gemma4_text'
    generation_overrides: dict  # {} | {'top_k': 20}
```

- `load_causal_lm(family, path, **kw)` and `load_tokenizer(family, path)` are
  used by `train_sft`, `merge_sft_adapter`, `train_grpo`,
  `generate_validation` and `evaluate_reward` in place of their direct
  `AutoModelForCausalLM.from_pretrained` calls.
- Each script gets `--family {qwen3,gemma4}` (default `qwen3`). Default output
  dirs derive from `tag`: `models/gemma4-e4b-lora-sft`,
  `models/gemma4-e4b-sft-merged`, `models/gemma4-e4b-grpo`. Validation CSVs
  are `base-gemma-4-E4B-it_val.csv`, `gemma4-e4b-lora-sft_val.csv`, and so on.
- **The Qwen path is unchanged in behaviour.** With `--family qwen3` every
  script resolves to exactly today's model, targets, masking and dirs. The
  existing tests stay green, and v1/v2 remain reproducible.
- `run_config.json` / `merge_info.json` record the family.

## 6. Loss masking for Gemma: prompt–completion format

`sft_data` gains `build_prompt_completion(tokenizer, row)`, which returns
`{'prompt': render_inference_prompt(...), 'completion': poem + end_of_turn}`.
`build_hf_dataset(rows, fmt='messages' | 'prompt_completion')` chooses the
column layout. `train_sft` chooses by `family.loss_masking`:
`assistant_only_loss=True` on `messages` for Qwen, and
`completion_only_loss=True` on prompt–completion for Gemma.

Both mask the loss to exactly the assistant poem plus its end-of-turn token.
A test checks this on the tiny Qwen model already used in `tests/`: for the
same example, the trainable-token mask from both formats is identical. That
test is what lets the thesis call the two students' SFT "the same objective."

## 7. Sequence length

Qwen SFT was trained at `max_seq_length=1024`, which truncated some
assistant turns (report §"Two defects"). For Gemma: use the smallest of 1024
and 2048 that truncates **no** training example under Gemma's tokenizer
(measured in step 0.4). Report each student's truncation rate side by side.
Retraining Qwen's SFT at 2048 is out of scope, because it would invalidate
GRPO-v1 and v2's start policy. The difference becomes a stated limitation,
not a hidden one.

## 8. Runs

All on one A100, same node pinning as the Qwen jobs, same `.sif` (no rebuild:
transformers 5.9.0 already has `gemma4`). The family is chosen with
`FAMILY=gemma4` in the existing Slurm wrappers, passed through as `--family`.

1. `gemma_compat.slurm`: §4.
2. `train_sft.slurm` with `FAMILY=gemma4`: same epochs, lr, LoRA r/α/dropout
   and batch as the Qwen SFT `run_config.json`.
3. `generate_validation`: base (`--no-adapter`) and SFT on the 468 val pairs,
   back to back on one GPU.
4. `merge_sft_adapter` → `train_grpo` with the v2 reward arguments (copy of
   `train_grpo_v2.slurm` with `--family gemma4`).
5. `eval_grpo.slurm` with `FAMILY=gemma4`, then `compare_generations
   --rewards --corpus-reference` over Qwen and Gemma runs together.

Memory: E4B in bf16 is about 16 GB of weights. GRPO's KL reference is the
same model with the adapter disabled, so there's no second copy. G=4 × 8
prompts × 512 tokens fits an 80 GB A100 with gradient checkpointing, as Qwen
does.

## 9. Testing

- `model_families`: registry lookup; Qwen resolves to today's exact values.
- LoRA regex: matched against a fixture list of real Gemma 4 module names
  (dumped by step 0 and committed to `tests/fixtures/`). It matches language
  q/k/v/o/MLP and none of the vision/audio/PLE modules.
- Mask equivalence (§6) on the tiny Qwen model.
- `build_prompt_completion`: the prompt is a prefix of `render_chat` of the
  full messages.
- Every script's `--family` default leaves its parsed config identical to
  today's (extends existing `test_defaults_match_the_spec` style tests).

## 10. Repository

This stays in the current repo on a `gemma-student` branch. The change is one
new module, a `--family` flag on five scripts and a dataset-format option.
None of the other three projects' code paths change, and Qwen defaults are
preserved bit-for-bit. A separate thesis repo isn't warranted by this change.
Revisit only if the human-evaluation tooling or the thesis LaTeX grows large
enough to clutter this repo.

## 11. Out of scope

Gemma 4 12B or 31B training; retuning hyperparameters for Gemma; retraining
Qwen SFT at 2048; the test-split final runs (a separate step that covers all
systems at once).

## 12. Risks

- **E4B checkpoint won't load text-only, or PEFT/merge fails on the PLE
  architecture.** Step 0 catches this within one job. Fall back to 12B, a
  dense encoder-free model with a simpler structure.
- **Same hyperparameters are suboptimal for Gemma.** Accepted and stated.
  Tuning one student and not the other would be the worse confound.
- **Teacher/student same family.** Stated (§2).
