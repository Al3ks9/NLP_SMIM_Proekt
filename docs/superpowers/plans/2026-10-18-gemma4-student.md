# Gemma 4 E4B Student Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing SFT → merge → GRPO-v2 → eval pipeline with `google/gemma-4-E4B-it` as the student, selected by a `--family` flag, with the Qwen path unchanged.

**Architecture:** A new `src/model_families.py` registry owns everything family-specific (hub name, dir prefix, LoRA targets, loss-masking mode, loader, decoding overrides). The five training and eval scripts take `--family {qwen3,gemma4}` (default `qwen3`) and resolve their defaults through it. Gemma's SFT uses TRL's prompt–completion format with `completion_only_loss`, because TRL 1.13 has no Gemma 4 training template. A GPU compatibility job runs first and decides whether the rest proceeds.

**Tech Stack:** transformers 5.9.0, TRL 1.13.0, PEFT 0.21.0, torch 2.6, pytest; Slurm + Singularity (`singularity/qwen-sft.sif`, no rebuild).

**Spec:** `docs/superpowers/specs/2026-10-02-gemma4-student-design.md`

## Global Constraints

- `--family qwen3` (the default everywhere) must resolve to exactly today's values: `Qwen/Qwen3-4B`, targets `['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']`, `assistant_only_loss=True`, dirs `models/qwen3-lora-sft`, `models/qwen3-sft-merged`, `models/qwen3-grpo`. All 315 existing tests stay green and unmodified.
- Gemma hub name: `google/gemma-4-E4B-it`. Dir/label tag: `gemma4-e4b`.
- Gemma LoRA targets: regex `.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$` when loaded multimodal; the plain suffix list when loaded text-only (no `language_model` prefix exists then). Measured on the real E4B config (`tests/fixtures/gemma4_e4b_config.json`): 42 layers, 18 KV-shared, regex matches 258 modules (q/o/gate/up/down × 42, k/v × 24); a plain suffix list on the multimodal model would also hit 160 `Gemma4ClippableLinear` modules in `vision_tower`/`audio_tower`.
- `per_layer_input_gate`, `per_layer_projection` and `per_layer_model_projection` are never LoRA targets.
- Gemma decoding override: `top_k=20` (Gemma ships `top_k=64, top_p=0.95`; Qwen3 ships `top_k=20`). An explicit `--top-k` on the command line always wins.
- Hyperparameters are identical to Qwen's (`slurm/train_sft.slurm`: 3 epochs, batch 4 × accum 4, lr 2e-4, LoRA r16/α32/dropout 0.05; GRPO: `slurm/train_grpo_v2.slurm` arguments). Only `--max-seq-length` may differ (spec §7).
- `run_config.json` and `merge_info.json` record `family`.

## Review Focus

1. **The LoRA target set matches zero modules** (wrong loader/regex pair). PEFT would raise late or train nothing. Expected: `build_lora_config` callers check the match count on the actual model and fail with the family name. Test in Task 1.
2. **The prompt–completion example doesn't end with Gemma's end-of-turn token** (`<turn|>`). The model would never learn to stop, and every generation would run to 512 tokens and be truncation-masked in GRPO. Test in Task 2.
3. **A family/adapter mismatch** (`--family gemma4` with a Qwen adapter or merged dir, or the reverse). Expected: a clear error naming both, not a shape mismatch deep inside PEFT. Test in Task 4.
4. **The `top_k` override silently replaces an explicit `--top-k`.** Expected: the CLI value wins. Test in Task 4.
5. **The text-only loader drops language weights** (key-prefix mismatch loading a multimodal checkpoint into `Gemma4ForCausalLM`). Expected: the compat job fails on any missing `language_model` weight rather than training on random init. Checked in Task 3.

---

### Task 1: Model-family registry

**Files:**
- Create: `src/model_families.py`
- Create: `tests/test_model_families.py`
- Already present: `tests/fixtures/gemma4_e4b_config.json` (real E4B `config.json`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class ModelFamily: key: str; hf_name: str; tag: str; target_modules: list | str; loss_masking: str; loader: str; generation_overrides: dict`
  - `FAMILIES: dict[str, ModelFamily]` with keys `'qwen3'`, `'gemma4'`
  - `get_family(key: str) -> ModelFamily` (raises `ValueError` listing the known keys)
  - `default_dirs(family) -> dict` with keys `'sft'`, `'merged'`, `'grpo'` → `MODELS / f'{tag}-lora-sft'`, `f'{tag}-sft-merged'`, `f'{tag}-grpo'`
  - `base_label(family) -> str` → `'base-' + hf_name.split('/')[-1]` (`base-Qwen3-4B`, `base-gemma-4-E4B-it`)
  - `load_causal_lm(family, path: str, **kwargs)` — `loader='auto'`: `AutoModelForCausalLM.from_pretrained(path, **kwargs)`. `loader='gemma4_text'`: if `path`'s config `model_type == 'gemma4'` (the hub checkpoint), load `Gemma4ForCausalLM` with `config=AutoConfig.from_pretrained(path).get_text_config()`; otherwise (a merged dir saved as `gemma4_text`) use `AutoModelForCausalLM`.
  - `lora_targets(family, model) -> list | str` — the configured targets, after checking they match ≥ 1 module of `model`. Otherwise `ValueError(f'{family.key}: LoRA targets {targets!r} match no module')`.
  - `count_target_matches(targets, module_names: list) -> Counter` (suffix → count), using PEFT's own matching rule: a str is a full-match regex, a list is suffix match on `.`-boundaries.
- Gemma's `loader` and `target_modules` start as `'gemma4_text'` and the suffix list. Task 3's result may flip them to `'auto'` plus the regex. That's a one-line change in `FAMILIES` and doesn't affect any test below.

- [ ] **Step 1: Write the failing tests**

```python
def test_qwen_resolves_to_todays_exact_values():
    f = mf.get_family('qwen3')
    assert f.hf_name == 'Qwen/Qwen3-4B'
    assert f.target_modules == ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    assert f.loss_masking == 'assistant_only' and f.generation_overrides == {}
    assert mf.default_dirs(f) == {'sft': MODELS / 'qwen3-lora-sft', 'merged': MODELS / 'qwen3-sft-merged',
                                  'grpo': MODELS / 'qwen3-grpo'}
    assert mf.base_label(f) == 'base-Qwen3-4B'

def test_gemma_family_values():
    f = mf.get_family('gemma4')
    assert f.hf_name == 'google/gemma-4-E4B-it' and f.tag == 'gemma4-e4b'
    assert f.loss_masking == 'completion_only' and f.generation_overrides == {'top_k': 20}

def test_unknown_family_lists_known_keys():
    with pytest.raises(ValueError, match='qwen3.*gemma4|gemma4.*qwen3'):
        mf.get_family('llama')

GEMMA_REGEX = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'

def test_regex_matches_only_language_model_on_real_e4b_config():
    names = _meta_module_names_from_fixture()   # Gemma4ForConditionalGeneration(config) under torch.device('meta')
    counts = mf.count_target_matches(GEMMA_REGEX, names)
    assert counts == {'q_proj': 42, 'o_proj': 42, 'gate_proj': 42, 'up_proj': 42, 'down_proj': 42,
                      'k_proj': 24, 'v_proj': 24}

def test_plain_suffixes_on_multimodal_e4b_would_hit_the_towers():
    names = _meta_module_names_from_fixture()
    hits = [n for n in names if mf.count_target_matches(mf.get_family('qwen3').target_modules, [n])]
    assert any('vision_tower' in n for n in hits) and any('audio_tower' in n for n in hits)

def test_per_layer_projections_are_never_targets():
    names = _meta_module_names_from_fixture()
    for targets in (GEMMA_REGEX, mf.get_family('gemma4').target_modules):
        assert not any('per_layer' in n for n in names if mf.count_target_matches(targets, [n]))

def test_lora_targets_refuses_a_target_set_that_matches_nothing():   # Review Focus 1
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match='qwen3.*match no module'):
        mf.lora_targets(mf.get_family('qwen3'), model)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_families.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'model_families'`

- [ ] **Step 3: Implement `src/model_families.py`** per the Interfaces block. `load_causal_lm` imports transformers lazily. `count_target_matches` returns a `collections.Counter` keyed on the matched module's last name component.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_families.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/model_families.py tests/test_model_families.py tests/fixtures/gemma4_e4b_config.json
git commit -m "Add model-family registry for the Qwen and Gemma students"
```

---

### Task 2: Prompt–completion dataset format

**Files:**
- Modify: `src/sft_data.py` (add after `build_hf_dataset`)
- Test: `tests/test_sft_data.py`

**Interfaces:**
- Consumes: `render_inference_prompt(tokenizer, source_poem, target_author)`, `render_chat(tokenizer, messages, add_generation_prompt)`, `build_messages(...)`
- Produces:
  - `end_of_turn(tokenizer) -> str`: the text the template emits right after an assistant turn's content. Derived by rendering `[system, user, assistant='X']` and taking what follows `'X'` up to (not including) a trailing newline. Qwen: `<|im_end|>`, Gemma: `<turn|>`.
  - `build_prompt_completion(tokenizer, source_poem, target_author, generated_poem) -> dict` → `{'prompt': render_inference_prompt(...), 'completion': generated_poem.strip() + end_of_turn(tokenizer)}`
  - `build_hf_dataset(rows, fmt='messages', tokenizer=None)`: `fmt='prompt_completion'` returns columns `prompt`, `completion` and requires `tokenizer`. The default behaviour is unchanged.

- [ ] **Step 1: Write the failing tests** (use the tiny Qwen already used in `tests/test_train_grpo.py`: `trl-internal-testing/tiny-Qwen3ForCausalLM`)

```python
def test_prompt_is_a_strict_prefix_of_the_full_chat(tiny_tok):
    pc = sd.build_prompt_completion(tiny_tok, 'извор\nред', 'Блаже Конески', 'нова песна')
    full = sd.render_chat(tiny_tok, sd.build_messages('извор\nред', 'Блаже Конески', 'нова песна'))
    assert full.startswith(pc['prompt']) and len(pc['prompt']) < len(full)

def test_completion_ends_with_the_templates_end_of_turn(tiny_tok):       # Review Focus 2
    pc = sd.build_prompt_completion(tiny_tok, 'извор', 'Блаже Конески', 'нова песна')
    assert sd.end_of_turn(tiny_tok) == '<|im_end|>'
    assert pc['completion'] == 'нова песна<|im_end|>'

def test_both_formats_train_on_exactly_the_same_tokens(tiny_tok):
    # assistant_only_loss on messages vs completion_only_loss on prompt+completion:
    # the set of loss-bearing token ids must be identical, or the two students
    # were not trained on the same objective
    a = _trainable_ids(tiny_tok, fmt='messages')            # via trl's own collator/tokenisation
    b = _trainable_ids(tiny_tok, fmt='prompt_completion')
    assert a == b and a

def test_messages_format_is_still_the_default():
    ds = sd.build_hf_dataset(_rows())
    assert ds.column_names == ['messages']
```

`_trainable_ids` runs TRL 1.13's `SFTTrainer` dataset preparation (`_prepare_dataset`) with the matching `SFTConfig` flag, then returns the input ids where the assistant/completion mask is 1. If that private method isn't reachable, apply `trl.chat_template_utils.get_training_chat_template` for the messages side and tokenize prompt+completion directly for the other.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sft_data.py -v -k "prefix or end_of_turn or same_tokens or still_the_default"`
Expected: FAIL, `AttributeError: module 'sft_data' has no attribute 'build_prompt_completion'`

- [ ] **Step 3: Implement** `end_of_turn`, `build_prompt_completion` and the `fmt` parameter in `src/sft_data.py`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_sft_data.py -v`
Expected: all pass, including the pre-existing tests

- [ ] **Step 5: Commit**

```bash
git add src/sft_data.py tests/test_sft_data.py
git commit -m "Add prompt-completion SFT format with a mask-equivalence test"
```

---

### Task 3: GPU compatibility check

**Files:**
- Create: `src/gemma_compat.py`
- Create: `slurm/gemma_compat.slurm` (headers copied from `slurm/train_grpo.slurm`, `--job-name=gemma-compat`, uses `slurm/grpo_common.sh`'s `run_in_container`)
- Test: `tests/test_gemma_compat.py`

**Interfaces:**
- Consumes: `model_families` (Task 1), `sft_data.build_prompt_completion` / `end_of_turn` (Task 2)
- Produces: `data/gemma_compat_report.json` (tracked) with keys `loader_used`, `missing_keys`, `unexpected_keys`, `lora_match_counts`, `end_of_turn`, `train_token_lengths` (`{p50, p95, p99, max, n_over_1024, n_over_2048}`), `sample_generation`, `checks` (`{name: 'pass' | 'fail: <reason>'}`). Exit 1 if any check fails.

Checks, in order. Each one is a function returning `(name, ok, detail)`, so the CPU-only parts are unit-testable:

1. `check_load`: the configured loader. Fail if any missing key contains `language_model` or `layers.`. On failure, retry with `loader='auto'` and record `loader_used='auto'` (Review Focus 5).
2. `check_lora_targets`: `count_target_matches` on the loaded model. Fail on zero matches, or if any match lies under `vision_tower`/`audio_tower` or contains `per_layer`.
3. `check_template`: the prompt is a prefix of the full chat, the completion ends with `end_of_turn`, and `'<|channel>thought'` is absent from the prompt.
4. `check_lengths`: Gemma-tokenized length of prompt+completion for all `split=train` rows of `synthetic_dataset.csv`. Always passes, and reports the distribution (spec §7 input).
5. `check_generation`: one sample for the first val pair at T=0.7, top_p=0.9, top_k=20, 512 tokens. Fail if empty or it contains `thought`.
6. `check_sft_merge_roundtrip`: 3 SFT steps on 8 rows (`completion_only_loss=True`) → save the adapter → merge → assert `abs(logprob_merged - logprob_adapter) < 0.5` on a fixed completion → `AutoModelForCausalLM.from_pretrained(merged_dir)` reloads.

The Slurm job then runs `train_grpo.py --smoke --family gemma4 --sft-merged <tmp merged dir>`.

- [ ] **Step 1: Write the failing tests** (CPU, fake model objects)

```python
def test_check_load_fails_on_missing_language_weights():
    ok, detail = compat.evaluate_load(missing_keys=['model.language_model.layers.0.mlp.up_proj.weight'],
                                      unexpected_keys=[])
    assert not ok and 'language_model' in detail

def test_check_lora_targets_rejects_tower_and_per_layer_matches():
    ok, _ = compat.evaluate_lora_matches(['model.vision_tower.encoder.layers.0.self_attn.q_proj'])
    assert not ok
    ok, _ = compat.evaluate_lora_matches(['model.language_model.layers.0.self_attn.q_proj'])
    assert ok

def test_report_exit_code_reflects_any_failed_check(tmp_path):
    assert compat.write_report({'checks': {'a': 'pass', 'b': 'fail: x'}}, tmp_path / 'r.json') == 1
    assert compat.write_report({'checks': {'a': 'pass'}}, tmp_path / 'r.json') == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_gemma_compat.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'gemma_compat'`

- [ ] **Step 3: Implement** `evaluate_load(missing_keys, unexpected_keys) -> (bool, str)`, `evaluate_lora_matches(matched_names) -> (bool, str)`, `write_report(report, path) -> int`, the six GPU check functions and `main()`, plus `slurm/gemma_compat.slurm`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_gemma_compat.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit, then the user runs it on the HPC**

```bash
git add src/gemma_compat.py slurm/gemma_compat.slurm tests/test_gemma_compat.py
git commit -m "Add Gemma 4 E4B compatibility check job"
```
HPC: `sbatch slurm/gemma_compat.slurm`. **Decision point:**
- All pass with `loader_used='gemma4_text'`: keep Task 1's Gemma entry.
- Pass only with `'auto'`: set Gemma's `loader='auto'` and `target_modules` to the regex in `FAMILIES`, and commit that.
- Load or merge fails under both loaders: stop and report. The fallback is Gemma 4 12B (spec §12).

Set Gemma's SFT `--max-seq-length` from `train_token_lengths`: 1024 if `n_over_1024 == 0`, else 2048.

---

### Task 4: `--family` across the five scripts

**Files:**
- Modify: `src/train_sft.py` (`load_model_and_tokenizer`, `train`, `parse_args`)
- Modify: `src/merge_sft_adapter.py` (`merge`, CLI)
- Modify: `src/generate_validation.py` (`load_adapter_model`, `_run_label`, `run_validation`, CLI)
- Modify: `src/train_grpo.py` (`load_policy`, `parse_args`, `write_run_config`)
- Modify: `src/evaluate_reward.py` (`build_parser`: `--merged-model` default)
- Test: `tests/test_train_sft.py`, `tests/test_merge_sft_adapter.py`, `tests/test_generate_validation.py`, `tests/test_train_grpo.py`

**Interfaces:**
- Consumes: Task 1 (`get_family`, `default_dirs`, `base_label`, `load_causal_lm`, `lora_targets`), Task 2 (`build_hf_dataset(fmt=..., tokenizer=...)`)
- Produces: `--family {qwen3,gemma4}` on every script (default `qwen3`). Path and model arguments default to `None` and resolve through the family. `train_sft`: `loss_masking == 'completion_only'` → `build_hf_dataset(rows, fmt='prompt_completion', tokenizer=tok)` plus `SFTConfig(completion_only_loss=True, assistant_only_loss=False)`. `generate_validation`: `top_k = args.top_k if args.top_k is not None else family.generation_overrides.get('top_k')`. Every `run_config.json` / `merge_info.json` gains `'family'`.
- Family check: `assert_family_matches(family, model_dir: Path)` in `model_families` reads `adapter_config.json`'s `base_model_name_or_path`, or `merge_info.json`'s `family`. If either names the other family, it raises `ValueError(f'--family {family.key} but {model_dir} was built from {other}')`.

- [ ] **Step 1: Write the failing tests**

```python
# test_train_sft.py
def test_family_default_leaves_qwen_config_unchanged():
    a = ts.parse_args([])
    assert ts.resolve(a).model_name == 'Qwen/Qwen3-4B' and ts.resolve(a).output_dir == MODELS / 'qwen3-lora-sft'

def test_gemma_family_uses_completion_only_loss(monkeypatch):
    cfg = ts.build_sft_config(ts.resolve(ts.parse_args(['--family', 'gemma4', '--no-bf16'])))
    assert cfg.completion_only_loss is True and cfg.assistant_only_loss is False

# test_generate_validation.py
def test_gemma_top_k_override_and_cli_wins():                        # Review Focus 4
    assert gv.effective_top_k(get_family('gemma4'), cli_top_k=None) == 20
    assert gv.effective_top_k(get_family('gemma4'), cli_top_k=50) == 50
    assert gv.effective_top_k(get_family('qwen3'), cli_top_k=None) is None

def test_gemma_base_label():
    assert gv._run_label(None, None, family=get_family('gemma4')) == 'base-gemma-4-E4B-it'

# test_merge_sft_adapter.py / test_train_grpo.py
def test_family_mismatch_is_refused(tmp_path):                        # Review Focus 3
    (tmp_path / 'merge_info.json').write_text('{"family": "qwen3"}')
    with pytest.raises(ValueError, match='--family gemma4 .* qwen3'):
        mf.assert_family_matches(get_family('gemma4'), tmp_path)

def test_grpo_gemma_defaults():
    a = tg.resolve(tg.parse_args(['--family', 'gemma4']))
    assert a.sft_merged == MODELS / 'gemma4-e4b-sft-merged' and a.output_dir == MODELS / 'gemma4-e4b-grpo'
```

`resolve(args) -> args` is a new helper in `train_sft` / `train_grpo` that fills `None` defaults from the family. `build_sft_config(args) -> SFTConfig` is extracted from `train_sft.train`. `effective_top_k(family, cli_top_k) -> int | None` lives in `generate_validation`. `_run_label` gains a `family=` keyword argument, defaulting to qwen3.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_train_sft.py tests/test_generate_validation.py tests/test_merge_sft_adapter.py tests/test_train_grpo.py -v`
Expected: the new tests FAIL (`unrecognized arguments: --family` / missing attributes), and the old tests pass

- [ ] **Step 3: Implement** the Interfaces block. Replace each direct `AutoModelForCausalLM.from_pretrained` / `AutoTokenizer.from_pretrained` on a hub base with `mf.load_causal_lm` / `AutoTokenizer`. Pass `lora_targets(family, model)` into `build_lora_config`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q tests/`
Expected: all pass (315 existing plus the new ones)

- [ ] **Step 5: Commit**

```bash
git add src/train_sft.py src/merge_sft_adapter.py src/generate_validation.py src/train_grpo.py src/evaluate_reward.py src/model_families.py tests/
git commit -m "Add --family to the SFT, merge, generation and GRPO scripts"
```

---

### Task 5: Slurm wiring and docs

**Files:**
- Modify: `slurm/train_sft.slurm`, `slurm/compare_base_lora.slurm`, `slurm/eval_grpo.slurm`: honour `FAMILY` (default `qwen3`) as `--family "$FAMILY"`. `compare_base_lora.slurm` and `eval_grpo.slurm` derive CSV names from `mf.base_label` / `default_dirs` (via a one-line `python -c` call) instead of hardcoded `qwen3` paths.
- Create: `slurm/train_grpo_gemma.slurm`: `train_grpo_v2.slurm`'s exact reward arguments plus `--family gemma4`, writing `models/gemma4-e4b-grpo/`.
- Modify: `slurm/grpo_common.sh` `grpo_preflight`: check for the family's SFT dir, not the hardcoded `models/qwen3-lora-sft`.
- Modify: `CLAUDE.md` (new "Gemma 4 student" subsection after the GRPO stage)

- [ ] **Step 1: Dry-run each wrapper's command assembly** with the `printf` stand-in for `$PY` (the same technique used to check `train_grpo_v2.slurm`'s JSON quoting), for `FAMILY=qwen3` and `FAMILY=gemma4`.
Expected: the qwen3 commands are byte-identical to today's; the gemma4 ones carry `--family gemma4` and `gemma4-e4b-*` paths.

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -q tests/`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add slurm/ CLAUDE.md
git commit -m "Wire FAMILY through the Slurm jobs; add Gemma GRPO job"
```

HPC run order (user): `FAMILY=gemma4 sbatch slurm/train_sft.slurm --max-seq-length <from Task 3>`, then `FAMILY=gemma4 sbatch slurm/compare_base_lora.slurm`, then `sbatch slurm/train_grpo_gemma.slurm`, then `FAMILY=gemma4 ADAPTER=models/gemma4-e4b-grpo sbatch slurm/eval_grpo.slurm`.
