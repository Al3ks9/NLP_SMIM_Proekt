"""
Gemma 4 E4B compatibility check (GPU, ~20 min): run before any Gemma training.
Spec: docs/superpowers/specs/2026-10-02-gemma4-student-design.md §4.

Checks, in order, each recorded as 'pass' or 'fail: <reason>' in
data/gemma_compat_report.json (tracked: it decides the loader, the LoRA target
form and Gemma's --max-seq-length):

1. load        the configured loader keeps every language-model weight
               (model_families.load_causal_lm raises otherwise; then the
               multimodal 'auto' loader is tried for the report only -- a
               failing text-only load fails the job, and FAMILIES needs a
               human edit)
2. lora        the LoRA targets match language-model attention/MLP only
3. template    prompt is a prefix of the full chat; completion ends with the
               template's end-of-turn token; no thinking channel in the prompt
4. lengths     Gemma-tokenized length of every train example (always passes;
               picks max_seq_length = smallest of 1024/2048/4096 truncating none)
5. generate    one untuned sample is non-empty and has no thinking channel
6. sft_merge   3 SFT steps -> merge; the fp32 merge must reproduce the fp32
               adapter's log-prob (the merge is correct), and the saved merge
               reloads through AutoModelForCausalLM. The bf16 merge's gap is
               recorded, not gated: bf16 storage loses part of a 3-step
               adapter's weight change (scripts/diagnose_merge.py)

Exit 1 if any check fails. slurm/gemma_compat.slurm then runs
train_grpo.py --smoke on the merged model.

Usage (inside the container, see slurm/gemma_compat.slurm):
    python src/gemma_compat.py [--work-dir models/gemma4-compat]
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / 'data' / 'gemma_compat_report.json'

LIMITS = (1024, 2048, 4096)   # 10 of 1,146 train rows exceed 2048 Gemma tokens (max 3,371)
MERGE_TOLERANCE_PER_TOKEN = 0.02   # bf16 merged vs unmerged, mean |delta log p| per completion token
TOWERS = ('vision_tower', 'audio_tower', 'embed_vision', 'embed_audio')


def evaluate_load(missing_keys: list, unexpected_keys: list) -> tuple:
    """A text-only load may leave tower weights unused (unexpected), but must
    not leave any language-model weight missing -- that would be random init."""
    lost = [k for k in missing_keys if 'language_model' in k or '.layers.' in k
            or k.startswith('layers.') or 'embed_tokens' in k]
    if lost:
        return False, f'{len(lost)} language-model weights missing, e.g. {lost[0]}'
    return True, f'{len(missing_keys)} missing / {len(unexpected_keys)} unexpected keys, none language-model'


def evaluate_lora_matches(matched_names: list) -> tuple:
    if not matched_names:
        return False, 'LoRA targets match no module'
    bad = [n for n in matched_names if any(t in n for t in TOWERS) or 'per_layer' in n]
    if bad:
        return False, f'{len(bad)} matches outside language-model attention/MLP, e.g. {bad[0]}'
    return True, f'{len(matched_names)} modules'


def length_summary(lengths: list) -> dict:
    s = sorted(lengths)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return {'n': len(s), 'p50': pick(0.5), 'p95': pick(0.95), 'p99': pick(0.99), 'max': s[-1],
            **{f'n_over_{limit}': sum(l > limit for l in s) for limit in LIMITS}}


def choose_max_seq_length(summary: dict):
    """Smallest limit that truncates no training example, None if even the
    largest truncates some (then the spec's §7 choice needs a human decision)."""
    for limit in LIMITS:
        if summary[f'n_over_{limit}'] == 0:
            return limit
    return None


def merge_agrees(lp_adapter: float, lp_merged: float, n_tokens: int) -> tuple:
    """Merged vs adapter model on one completion, judged per token: a summed
    log-prob over ~300 bf16 tokens drifts by more than any fixed total."""
    per_token = abs(lp_merged - lp_adapter) / max(n_tokens, 1)
    return (per_token < MERGE_TOLERANCE_PER_TOKEN,
            f'adapter {lp_adapter:.3f} vs merged {lp_merged:.3f} over {n_tokens} tokens '
            f'({per_token:.4f}/token, tolerance {MERGE_TOLERANCE_PER_TOKEN})')


def write_report(report: dict, path: Path = REPORT_PATH) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return 1 if any(str(v).startswith('fail') for v in report['checks'].values()) else 0


# ── GPU checks (run by main(); not unit-tested -- they need the real model) ────

def _record(report: dict, name: str, ok: bool, detail: str) -> bool:
    report['checks'][name] = 'pass' if ok else f'fail: {detail}'
    report.setdefault('details', {})[name] = detail
    return ok


def check_load(report: dict, family) -> None:
    """The configured loader must keep every language-model weight
    (model_families.load_causal_lm raises if not). On failure the multimodal
    'auto' loader is tried as well, so the report says whether switching
    FAMILIES['gemma4'] to it would work -- but the check still fails: the
    production scripts use FAMILIES as committed, not this process's view."""
    import dataclasses
    import torch
    import model_families as mf

    report['loader_used'] = family.loader
    try:
        mf.load_causal_lm(family, family.hf_name, dtype=torch.bfloat16)
        report['missing_keys'] = []
        _record(report, 'load', True, f'{family.loader}: every language-model weight loaded')
        return
    except RuntimeError as exc:
        detail = f'{family.loader}: {exc}'
    auto = dataclasses.replace(family, loader='auto', target_modules=mf.GEMMA4_LM_REGEX)
    _, info = mf.load_causal_lm(auto, auto.hf_name, dtype=torch.bfloat16, output_loading_info=True)
    ok, auto_detail = evaluate_load(sorted(info['missing_keys']), sorted(info['unexpected_keys']))
    report['missing_keys'] = sorted(info['missing_keys'])
    _record(report, 'load', False,
            f'{detail}; auto loader would {"work" if ok else "also fail"} ({auto_detail}) -- '
            "set FAMILIES['gemma4'] loader='auto', target_modules=GEMMA4_LM_REGEX if it works")


def check_lora(report: dict, family) -> None:
    import torch
    import model_families as mf
    model = mf.load_causal_lm(family, family.hf_name, dtype=torch.bfloat16)
    names = [n for n, _ in model.named_modules()]
    matched = [n for n in names if mf.count_target_matches(family.target_modules, [n])]
    report['lora_match_counts'] = dict(mf.count_target_matches(family.target_modules, names))
    _record(report, 'lora', *evaluate_lora_matches(matched))


def check_template(report: dict, tokenizer) -> None:
    import sft_data
    source, target, poem = 'извор\nред', 'Блаже Конески', 'нова песна\nвтор стих'
    pc = sft_data.build_prompt_completion(tokenizer, source, target, poem)
    full = sft_data.render_chat(tokenizer, sft_data.build_messages(source, target, poem))
    eot = sft_data.end_of_turn(tokenizer)
    report['end_of_turn'] = eot
    problems = []
    if not full.startswith(pc['prompt']):
        problems.append('prompt is not a prefix of the full chat')
    if not eot or not pc['completion'].endswith(eot):
        problems.append(f'completion does not end with end-of-turn {eot!r}')
    if 'thought' in pc['prompt']:
        problems.append('thinking channel in the inference prompt')
    _record(report, 'template', not problems, '; '.join(problems) or f'end_of_turn={eot!r}')


def check_lengths(report: dict, tokenizer) -> None:
    import sft_data
    rows = sft_data.load_synthetic_rows(split='train')
    lengths = []
    for source, target, poem in sft_data._source_target_poem(rows):
        pc = sft_data.build_prompt_completion(tokenizer, source, target, poem)
        lengths.append(len(tokenizer(pc['prompt'] + pc['completion'],
                                     add_special_tokens=False)['input_ids']))
    summary = length_summary(lengths)
    report['train_token_lengths'] = summary
    report['max_seq_length'] = choose_max_seq_length(summary)
    _record(report, 'lengths', True, f"max_seq_length={report['max_seq_length']} ({summary})")


def check_generation(report: dict, family, tokenizer) -> None:
    import torch
    import model_families as mf
    import sft_data
    from make_splits import poems_in_split

    poem = sorted(poems_in_split('val'), key=lambda p: int(p['poem_id']))[0]
    target = sft_data.select_target_authors(10)[0]
    prompt = sft_data.render_inference_prompt(tokenizer, poem['song_text'], target)
    model = mf.load_causal_lm(family, family.hf_name, dtype=torch.bfloat16).to('cuda').eval()
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
    torch.manual_seed(42)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7,
                             top_p=0.9, top_k=family.generation_overrides.get('top_k'))
    raw = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    clean = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    report['sample_generation'] = {'source_poem_id': poem['poem_id'], 'target_author': target,
                                   'raw': raw}
    problems = (['empty'] if not clean else []) + (['thinking channel'] if 'thought' in raw else [])
    _record(report, 'generate', not problems, '; '.join(problems) or f'{len(clean)} chars')


def check_sft_merge(report: dict, family, work_dir: Path, max_seq_length: int) -> Path:
    """3 SFT steps on 8 train rows -> merge -> merged and adapter models give
    the same log-prob for a fixed completion -> merged reloads via Auto."""
    import csv
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import generate_validation as gv
    import merge_sft_adapter as msa
    import model_families as mf
    import sft_data
    import train_sft
    from train_grpo import completion_logprob

    rows = sft_data.load_synthetic_rows(split='train')[:8]
    data = work_dir / 'synthetic_8.csv'
    data.parent.mkdir(parents=True, exist_ok=True)
    with open(data, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sft_dir = train_sft.train(train_sft.parse_args([
        '--family', family.key, '--synthetic-data', str(data), '--output-dir', str(work_dir / 'sft'),
        '--eval-split', '', '--num-train-epochs', '1.5', '--per-device-train-batch-size', '4',
        '--gradient-accumulation-steps', '1', '--save-strategy', 'no', '--logging-steps', '1',
        '--max-seq-length', str(max_seq_length)]))
    merged_dir = msa.merge(sft_dir, work_dir / 'merged', family=family.key, force=True)

    tokenizer = AutoTokenizer.from_pretrained(str(sft_dir))
    source, target, poem = sft_data._source_target_poem(rows[:1])[0]
    pc = sft_data.build_prompt_completion(tokenizer, source, target, poem)
    n_tokens = len(tokenizer(pc['completion'], add_special_tokens=False)['input_ids'])
    lp = lambda model: completion_logprob(model, tokenizer, pc['prompt'], pc['completion'])

    # Gate: the merge itself, in fp32 -- merged and adapter must agree.
    from peft import PeftModel
    fp32 = PeftModel.from_pretrained(mf.load_causal_lm(family, family.hf_name, dtype=torch.float32),
                                     str(sft_dir)).to('cuda').eval()
    lp_adapter32 = lp(fp32)
    lp_merged32 = lp(fp32.merge_and_unload())
    del fp32
    torch.cuda.empty_cache()

    # Information only: what bf16 storage costs. On the first compat run a
    # 3-step Gemma adapter kept ~85 % of its log-prob shift after a bf16 merge
    # (Qwen's full SFT merge loses ~0); scripts/diagnose_merge.py measures it
    # on the real adapter before GRPO.
    adapter_model, _ = gv.load_adapter_model(sft_dir, family=family)
    lp_adapter16 = lp(adapter_model)
    del adapter_model
    torch.cuda.empty_cache()
    merged = AutoModelForCausalLM.from_pretrained(str(merged_dir), dtype=torch.bfloat16).to('cuda')
    lp_merged16 = lp(merged)
    report['merge_logprob'] = {
        'n_tokens': n_tokens, 'adapter_fp32': lp_adapter32, 'merged_fp32': lp_merged32,
        'adapter_bf16': lp_adapter16, 'merged_saved_bf16': lp_merged16,
        'bf16_merge_gap_per_token': (lp_merged16 - lp_adapter16) / n_tokens}
    _record(report, 'sft_merge', *merge_agrees(lp_adapter32, lp_merged32, n_tokens))
    return merged_dir


def main(argv=None) -> int:
    import argparse
    import logging
    import traceback
    from transformers import AutoTokenizer
    import model_families as mf

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--work-dir', type=Path, default=ROOT / 'models' / 'gemma4-compat')
    p.add_argument('--report', type=Path, default=REPORT_PATH)
    args = p.parse_args(argv)

    family = mf.get_family('gemma4')
    report = {'family': family.key, 'hf_name': family.hf_name, 'checks': {}}
    tokenizer = AutoTokenizer.from_pretrained(family.hf_name)
    steps = [
        ('load', lambda: check_load(report, family)),
        ('lora', lambda: check_lora(report, family)),
        ('template', lambda: check_template(report, tokenizer)),
        ('lengths', lambda: check_lengths(report, tokenizer)),
        ('generate', lambda: check_generation(report, family, tokenizer)),
        ('sft_merge', lambda: check_sft_merge(report, family, args.work_dir,
                                              report.get('max_seq_length') or 4096)),
    ]
    for name, step in steps:
        try:
            step()
        except Exception as exc:   # record and keep going: one report with every finding
            _record(report, name, False, f'{type(exc).__name__}: {exc}')
            report.setdefault('tracebacks', {})[name] = traceback.format_exc()
        print(f'{name:10s} {report["checks"].get(name)}', flush=True)
    return write_report(report, args.report)


if __name__ == '__main__':
    raise SystemExit(main())
