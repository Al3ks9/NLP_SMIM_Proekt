"""
Standalone GRPO reward inspector -- run it before any long GRPO job to check
that the reward behaves sensibly.

Single generation:
    uv run python src/evaluate_reward.py --source-poem-id 301 \\
        --target-author "Адем Гајтани" --generated-file poem.txt
    uv run python src/evaluate_reward.py \\
        --generations-csv data/validation_generations/qwen3-lora-sft_val.csv --row 0

Hand-built fixture cases (tests/fixtures/reward_cases.json), with ordering
checks; exits 1 if an expectation fails:
    uv run python src/evaluate_reward.py --fixtures

Within-group reward spread from the SFT policy (GPU; see --groups help).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import grpo_rewards as gr
from make_splits import load_stripped_songs

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
FIXTURES_PATH = ROOT / 'tests' / 'fixtures' / 'reward_cases.json'
SYNTHETIC_PATH = DATA / 'synthetic' / 'synthetic_dataset.csv'

GROUP_KEYS = ['R', 'R_style_g', 'R_content_g', 'R_style', 'R_content'] + \
    [f'group/{g}' for g in gr.STYLE_GROUPS] + \
    ['gate/validity', 'gate/line_uniqueness', 'gate/copy_novelty']
DEFAULT_GROUPS_OUT = DATA / 'grpo' / 'reward_groups.csv'

MARKDOWN_ENGLISH = ('**Here is the poem rewritten in the requested style:**\n\n'
                    '- The sun rises over the mountain\n- A bird sings in the forest')


def _poem(poem_id: str) -> dict:
    return next(r for r in load_stripped_songs() if r['poem_id'] == poem_id)


def _near_copy(text: str) -> str:
    """Source with the last word of every line of >= 4 words replaced -- the
    'changed one word per line' copy the copy gate must still catch."""
    out = []
    for line in text.splitlines():
        words = line.split()
        out.append(' '.join(words[:-1] + ['ноќта']) if len(words) >= 4 else line)
    return '\n'.join(out)


def _gemma(source_poem_id: str, target_author: str) -> str:
    with open(SYNTHETIC_PATH, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r['source_poem_id'], r['target_author'], r['sample_index']) == (
                    source_poem_id, target_author, '0'):
                return r['generated_poem']
    raise KeyError(f'no Gemma row for ({source_poem_id}, {target_author})')


def build_candidates(case: dict) -> dict:
    source = _poem(case['source_poem_id'])['song_text']
    first_line = next(l for l in source.splitlines() if l.strip())
    return {
        'real_target': _poem(case['target_poem_id'])['song_text'],
        'verbatim_source': source,
        'near_copy': _near_copy(source),
        'gemma': _gemma(case['source_poem_id'], case['target_author']),
        'repetition': '\n'.join([first_line] * 12),
        'markdown_english': MARKDOWN_ENGLISH,
        'empty': '',
    }


def check_expectations(scores: dict) -> list:
    """Ordering checks from the spec §3.6. Returns human-readable failures."""
    s = scores
    checks = [
        (s['verbatim_source']['R_content_g'] == 0.0, 'verbatim_source: gated content should be 0'),
        (s['near_copy']['R_content_g'] <= 0.3 * max(s['near_copy']['R_content'], 1e-9),
         'near_copy: copy gate should remove >= 70% of its content reward'),
        (s['real_target']['R_style'] > s['verbatim_source']['R_style'],
         'real_target: style should beat verbatim_source (target voice vs source voice)'),
        (s['verbatim_source']['R_content'] > s['real_target']['R_content'],
         'content scorer: raw content of the source should beat an unrelated poem'),
        (all(s[k]['R'] < s['gemma']['R'] for k in ('repetition', 'markdown_english', 'empty')),
         'gemma should out-score repetition / markdown_english / empty'),
        (s['markdown_english']['R'] == 0.0 and s['empty']['R'] == 0.0,
         'markdown_english and empty should be zeroed by the validity gate'),
    ]
    return [msg for ok, msg in checks if not ok]


def format_report(row: dict, cfg: gr.RewardConfig) -> str:
    lines = [
        f"Style reward:   {row['R_style']:.3f}   (gated: {row['R_style_g']:.3f})",
        f"Content reward: {row['R_content']:.3f}   (gated: {row['R_content_g']:.3f})",
        f"Total reward:   {row['R']:.3f}   "
        f"(= {cfg.style_weight} * style_g + {cfg.content_weight} * content_g)",
        '',
        'Gates: ' + '  '.join(f"{k[5:]}={row[k]:.2f}" for k in row if k.startswith('gate/')),
        f"Content: precision={row['content/precision']:.3f} recall={row['content/recall']:.3f} "
        f"f1={row['content/f1']:.3f}",
        '',
        f"{'group':<18}{'weight':>7}{'score':>7}   features (raw -> z -> score)",
    ]
    for group, members in gr.STYLE_GROUPS.items():
        parts = []
        for m in members:
            z = f" z={row[f'z/{m}']:+.2f}" if f'z/{m}' in row else ''
            parts.append(f"{m}={row[f'raw/{m}']:.3f}{z} -> {row[f'score/{m}']:.2f}")
        lines.append(f"{group:<18}{cfg.style_weights.get(group, 0.0):>7.2f}"
                     f"{row[f'group/{group}']:>7.3f}   " + '; '.join(parts))
    return '\n'.join(lines)


def run_fixtures(cfg: gr.RewardConfig) -> int:
    cases = json.loads(FIXTURES_PATH.read_text(encoding='utf-8'))
    failed = 0
    for case in cases:
        cands = build_candidates(case)
        source = _poem(case['source_poem_id'])['song_text']
        rows = gr.score_batch([source] * len(cands), list(cands.values()),
                              [case['target_author']] * len(cands), cfg)
        scores = dict(zip(cands, rows))
        print(f"\n{case['source_author']} (poem {case['source_poem_id']}) -> {case['target_author']}")
        print(f"  {'candidate':<18}{'style':>7}{'style_g':>9}{'content':>9}{'content_g':>11}{'R':>7}")
        for name, r in scores.items():
            print(f"  {name:<18}{r['R_style']:>7.3f}{r['R_style_g']:>9.3f}"
                  f"{r['R_content']:>9.3f}{r['R_content_g']:>11.3f}{r['R']:>7.3f}")
        for msg in check_expectations(scores):
            print(f'  FAIL: {msg}')
            failed += 1
    print(f'\n{failed} expectation(s) failed' if failed else '\nall expectations hold')
    return 1 if failed else 0


def run_single(args, cfg: gr.RewardConfig) -> int:
    if args.generations_csv:
        with open(args.generations_csv, encoding='utf-8') as f:
            row = list(csv.DictReader(f))[args.row]
        source = _poem(row['source_poem_id'])['song_text']
        target, generated = row['target_author'], row['generated_poem']
    else:
        source = (_poem(args.source_poem_id)['song_text'] if args.source_poem_id
                  else Path(args.source_file).read_text(encoding='utf-8'))
        target = args.target_author
        generated = Path(args.generated_file).read_text(encoding='utf-8')
    print(format_report(gr.score(source, generated, target, cfg), cfg))
    return 0


def summarise_groups(rows_by_prompt: list, keys: list) -> dict:
    """Under GRPO's per-group advantage normalisation, what decides which
    component drives the update is its spread *within* a group of samples for
    one prompt, not its mean -- so that is what alpha/beta are set from."""
    out = {}
    for k in keys:
        per_group = [[r[k] for r in rows] for rows in rows_by_prompt]
        out[k] = {
            'mean': float(np.mean([v for g in per_group for v in g])),
            'within_group_std': float(np.mean([np.std(g) for g in per_group])),
            'between_prompt_std': float(np.std([np.mean(g) for g in per_group])),
        }
    return out


def grpo_sampling_kwargs() -> dict:
    """generate() settings matching train_grpo's GRPO defaults. The within-
    group spread only predicts training if the samples come from the same
    distribution -- validation generation's top_p=0.9 plus Qwen3's shipped
    top_k=20 is noticeably narrower than GRPOTrainer's top_p=1.0 / top_k=0."""
    import train_grpo
    d = train_grpo.parse_args([])
    return {'max_new_tokens': d.max_completion_length, 'temperature': d.temperature,
            'top_p': d.top_p, 'top_k': d.top_k}


def run_groups(args, cfg: gr.RewardConfig) -> int:
    import torch
    import generate_validation as gv
    from grpo_data import build_prompt_rows

    model, tokenizer = gv.load_adapter_model(args.adapter, base_model=str(args.merged_model))
    prompts = build_prompt_rows(seed=args.seed)[:args.num_prompts]
    rows_by_prompt = []
    for i, p in enumerate(prompts):
        gens = []
        for g in range(args.num_generations):
            torch.manual_seed(args.seed + i * args.num_generations + g)
            gens.append(gv.generate_one(model, tokenizer, p['source_text'], p['target_author'],
                                        **grpo_sampling_kwargs()))
        rows_by_prompt.append(gr.score_batch([p['source_text']] * len(gens), gens,
                                             [p['target_author']] * len(gens), cfg))
        print(f'{i + 1}/{len(prompts)} poem {p["source_poem_id"]} -> {p["target_author"]}: '
              f'R={[round(r["R"], 3) for r in rows_by_prompt[-1]]}')

    summary = summarise_groups(rows_by_prompt, GROUP_KEYS)
    print(f"\n{'component':<26}{'mean':>8}{'within-grp std':>16}{'between std':>13}")
    for k, s in summary.items():
        print(f"{k:<26}{s['mean']:>8.3f}{s['within_group_std']:>16.3f}{s['between_prompt_std']:>13.3f}")

    args.groups_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.groups_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['component', 'mean', 'within_group_std', 'between_prompt_std'])
        for k, s in summary.items():
            w.writerow([k, s['mean'], s['within_group_std'], s['between_prompt_std']])
    print(f'\nwrote {args.groups_out}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fixtures', action='store_true')
    p.add_argument('--source-poem-id')
    p.add_argument('--source-file')
    p.add_argument('--target-author')
    p.add_argument('--generated-file')
    p.add_argument('--generations-csv', type=Path)
    p.add_argument('--row', type=int, default=0)
    p.add_argument('--style-weight', type=float, default=0.5)
    p.add_argument('--content-weight', type=float, default=0.5)
    p.add_argument('--style-weights', type=json.loads, default=None,
                   help='JSON {group: weight}; omitted groups keep their default')
    p.add_argument('--groups', action='store_true',
                   help='sample --num-generations completions per train prompt from the '
                        'SFT policy and report each component\'s within-group spread (GPU)')
    p.add_argument('--merged-model', type=Path, default=ROOT / 'models' / 'qwen3-sft-merged')
    p.add_argument('--adapter', type=Path, default=None,
                   help='optional LoRA on top of --merged-model (e.g. a GRPO checkpoint)')
    p.add_argument('--num-prompts', type=int, default=20)
    p.add_argument('--num-generations', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--groups-out', type=Path, default=DEFAULT_GROUPS_OUT)
    return p


def config_from_args(args) -> gr.RewardConfig:
    weights = dict(gr.DEFAULT_STYLE_WEIGHTS, **(args.style_weights or {}))
    return gr.RewardConfig(style_weights=weights, style_weight=args.style_weight,
                           content_weight=args.content_weight)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.groups:
        return run_groups(args, cfg)
    if args.fixtures:
        return run_fixtures(cfg)
    if not (args.generations_csv or (args.target_author and args.generated_file
                                     and (args.source_poem_id or args.source_file))):
        build_parser().error('give --fixtures, --generations-csv, or source + '
                             '--target-author + --generated-file')
    return run_single(args, cfg)


if __name__ == '__main__':
    sys.exit(main())
