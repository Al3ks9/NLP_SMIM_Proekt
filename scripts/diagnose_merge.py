"""
Why does a merged model disagree with its adapter? (gemma_compat's sft_merge
check failed: 0.136 nats/token on a 3-step Gemma adapter.)

Scores the same train completions under six models and prints the mean
per-token log-prob of each, relative to the bf16 adapter model:

  base           adapter disabled
  adapter_bf16   base (bf16) + LoRA        -- what training/eval see
  merged_saved   the merged dir on disk     -- what GRPO starts from
  adapter_fp32   base (fp32) + LoRA
  merged_fp32    adapter_fp32 merged in memory, no rounding
  merged_fp32_to_bf16  merged_fp32 cast to bf16 (what saving does)

Reading it:
  merged_fp32 ~ adapter_fp32, merged_saved ~ base  -> bf16 rounding erases a
      tiny adapter: the check is too strict for 3 steps, the merge is sound
  merged_fp32 != adapter_fp32                       -> PEFT merge is wrong for
      this architecture: a real bug
  adapter_bf16 != adapter_fp32                      -> bf16 forward noise

Run on the cluster (slurm/diagnose_merge.slurm), for Gemma and, as the
control whose merge is known to reproduce SFT's scores, Qwen:
    python scripts/diagnose_merge.py --family gemma4 \
        --adapter models/gemma4-compat/sft --merged models/gemma4-compat/merged
    python scripts/diagnose_merge.py --family qwen3 \
        --adapter models/qwen3-lora-sft --merged models/qwen3-sft-merged
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import model_families as mf  # noqa: E402
import sft_data  # noqa: E402
from train_grpo import completion_logprob  # noqa: E402


def score(model, items) -> list:
    """Mean log-prob per completion token, one value per item."""
    return [completion_logprob(model, TOKENIZER, p, c) / n for p, c, n in items]


def main() -> None:
    global TOKENIZER
    from peft import PeftModel
    from transformers import AutoTokenizer

    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--family', required=True, choices=sorted(mf.FAMILIES))
    a.add_argument('--adapter', type=Path, required=True)
    a.add_argument('--merged', type=Path, required=True)
    a.add_argument('--n', type=int, default=4, help='train completions to score')
    a.add_argument('--device', default='cuda')
    a.add_argument('--base-model', default=None, help='default: from adapter_config.json')
    args = a.parse_args()

    fam = mf.get_family(args.family)
    base_name = args.base_model or json.loads(
        (args.adapter / 'adapter_config.json').read_text())['base_model_name_or_path']
    TOKENIZER = AutoTokenizer.from_pretrained(str(args.adapter))
    rows = sft_data.load_synthetic_rows(split='train')[:args.n]
    items = []
    for source, target, poem in sft_data._source_target_poem(rows):
        pc = sft_data.build_prompt_completion(TOKENIZER, source, target, poem)
        n = len(TOKENIZER(pc['completion'], add_special_tokens=False)['input_ids'])
        items.append((pc['prompt'], pc['completion'], n))

    results = {}
    m = PeftModel.from_pretrained(mf.load_causal_lm(fam, base_name, dtype=torch.bfloat16),
                                  str(args.adapter)).to(args.device).eval()
    results['adapter_bf16'] = score(m, items)
    with m.disable_adapter():
        results['base'] = score(m, items)
    del m
    torch.cuda.empty_cache()

    m = mf.load_causal_lm(fam, args.merged, dtype=torch.bfloat16).to(args.device).eval()
    results['merged_saved'] = score(m, items)
    del m
    torch.cuda.empty_cache()

    m = PeftModel.from_pretrained(mf.load_causal_lm(fam, base_name, dtype=torch.float32),
                                  str(args.adapter)).to(args.device).eval()
    results['adapter_fp32'] = score(m, items)
    m = m.merge_and_unload()
    results['merged_fp32'] = score(m, items)
    m = m.to(torch.bfloat16)
    results['merged_fp32_to_bf16'] = score(m, items)

    ref = results['adapter_bf16']
    print(f'\n{args.family}: {args.adapter} vs {args.merged}, {len(items)} train completions, '
          f'{sum(n for *_, n in items)} tokens')
    print(f'{"model":22s} {"mean logp/token":>16s} {"vs adapter_bf16":>16s}  per completion')
    for name, vals in results.items():
        diffs = [v - r for v, r in zip(vals, ref)]
        print(f'{name:22s} {sum(vals) / len(vals):16.4f} {sum(diffs) / len(diffs):+16.4f}  '
              + ' '.join(f'{d:+.3f}' for d in diffs))


if __name__ == '__main__':
    main()
