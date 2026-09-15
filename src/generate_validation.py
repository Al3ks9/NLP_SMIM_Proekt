"""
Validation generation (Qwen SFT pipeline, stage 6): load the base model plus a
trained LoRA adapter (src/train_sft.py's output) and run style transfer over
real poems from one split -- val by default, never the split trained on --
producing candidates for manual inspection now and, unmodified, for the GRPO
reward stage this pipeline is meant to plug into next (see CLAUDE.md's
pipeline-order note in this module's own docstring context: "keep the
pipeline modular so the next stage can load the SFT model and add generated
candidates -> style/content reward -> GRPO").

Uses the exact same chat format as training (sft_data.render_inference_prompt)
so there is no train/inference prompt skew, and reuses
llm_style_transfer.structural_targets/validate_structure for the fit metrics
-- the same yardstick generate_synthetic.py's Gemma outputs are measured
against, so Qwen's outputs are directly comparable to Gemma's.

Usage:
    uv run python src/generate_validation.py \\
        --adapter models/qwen3-lora-sft --split val --num-source-poems 50
"""

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import llm_style_transfer as lst
from generate_synthetic import build_pairs, sample_source_poems
from sft_data import DEFAULT_MODEL_NAME, render_inference_prompt, select_target_authors

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

DEFAULT_ADAPTER_DIR = ROOT / 'models' / 'qwen3-lora-sft'
DEFAULT_OUTPUT_DIR = DATA / 'validation_generations'
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_SEED = 42

log = logging.getLogger('generate_validation')


# ── Model loading ────────────────────────────────────────────────────────────────

def load_adapter_model(adapter_dir: Path, base_model: str = None):
    """Base model + the trained LoRA adapter attached via PEFT, plus the
    tokenizer saved alongside the adapter (train_sft.py saves both to the
    same directory)."""
    if base_model is None:
        base_model = _infer_base_model(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def _infer_base_model(adapter_dir: Path) -> str:
    """base_model_name_or_path out of the adapter's own adapter_config.json,
    so --base-model only needs overriding when that record is missing."""
    config_path = adapter_dir / 'adapter_config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if config.get('base_model_name_or_path'):
            return config['base_model_name_or_path']
    log.warning('could not infer base model from %s -- falling back to %s',
               config_path, DEFAULT_MODEL_NAME)
    return DEFAULT_MODEL_NAME


# ── Generation ───────────────────────────────────────────────────────────────────

def generate_one(model, tokenizer, source_poem_text: str, target_author: str,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, do_sample: bool = True,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
    """
    One style-transfer generation from the loaded adapter model, returning
    just the newly generated text (the prompt is not echoed back).

    Reproducibility is the caller's job (torch.manual_seed before calling) --
    generate() takes no generator= kwarg in current transformers (sampling
    reads the global RNG via plain torch.multinomial), so a per-call
    torch.Generator can't be threaded through here.
    """
    prompt = render_inference_prompt(tokenizer, source_poem_text, target_author)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    kwargs = {'max_new_tokens': max_new_tokens, 'do_sample': do_sample,
             'pad_token_id': tokenizer.pad_token_id}
    if do_sample:
        kwargs['temperature'] = temperature
        kwargs['top_p'] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)
    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Orchestration ─────────────────────────────────────────────────────────────────

def run_validation(adapter_dir: Path, split: str = 'val', num_source_poems: int = 50,
                   target_authors: list = None, num_target_authors: int = 10,
                   base_model: str = None, seed: int = DEFAULT_SEED,
                   samples_per_pair: int = 1, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                   do_sample: bool = True, temperature: float = 0.7, top_p: float = 0.9,
                   output_path: Path = None) -> Path:
    authors = select_target_authors(num_target_authors, explicit=target_authors)
    source_poems = sample_source_poems(num_source_poems, seed=seed, split=split)
    pairs = build_pairs(source_poems, authors, samples_per_pair)

    model, tokenizer = load_adapter_model(adapter_dir, base_model=base_model)

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f'{adapter_dir.name}_{split}.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['example_id', 'source_poem_id', 'source_author', 'source_title',
                 'target_author', 'split', 'sample_index', 'generated_poem',
                 'adapter', 'base_model', 'delta_lines', 'delta_stanzas',
                 'delta_tokens_per_line', 'delta_lines_per_stanza', 'timestamp']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (poem, target_author, sample_index) in enumerate(pairs, 1):
            torch.manual_seed(seed + i)  # reproducible run-to-run without a generate() generator=
            generated = generate_one(
                model, tokenizer, poem['song_text'], target_author,
                max_new_tokens=max_new_tokens, do_sample=do_sample,
                temperature=temperature, top_p=top_p)
            targets = lst.structural_targets(target_author)
            fit = lst.validate_structure(generated, targets)

            writer.writerow({
                'example_id': f'{poem["poem_id"]}_{target_author}_{sample_index}',
                'source_poem_id': poem['poem_id'], 'source_author': poem['author'],
                'source_title': poem['song_title'], 'target_author': target_author,
                'split': split, 'sample_index': sample_index,
                'generated_poem': generated, 'adapter': str(adapter_dir),
                'base_model': base_model or _infer_base_model(adapter_dir),
                'delta_lines': fit['delta_lines'], 'delta_stanzas': fit['delta_stanzas'],
                'delta_tokens_per_line': fit['delta_tokens_per_line'],
                'delta_lines_per_stanza': fit['delta_lines_per_stanza'],
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })
            f.flush()
            log.info('%d/%d poem_id=%s (%s) -> %s', i, len(pairs), poem['poem_id'],
                    poem['author'], target_author)

    log.info('wrote %d generations to %s', len(pairs), output_path)
    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--adapter', type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument('--base-model', default=None,
                       help='defaults to the adapter_config.json base model')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num-source-poems', type=int, default=50)
    parser.add_argument('--num-target-authors', type=int, default=10)
    parser.add_argument('--target-authors', nargs='+', default=None)
    parser.add_argument('--samples-per-pair', type=int, default=1)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max-new-tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--greedy', action='store_true',
                       help='disable sampling (do_sample=False) for deterministic output')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    run_validation(
        adapter_dir=args.adapter, split=args.split, num_source_poems=args.num_source_poems,
        target_authors=args.target_authors, num_target_authors=args.num_target_authors,
        base_model=args.base_model, seed=args.seed, samples_per_pair=args.samples_per_pair,
        max_new_tokens=args.max_new_tokens, do_sample=not args.greedy,
        temperature=args.temperature, top_p=args.top_p, output_path=args.output,
    )
