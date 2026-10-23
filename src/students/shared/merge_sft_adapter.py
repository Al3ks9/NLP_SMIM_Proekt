"""
Fold the SFT LoRA adapter into Qwen3-4B's weights (GRPO stage, step 0).

Why: TRL's GRPOTrainer, given a PEFT model, gets its KL reference by
switching the trainable adapter off. Training the SFT adapter directly would
make that reference the *untuned* base model; merging SFT into the weights
and attaching a fresh LoRA makes both the start policy and the KL reference
exactly the SFT model. (TRL itself refuses a PeftModel + peft_config and
says to merge first.)

Idempotent: re-running with the same adapter is a no-op; a merge from a
different adapter is refused without --force.

Usage:
    uv run python src/merge_sft_adapter.py                   # models/qwen3-lora-sft -> models/qwen3-sft-merged
    uv run python src/merge_sft_adapter.py --family gemma4   # models/gemma4-e4b-lora-sft -> models/gemma4-e4b-sft-merged

The base model loads through its family's loader (Gemma 4: text-only), so a
merged Gemma dir is a plain gemma4_text causal LM that AutoModelForCausalLM
reloads. merge_info.json records the family; a --family that disagrees with
the adapter's provenance is refused.
"""

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path

import torch

import model_families as mf
from generate_validation import _infer_base_model

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
DEFAULT_SFT_ADAPTER = MODELS / 'qwen3-lora-sft'
DEFAULT_MERGED_DIR = MODELS / 'qwen3-sft-merged'

log = logging.getLogger('merge_sft_adapter')


def adapter_sha256(adapter_dir: Path) -> str:
    return hashlib.sha256((Path(adapter_dir) / 'adapter_model.safetensors').read_bytes()).hexdigest()


def merge(adapter_dir: Path, output_dir: Path, base_model: str = None, force: bool = False,
          family: str = None) -> Path:
    """family=None infers it from the adapter's base model (default qwen3)."""
    from peft import PeftModel
    from transformers import AutoTokenizer

    adapter_dir, output_dir = Path(adapter_dir), Path(output_dir)
    # family check before anything below can delete an existing merge (--force)
    base_model = base_model or _infer_base_model(adapter_dir)
    fam = mf.get_family(family or mf.family_of(base_model) or mf.DEFAULT_FAMILY)
    mf.assert_family_matches(fam, adapter_dir)
    digest = adapter_sha256(adapter_dir)
    info_path = output_dir / 'merge_info.json'
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding='utf-8'))
        if info.get('adapter_sha256') == digest:
            log.info('%s already holds a merge of %s -- nothing to do', output_dir, adapter_dir)
            return output_dir
        if not force:
            raise SystemExit(f'{output_dir} holds a merge of a different adapter '
                             f'({info.get("adapter_dir")}); pass --force to replace it')
        shutil.rmtree(output_dir)

    log.info('merging %s into %s (family %s)', adapter_dir, base_model, fam.key)
    model = mf.load_causal_lm(fam, base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(output_dir)
    info_path.write_text(json.dumps({
        'base_model': base_model, 'adapter_dir': str(adapter_dir), 'adapter_sha256': digest,
        'family': fam.key,
    }, indent=2), encoding='utf-8')
    log.info('merged model saved to %s', output_dir)
    return output_dir


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--family', default=mf.DEFAULT_FAMILY, choices=sorted(mf.FAMILIES))
    parser.add_argument('--adapter', type=Path, default=None,
                       help='default: models/<family tag>-lora-sft')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='default: models/<family tag>-sft-merged')
    parser.add_argument('--base-model', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    dirs = mf.default_dirs(mf.get_family(args.family))
    merge(args.adapter or dirs['sft'], args.output_dir or dirs['merged'], args.base_model,
          args.force, family=args.family)
