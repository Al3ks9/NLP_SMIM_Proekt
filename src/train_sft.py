"""
LoRA supervised fine-tuning of Qwen/Qwen3-4B on the synthetic style-transfer
dataset (Qwen SFT pipeline, stage 3-4).

Task: given a source poem and a target author (sft_data.build_messages), the
model learns to produce the poem generate_synthetic.py's Gemma pipeline wrote
for that pair -- i.e. distilling the heavier, hand-scaffolded Gemma prompt
(style profile prose, vocabulary palette, exemplar lines, structural targets)
into a small model that performs the transfer from just the two things the
task is actually defined on.

Trains on split='train' rows only. split='val' rows are used, if present,
purely as SFTTrainer's eval_dataset -- a forward-pass loss for monitoring,
never backpropagated through -- so "don't train on validation data" holds in
the sense that matters (no gradient ever sees it). split='test' rows are never
touched here at all; they're reserved for generate_validation.py-style final
inspection once this stage's output is judged.

Loss is masked to the assistant completion (the poem) via TRL's native
assistant_only_loss=True, which uses the chat template's `{% generation %}`
markers to build a per-token assistant mask -- Qwen3 is one of the model
families TRL's get_training_chat_template() explicitly patches such markers
onto, so this works even though Qwen3's own published chat template doesn't
define them itself. Without it the model would also be trained to reproduce
the prompt tokens, which is wasted signal for a generation task like this
one. This requires the training dataset to carry a "messages" column (a list
of {role, content} dicts) rather than a single pre-rendered "text" string --
see sft_data.build_hf_dataset -- since the completion-only mask is computed
per role, not by locating a marker substring.

LoRA target modules were confirmed against transformers' actual Qwen3
modeling code (Qwen3Attention: q_proj/k_proj/v_proj/o_proj -- q_norm/k_norm
are RMSNorm layers, not nn.Linear, and are deliberately excluded; Qwen3MLP
inherits GemmaMLP: gate_proj/up_proj/down_proj), not assumed from a generic
Llama-family default.

Usage (single A100 80GB; see slurm/train_sft.slurm for the cluster job):
    uv run python src/train_sft.py \\
        --num-train-epochs 3 --per-device-train-batch-size 4 \\
        --gradient-accumulation-steps 4 --learning-rate 2e-4
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from sft_data import DEFAULT_MODEL_NAME, build_hf_dataset, load_synthetic_rows

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'

DEFAULT_OUTPUT_DIR = MODELS / 'qwen3-lora-sft'
DEFAULT_SYNTHETIC_PATH = DATA / 'synthetic' / 'synthetic_dataset.csv'

# Confirmed via transformers' Qwen3 modeling source -- see module docstring.
DEFAULT_TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
]

log = logging.getLogger('train_sft')


# ── Model / tokenizer / LoRA setup ──────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, gradient_checkpointing: bool,
                             attn_implementation: str = 'sdpa'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Qwen3 ties pad/eos in most configs, but don't assume -- an unset pad
        # token makes the data collator's padding silently wrong.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation,
    )
    if gradient_checkpointing:
        model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, tokenizer


def build_lora_config(r: int, alpha: int, dropout: float,
                      target_modules: list) -> LoraConfig:
    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias='none', task_type='CAUSAL_LM',
    )


# ── Dataset ──────────────────────────────────────────────────────────────────────

def build_datasets(synthetic_path: Path, eval_split: str = 'val'):
    """(train_dataset, eval_dataset). eval_dataset is None when eval_split is
    falsy or the synthetic CSV has no rows for it."""
    train_rows = load_synthetic_rows(split='train', path=synthetic_path)
    if not train_rows:
        raise ValueError(f'no split="train" rows in {synthetic_path} -- run '
                         'generate_synthetic.py first')
    train_dataset = build_hf_dataset(train_rows)

    eval_dataset = None
    if eval_split:
        eval_rows = load_synthetic_rows(split=eval_split, path=synthetic_path)
        if eval_rows:
            eval_dataset = build_hf_dataset(eval_rows)
        else:
            log.warning('no split=%r rows in %s -- training without eval', eval_split,
                       synthetic_path)
    return train_dataset, eval_dataset


# ── Training ─────────────────────────────────────────────────────────────────────

def train(args) -> Path:
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.gradient_checkpointing, args.attn_implementation)
    train_dataset, eval_dataset = build_datasets(args.synthetic_data, eval_split=args.eval_split)

    lora_config = build_lora_config(
        args.lora_r, args.lora_alpha, args.lora_dropout, args.target_modules)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_seq_length,
        packing=False,  # dataset is ~thousands of short examples, not worth the complexity
        assistant_only_loss=True,  # mask loss to the poem; see module docstring
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy if eval_dataset is not None else 'no',
        bf16=args.bf16,
        seed=args.seed,
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / 'run_config.json').write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding='utf-8')

    log.info('adapter saved to %s', args.output_dir)
    return args.output_dir


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME)
    parser.add_argument('--synthetic-data', type=Path, default=DEFAULT_SYNTHETIC_PATH)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--eval-split', default='val',
                       help="synthetic-CSV split used as eval_dataset (loss monitoring "
                            "only, never trained on); pass '' to disable")

    lora = parser.add_argument_group('LoRA')
    lora.add_argument('--lora-r', type=int, default=16)
    lora.add_argument('--lora-alpha', type=int, default=32)
    lora.add_argument('--lora-dropout', type=float, default=0.05)
    lora.add_argument('--target-modules', nargs='+', default=DEFAULT_TARGET_MODULES)

    train_args = parser.add_argument_group('training')
    train_args.add_argument('--num-train-epochs', type=float, default=3.0)
    train_args.add_argument('--per-device-train-batch-size', type=int, default=4)
    train_args.add_argument('--per-device-eval-batch-size', type=int, default=4)
    train_args.add_argument('--gradient-accumulation-steps', type=int, default=4)
    train_args.add_argument('--max-seq-length', type=int, default=1024)
    train_args.add_argument('--learning-rate', type=float, default=2e-4)
    train_args.add_argument('--lr-scheduler-type', default='cosine')
    train_args.add_argument('--warmup-ratio', type=float, default=0.03)
    train_args.add_argument('--logging-steps', type=int, default=10)
    train_args.add_argument('--save-strategy', default='epoch')
    train_args.add_argument('--eval-strategy', default='epoch')
    train_args.add_argument('--seed', type=int, default=42)
    train_args.add_argument('--attn-implementation', default='sdpa',
                           choices=['sdpa', 'eager', 'flash_attention_2'])
    train_args.add_argument('--report-to', default='none')
    train_args.add_argument('--resume-from-checkpoint', default=None)

    bf16_group = train_args.add_mutually_exclusive_group()
    bf16_group.add_argument('--bf16', dest='bf16', action='store_true', default=True)
    bf16_group.add_argument('--no-bf16', dest='bf16', action='store_false')

    gc_group = train_args.add_mutually_exclusive_group()
    gc_group.add_argument('--gradient-checkpointing', dest='gradient_checkpointing',
                         action='store_true', default=True)
    gc_group.add_argument('--no-gradient-checkpointing', dest='gradient_checkpointing',
                         action='store_false')

    return parser.parse_args(argv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    train(parse_args())
