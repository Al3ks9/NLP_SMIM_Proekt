"""
LoRA supervised fine-tuning of Qwen/Qwen3-4B on the synthetic style-transfer
dataset (Qwen SFT pipeline, stage 3-4).

--family gemma4 trains the Gemma 4 E4B student instead (src/model_families.py):
text-only load, prompt-completion rows with completion_only_loss (TRL has no
Gemma 4 training template), output models/gemma4-e4b-lora-sft. The notes
below describe the default Qwen family.

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
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import model_families as mf
from sft_data import DEFAULT_MODEL_NAME, build_hf_dataset, end_of_turn, load_synthetic_rows

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
                             attn_implementation: str = 'sdpa', family: str = mf.DEFAULT_FAMILY,
                             target_modules=None):
    """Model + tokenizer through the family's loader (Gemma 4: text-only).
    With target_modules, also checks they match a module of the loaded model
    (model_families.lora_targets) before any training starts."""
    fam = mf.get_family(family)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Qwen3 ties pad/eos in most configs, but don't assume -- an unset pad
        # token makes the data collator's padding silently wrong.
        tokenizer.pad_token = tokenizer.eos_token

    model = mf.load_causal_lm(
        fam, model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation,
    )
    if target_modules is not None:
        mf.lora_targets(fam, model, target_modules)
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

def _to_dataset(rows: list, fmt: str, tokenizer):
    # the Qwen call stays exactly build_hf_dataset(rows), as before families
    if fmt == 'messages':
        return build_hf_dataset(rows)
    return build_hf_dataset(rows, fmt=fmt, tokenizer=tokenizer)


def build_datasets(synthetic_path: Path, eval_split: str = 'val', fmt: str = 'messages',
                   tokenizer=None):
    """(train_dataset, eval_dataset). eval_dataset is None when eval_split is
    falsy or the synthetic CSV has no rows for it. fmt / tokenizer go to
    sft_data.build_hf_dataset ('prompt_completion' for Gemma)."""
    train_rows = load_synthetic_rows(split='train', path=synthetic_path)
    if not train_rows:
        raise ValueError(f'no split="train" rows in {synthetic_path} -- run '
                         'generate_synthetic.py first')
    train_dataset = _to_dataset(train_rows, fmt, tokenizer)

    eval_dataset = None
    if eval_split:
        eval_rows = load_synthetic_rows(split=eval_split, path=synthetic_path)
        if eval_rows:
            eval_dataset = _to_dataset(eval_rows, fmt, tokenizer)
        else:
            log.warning('no split=%r rows in %s -- training without eval', eval_split,
                       synthetic_path)
    return train_dataset, eval_dataset


# ── Training ─────────────────────────────────────────────────────────────────────

def build_sft_config(args, has_eval: bool = True, eos_token: str = None) -> SFTConfig:
    """Loss masked to the poem either way: assistant_only_loss on 'messages'
    (Qwen, via TRL's patched training template) or completion_only_loss on
    prompt-completion rows (Gemma, no TRL training template). See
    tests/test_sft_data.py for what each actually puts loss on.

    eos_token (Gemma: '<turn|>'): TRL appends tokenizer.eos_token to every
    completion and sets it on the saved tokenizer, which the merge copies and
    GRPO stops generation on. Gemma's eos is <eos>, but its turn ends with
    <turn|> -- without this, SFT trains '<turn|><eos>' and GRPO samples run
    past the turn. None leaves Qwen exactly as before."""
    completion_only = mf.get_family(args.family).loss_masking == 'completion_only'
    return SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_seq_length,
        packing=False,  # dataset is ~thousands of short examples, not worth the complexity
        assistant_only_loss=not completion_only,  # mask loss to the poem; see module docstring
        completion_only_loss=completion_only,
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
        eval_strategy=args.eval_strategy if has_eval else 'no',
        bf16=args.bf16,
        seed=args.seed,
        report_to=args.report_to,
        eos_token=eos_token,
    )


def train(args) -> Path:
    args = resolve(args)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.gradient_checkpointing, args.attn_implementation,
        family=args.family, target_modules=args.target_modules)
    fmt = ('prompt_completion' if mf.get_family(args.family).loss_masking == 'completion_only'
           else 'messages')
    train_dataset, eval_dataset = build_datasets(args.synthetic_data, eval_split=args.eval_split,
                                                 fmt=fmt, tokenizer=tokenizer)

    lora_config = build_lora_config(
        args.lora_r, args.lora_alpha, args.lora_dropout, args.target_modules)

    sft_config = build_sft_config(args, has_eval=eval_dataset is not None,
                                  eos_token=end_of_turn(tokenizer) if fmt == 'prompt_completion' else None)

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
    parser.add_argument('--family', default=mf.DEFAULT_FAMILY, choices=sorted(mf.FAMILIES),
                       help='student model family (model_families.py); sets the defaults below')
    parser.add_argument('--model-name', default=None, help="default: the family's hub model")
    parser.add_argument('--synthetic-data', type=Path, default=DEFAULT_SYNTHETIC_PATH)
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='default: models/<family tag>-lora-sft')
    parser.add_argument('--eval-split', default='val',
                       help="synthetic-CSV split used as eval_dataset (loss monitoring "
                            "only, never trained on); pass '' to disable")

    lora = parser.add_argument_group('LoRA')
    lora.add_argument('--lora-r', type=int, default=16)
    lora.add_argument('--lora-alpha', type=int, default=32)
    lora.add_argument('--lora-dropout', type=float, default=0.05)
    lora.add_argument('--target-modules', nargs='+', default=None,
                      help="default: the family's targets (model_families.py)")

    train_args = parser.add_argument_group('training')
    train_args.add_argument('--num-train-epochs', type=float, default=3.0)
    train_args.add_argument('--per-device-train-batch-size', type=int, default=4)
    train_args.add_argument('--per-device-eval-batch-size', type=int, default=4)
    train_args.add_argument('--gradient-accumulation-steps', type=int, default=4)
    train_args.add_argument('--max-seq-length', type=int, default=None,
                            help="default: the family's (qwen3 1024, gemma4 4096)")
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

    return resolve(parser.parse_args(argv))


def resolve(args):
    """Fill family-dependent defaults left as None. Idempotent; --family
    qwen3 resolves to exactly the pre-family values."""
    fam = mf.get_family(args.family)
    if args.model_name is None:
        args.model_name = fam.hf_name
    if args.output_dir is None:
        args.output_dir = mf.default_dirs(fam)['sft']
    if args.target_modules is None:
        args.target_modules = fam.target_modules
    if args.max_seq_length is None:
        args.max_seq_length = fam.max_seq_length
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    train(parse_args())
