"""
GRPO fine-tuning of the SFT Qwen3-4B style-transfer model (stage 2).

Start policy = the SFT adapter merged into the base weights
(src/merge_sft_adapter.py -> models/qwen3-sft-merged), plus a FRESH LoRA.
With beta > 0, TRL's KL reference is the same model with the adapter off --
i.e. exactly the SFT model. The base weights stay frozen.

Prompts: train-split (source poem, target author) pairs (src/grpo_data.py),
in the SFT chat format. Rewards: src/grpo_rewards.py's style_reward and
content_reward, combined by TRL as reward_weights=[style_weight,
content_weight]. Every component is logged via TRL's log_metric and written
per completion to <output-dir>/reward_components.jsonl.

--smoke runs 3 tiny steps and asserts the loop works end to end (see
run_smoke_checks). No evaluation inside training: checkpoints are evaluated
afterwards with generate_validation.py (slurm/eval_grpo.slurm).

Usage:
    uv run python src/train_grpo.py --smoke
    uv run python src/train_grpo.py --max-steps 500
"""

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import grpo_rewards as gr
from grpo_data import DEFAULT_MAX_PROMPT_LENGTH, build_grpo_dataset
from grpo_rewards import make_reward_funcs
from merge_sft_adapter import DEFAULT_MERGED_DIR
from sft_data import DEFAULT_MODEL_NAME
from train_sft import DEFAULT_TARGET_MODULES, build_lora_config

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
DEFAULT_OUTPUT_DIR = MODELS / 'qwen3-grpo'

log = logging.getLogger('train_grpo')


# ── Model ────────────────────────────────────────────────────────────────────────

def load_policy(sft_merged: Path, attn_implementation: str):
    sft_merged = Path(sft_merged)
    if not (sft_merged / 'merge_info.json').exists():
        raise SystemExit(f'{sft_merged}/merge_info.json missing -- build the start policy with '
                         '`python src/merge_sft_adapter.py` (GRPO must start from SFT, not base)')
    tokenizer = AutoTokenizer.from_pretrained(str(sft_merged))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(sft_merged), dtype=torch.bfloat16, attn_implementation=attn_implementation)
    return model, tokenizer


def completion_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Sum of log p(completion | prompt) -- the smoke test's deterministic
    'did the update change the policy' probe."""
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
    ids = tokenizer(prompt + completion, add_special_tokens=False,
                    return_tensors='pt')['input_ids'].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(1, ids[0, 1:, None])[:, 0]
    return logp[n_prompt - 1:].sum().item()


# ── Config ───────────────────────────────────────────────────────────────────────

def reward_config(args) -> gr.RewardConfig:
    return gr.RewardConfig(
        style_weights=dict(gr.DEFAULT_STYLE_WEIGHTS, **(args.style_weights or {})),
        style_weight=args.style_weight, content_weight=args.content_weight,
        components_log_path=Path(args.output_dir) / 'reward_components.jsonl')


def build_grpo_config(args) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        lr_scheduler_type='constant_with_warmup',
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=args.mask_truncated_completions,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=args.beta,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        num_iterations=args.num_iterations,
        reward_weights=[args.style_weight, args.content_weight],
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        report_to=args.report_to,
        log_completions=True,
        num_completions_to_print=2,
        use_cpu=args.use_cpu,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def _git_sha() -> str:
    """HEAD without calling git (the container has no git binary)."""
    head = ROOT / '.git' / 'HEAD'
    if not head.exists():
        return None
    ref = head.read_text().strip()
    if ref.startswith('ref: '):
        ref_path = ROOT / '.git' / ref[5:]
        if ref_path.exists():
            return ref_path.read_text().strip()
        packed = ROOT / '.git' / 'packed-refs'
        for line in packed.read_text().splitlines() if packed.exists() else []:
            if line.endswith(ref[5:]):
                return line.split()[0]
        return None
    return ref


def write_run_config(args, grpo_config: GRPOConfig, dataset_stats: dict) -> None:
    import peft
    import transformers
    import trl
    merge_info = json.loads((Path(args.sft_merged) / 'merge_info.json').read_text())
    record = {
        'args': vars(args), 'git_sha': _git_sha(), 'merge_info': merge_info,
        'calibration_sha256': _sha256(gr.CALIBRATION_PATH), 'dataset': dataset_stats,
        'versions': {'trl': trl.__version__, 'transformers': transformers.__version__,
                     'peft': peft.__version__, 'torch': torch.__version__},
        'grpo_config': grpo_config.to_dict(),
    }
    out = Path(args.output_dir) / 'run_config.json'
    out.write_text(json.dumps(record, default=str, ensure_ascii=False, indent=2), encoding='utf-8')


# ── Smoke checks ────────────────────────────────────────────────────────────────

def _history_failures(trainer) -> list:
    failures = []
    history = trainer.state.log_history
    clipped = [h['completions/clipped_ratio'] for h in history if 'completions/clipped_ratio' in h]
    masked = getattr(trainer.args, 'mask_truncated_completions', True)
    if masked and clipped and all(c >= 1.0 for c in clipped):
        failures.append('every completion hit --max-completion-length and was truncated '
                        '(masked from the loss) -- raise the cap')
    losses = [h['loss'] for h in history if 'loss' in h]
    if not losses or not all(math.isfinite(l) for l in losses):
        failures.append(f'loss not finite: {losses}')
    stds = [h['reward_std'] for h in history if 'reward_std' in h]
    if not any(s > 0 for s in stds):
        failures.append(f'reward constant within every group (reward_std={stds})')
    rewards = [h['reward'] for h in history if 'reward' in h]
    if not all(math.isfinite(r) for r in rewards):
        failures.append(f'reward not finite: {rewards}')
    return failures


def run_smoke_checks(trainer, tokenizer, probe: tuple) -> list:
    failures = _history_failures(trainer)
    lora_b = sum(p.detach().float().norm().item()
                 for n, p in trainer.model.named_parameters() if 'lora_B' in n)
    if lora_b == 0.0:
        failures.append('LoRA B weights still zero -- no update reached the adapter')
    # Adapter on vs off on the same model, same code path: the difference is
    # exactly what the GRPO LoRA contributes. The fresh LoRA starts as an
    # identity (B = 0), so "off" is the starting (SFT) policy.
    trainer.model.eval()
    after_logp = completion_logprob(trainer.model, tokenizer, *probe)
    with trainer.model.disable_adapter():
        start_logp = completion_logprob(trainer.model, tokenizer, *probe)
    log.info('probe log-prob start=%.5f after=%.5f', start_logp, after_logp)
    if abs(after_logp - start_logp) <= 1e-4:
        failures.append('probe completion log-prob unchanged -- update did not move the policy')
    return failures


def _probe(dataset) -> tuple:
    row = dataset[0]
    first_lines = '\n'.join(l for l in row['source_text'].splitlines() if l.strip())
    return row['prompt'], '\n'.join(first_lines.splitlines()[:3])


# ── Train ────────────────────────────────────────────────────────────────────────

def train(args) -> Path:
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_policy(args.sft_merged, args.attn_implementation)
    dataset, stats = build_grpo_dataset(
        tokenizer, num_prompts=args.num_prompts, max_prompt_length=args.max_prompt_length,
        target_authors=args.target_authors, num_target_authors=args.num_target_authors,
        seed=args.seed)

    style_reward, content_reward = make_reward_funcs(reward_config(args))
    grpo_config = build_grpo_config(args)
    lora = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.target_modules)

    trainer = GRPOTrainer(model=model, reward_funcs=[style_reward, content_reward],
                          args=grpo_config, train_dataset=dataset,
                          processing_class=tokenizer, peft_config=lora)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    write_run_config(args, grpo_config, stats)
    log.info('GRPO adapter saved to %s', args.output_dir)

    if args.smoke:
        failures = run_smoke_checks(trainer, tokenizer, _probe(dataset))
        for f in failures:
            log.error('SMOKE FAIL: %s', f)
        if failures:
            sys.exit(1)
        log.info('SMOKE OK: generate -> reward -> GRPO update -> policy changed')
    return args.output_dir


# ── CLI ─────────────────────────────────────────────────────────────────────────

def apply_smoke_overrides(args):
    args.max_steps, args.num_prompts, args.num_generations = 3, 4, 4
    args.per_device_train_batch_size, args.gradient_accumulation_steps = 8, 1
    # max_completion_length is left alone: SFT completions have a median of
    # 291 tokens, and with truncation masking a short cap zeroes the loss of
    # every smoke completion.
    args.save_steps = 10_000
    args.learning_rate = 1e-4   # large enough that 3 steps visibly move the probe
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = MODELS / 'qwen3-grpo-smoke'
    return args


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sft-merged', type=Path, default=DEFAULT_MERGED_DIR,
                   help='SFT-merged start policy (merge_sft_adapter.py output)')
    p.add_argument('--base-model', default=DEFAULT_MODEL_NAME,
                   help='recorded for provenance; the weights come from --sft-merged')
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--smoke', action='store_true')

    d = p.add_argument_group('data')
    d.add_argument('--num-prompts', type=int, default=None)
    d.add_argument('--num-target-authors', type=int, default=10)
    d.add_argument('--target-authors', nargs='+', default=None)
    d.add_argument('--max-prompt-length', type=int, default=DEFAULT_MAX_PROMPT_LENGTH)

    l = p.add_argument_group('LoRA')
    l.add_argument('--lora-r', type=int, default=16)
    l.add_argument('--lora-alpha', type=int, default=32)
    l.add_argument('--lora-dropout', type=float, default=0.0)
    l.add_argument('--target-modules', nargs='+', default=DEFAULT_TARGET_MODULES)

    r = p.add_argument_group('reward')
    r.add_argument('--style-weight', type=float, default=0.5, help='alpha')
    r.add_argument('--content-weight', type=float, default=0.5, help='beta (content)')
    r.add_argument('--style-weights', type=json.loads, default=None,
                   help='JSON {group: weight}; omitted groups keep their default')

    t = p.add_argument_group('GRPO')
    t.add_argument('--learning-rate', type=float, default=1e-5)
    t.add_argument('--warmup-steps', type=int, default=10)
    t.add_argument('--num-generations', type=int, default=4)
    t.add_argument('--per-device-train-batch-size', type=int, default=8)
    t.add_argument('--gradient-accumulation-steps', type=int, default=4)
    t.add_argument('--max-completion-length', type=int, default=512)
    t.add_argument('--no-mask-truncated-completions', dest='mask_truncated_completions',
                   action='store_false', default=True,
                   help='give truncated completions loss too (tests with a random model only)')
    t.add_argument('--temperature', type=float, default=0.9)
    t.add_argument('--top-p', type=float, default=1.0,
                   help='GRPO sampling nucleus (TRL default; recorded explicitly because '
                        'evaluate_reward --groups must sample the same way)')
    t.add_argument('--top-k', type=int, default=0, help='0 disables top-k (TRL default)')
    t.add_argument('--beta', type=float, default=0.04, help='KL coefficient (0 disables the reference)')
    t.add_argument('--loss-type', default='dapo')
    t.add_argument('--scale-rewards', default='group')
    t.add_argument('--num-iterations', type=int, default=1)
    t.add_argument('--max-steps', type=int, default=500)
    t.add_argument('--logging-steps', type=int, default=1)
    t.add_argument('--save-steps', type=int, default=50)
    t.add_argument('--save-total-limit', type=int, default=5)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--attn-implementation', default='sdpa',
                   choices=['sdpa', 'eager', 'flash_attention_2'])
    t.add_argument('--report-to', default='none')
    t.add_argument('--resume-from-checkpoint', default=None)
    t.add_argument('--use-cpu', action='store_true', help='tests only')

    bf = t.add_mutually_exclusive_group()
    bf.add_argument('--bf16', dest='bf16', action='store_true', default=True)
    bf.add_argument('--no-bf16', dest='bf16', action='store_false')
    gc = t.add_mutually_exclusive_group()
    gc.add_argument('--gradient-checkpointing', dest='gradient_checkpointing',
                    action='store_true', default=True)
    gc.add_argument('--no-gradient-checkpointing', dest='gradient_checkpointing',
                    action='store_false')
    return p.parse_args(argv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    arguments = parse_args()
    if arguments.smoke:
        arguments = apply_smoke_overrides(arguments)
    train(arguments)
