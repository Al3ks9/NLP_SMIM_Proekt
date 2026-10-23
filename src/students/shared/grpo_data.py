"""
GRPO prompt dataset: (train-split source poem, target author) pairs rendered
in the exact chat format SFT trained on (sft_data.render_inference_prompt,
thinking disabled), with the raw source text and target author kept as
columns -- TRL passes extra columns to the reward functions as keyword lists,
and those two are all the reward is allowed to see.

Train split only (make_splits.poems_in_split('train')); the same target
authors as SFT (sft_data.select_target_authors); self-transfer skipped
(generate_synthetic.build_pairs). No Gemma output is involved.

Prompts over max_prompt_length tokens are dropped, never truncated: a cut
source poem would have the content reward compare against text the model
never saw.
"""

import logging
import random

from generate_synthetic import build_pairs
from make_splits import poems_in_split
from sft_data import render_inference_prompt, select_target_authors

DEFAULT_SEED = 42
DEFAULT_MAX_PROMPT_LENGTH = 1024

log = logging.getLogger('grpo_data')


def build_prompt_rows(target_authors: list = None, num_target_authors: int = 10,
                      seed: int = DEFAULT_SEED) -> list:
    authors = select_target_authors(num_target_authors, explicit=target_authors)
    pairs = build_pairs(poems_in_split('train'), authors, samples_per_pair=1)
    random.Random(seed).shuffle(pairs)
    return [{'source_poem_id': p['poem_id'], 'source_author': p['author'],
             'source_text': p['song_text'], 'target_author': t} for p, t, _ in pairs]


def build_grpo_dataset(tokenizer, num_prompts: int = None,
                       max_prompt_length: int = DEFAULT_MAX_PROMPT_LENGTH,
                       target_authors: list = None, num_target_authors: int = 10,
                       seed: int = DEFAULT_SEED) -> tuple:
    from datasets import Dataset

    candidates = build_prompt_rows(target_authors, num_target_authors, seed)
    kept, dropped = [], 0
    for row in candidates:
        prompt = render_inference_prompt(tokenizer, row['source_text'], row['target_author'])
        if len(tokenizer(prompt, add_special_tokens=False)['input_ids']) > max_prompt_length:
            dropped += 1
            continue
        kept.append({'prompt': prompt, **row})
        if num_prompts and len(kept) >= num_prompts:
            break
    stats = {'kept': len(kept), 'dropped_too_long': dropped, 'candidates': len(candidates)}
    log.info('GRPO prompts: %(kept)d kept, %(dropped_too_long)d dropped as too long '
             '(of %(candidates)d train pairs)', stats)
    if not kept:
        raise ValueError('no GRPO prompts fit max_prompt_length')
    return Dataset.from_list(kept), stats
