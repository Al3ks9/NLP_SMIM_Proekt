"""
Shared infrastructure for the Qwen3-4B SFT pipeline: target-author selection,
the chat-message format the SFT task is trained and inferred on, and loading
generate_synthetic.py's output into a training-ready HF Dataset.

Owns the compact (source poem, target author) -> poem chat format so
train_sft.py and generate_validation.py can't drift on it -- same reasoning as
tagging.py owning the POS normalizations for pos_tag_corpus.py and the
inference path.

This is deliberately NOT the heavy prompt llm_style_transfer.py builds for
Gemma (style profile prose, vocabulary palette, exemplar lines, structural
targets). The whole point of this stage is to distill that scaffolded
capability into a small model that performs the transfer given just the two
things the task is actually defined on: a source poem and a target author.
"""

import csv
from pathlib import Path

from make_splits import load_stripped_songs, SPLIT_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

DEFAULT_MODEL_NAME = 'Qwen/Qwen3-4B'

SYSTEM_PROMPT = (
    "You rewrite Macedonian poems in a different poet's style. Given a source "
    "poem and a target author, write a new poem in Macedonian, in Cyrillic "
    "script, that conveys the source poem's content and imagery in the "
    "target author's voice."
)

USER_TEMPLATE = 'Source poem:\n{source_poem}\n\nTarget author: {target_author}'

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


# ── Chat message format ─────────────────────────────────────────────────────────

def build_messages(source_poem: str, target_author: str, generated_poem: str = None) -> list:
    """
    [system, user] messages for the style-transfer task, or [system, user,
    assistant] when generated_poem is given (building a training example).

    enable_thinking=False belongs at the tokenizer.apply_chat_template() call
    site (render_chat / render_prompt below), not here -- these are the raw
    messages, backend-agnostic.
    """
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_TEMPLATE.format(
            source_poem=source_poem.strip(), target_author=target_author)},
    ]
    if generated_poem is not None:
        messages.append({'role': 'assistant', 'content': generated_poem.strip()})
    return messages


def render_chat(tokenizer, messages: list, add_generation_prompt: bool = False) -> str:
    """
    messages -> a single ChatML string via the tokenizer's own chat template.

    enable_thinking=False matches how llm_client.call_ollama already treats
    Qwen3 for this project (think=False) and how llm_client._strip_think
    defensively drops any <think> block that leaks through anyway -- Qwen3's
    reasoning mode would otherwise bury (and lengthen) every training example
    and every generation in a thinking block irrelevant to the poem itself.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def render_inference_prompt(tokenizer, source_poem: str, target_author: str) -> str:
    """Rendered ChatML string (system+user, generation prompt open) to feed
    generate() at inference/validation time."""
    messages = build_messages(source_poem, target_author)
    return render_chat(tokenizer, messages, add_generation_prompt=True)


# ── Target-author selection ─────────────────────────────────────────────────────

def load_profile_rows() -> list:
    def build():
        with open(DATA / 'author_style_profiles.csv', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    return _cached('profile_rows', build)


def select_target_authors(n: int, explicit: list = None) -> list:
    """
    n distinct target authors for style transfer.

    explicit, when given, is used as-is (after checking every name has a
    style profile) -- this is generate_synthetic.py's/generate_validation.py's
    --target-authors override. Otherwise: the n authors with the most poems
    among those with a style profile at all (author_style_profiles.csv already
    applies style_profiler.py's own MIN_POEMS=5 floor), ties broken by author
    name for determinism. Richer authors give exemplar_selection and
    vocabulary_palette more to work with, and being real distinct poets makes
    "distinct target styles" automatic without a bespoke diversity metric.
    """
    rows = load_profile_rows()
    by_author = {r['author']: r for r in rows}

    if explicit:
        missing = [a for a in explicit if a not in by_author]
        if missing:
            raise KeyError(f'no style profile for author(s): {missing}')
        return list(explicit)

    if n > len(rows):
        raise ValueError(
            f'requested {n} target authors but only {len(rows)} authors have a '
            f'style profile (author_style_profiles.csv)')

    ranked = sorted(rows, key=lambda r: (-int(r['num_poems']), r['author']))
    return [r['author'] for r in ranked[:n]]


# ── Loading generate_synthetic.py's output ──────────────────────────────────────

SYNTHETIC_PATH = DATA / 'synthetic' / 'synthetic_dataset.csv'


def load_synthetic_rows(split: str = None, path: Path = SYNTHETIC_PATH) -> list:
    """Rows of generate_synthetic.py's output CSV, optionally filtered to one
    split ('train' / 'val' / 'test')."""
    if split is not None and split not in SPLIT_NAMES:
        raise ValueError(f'split must be one of {SPLIT_NAMES}, got {split!r}')
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if split is not None:
        rows = [r for r in rows if r['split'] == split]
    return rows


def build_synthetic_messages(rows: list) -> list:
    """
    Synthetic-dataset rows -> a list of [system, user, assistant] message
    lists, one per row -- TRL's SFTTrainer trains on a "messages" column
    directly (applying the model's own chat template and, with
    assistant_only_loss=True, masking loss to the assistant turn via the
    template's `{% generation %}` markers), so no chat-string rendering or
    tokenizer happens on the training path at all -- that's why this needs no
    tokenizer argument, unlike render_inference_prompt.

    Pure aside from the stripped_songs.csv lookup -- the piece of
    build_hf_dataset() that doesn't need the `datasets` library, kept separate
    so it's testable without it.

    Looks the source poem's text up from stripped_songs.csv by poem_id rather
    than trusting a copy in the synthetic CSV -- one source of truth for poem
    text, matching load_poem_text_by_id's role in llm_style_transfer.py.
    """
    poems_by_id = {r['poem_id']: r for r in load_stripped_songs()}
    messages = []
    for row in rows:
        source = poems_by_id.get(row['source_poem_id'])
        if source is None:
            raise KeyError(f'synthetic row references unknown source_poem_id='
                           f'{row["source_poem_id"]!r}')
        messages.append(build_messages(
            source['song_text'], row['target_author'], row['generated_poem']))
    return messages


def build_hf_dataset(rows: list):
    """
    Synthetic-dataset rows -> a datasets.Dataset with one 'messages' column,
    ready for TRL's SFTTrainer (see build_synthetic_messages). Imports
    `datasets` lazily so importing this module never requires it -- only
    actually building a training set does.
    """
    from datasets import Dataset
    return Dataset.from_dict({'messages': build_synthetic_messages(rows)})
