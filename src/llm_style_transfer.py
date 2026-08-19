"""
LLM-based generative style transfer.

A separate pipeline from candidate_selection.py / style_transfer.py's
token-substitution approach: instead of splicing candidate words into masked
slots, this sends a single LLM prompt that combines an LLM-written summary of
a source poem, its distinctive content keywords, the target author's style
profile translated into prose, a relevance-filtered vocabulary palette, tone
exemplar lines in the target author's voice, and explicit structural targets —
and asks the model to write a new poem from scratch in that style.

Reuses existing pipeline data/infrastructure (author_style_profiles.csv,
tfidf_results.csv, word_embeddings.json, poem_tfidf.py, style_narrator.py,
exemplar_selection.py) rather than recomputing any of it; this module owns the
LLM calls, prompt assembly, structural validation, and per-run audit logging.

Usage:
    uv run python src/llm_style_transfer.py \\
        --source-author "Кочо Рацин" --source-title "..." \\
        --target-author "Блаже Конески"
"""

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import exemplar_selection
import poem_tfidf
import style_narrator
from candidate_selection import load_pos_rows, load_tfidf, load_word_embeddings
from llm_client import call

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

log = logging.getLogger('llm_style_transfer')

DEFAULT_MODEL = 'qwen3:14b'
DEFAULT_BACKEND = 'ollama'
DEFAULT_INSTR_LANG = 'en'  # English instructions measured better than Macedonian
                            # ones in llm_probe's own testing; poem content stays
                            # Macedonian throughout regardless of this setting.

CACHE_PATH = DATA / 'poem_summary_cache.json'
LOG_DIR = DATA / 'llm_transfer_logs'

SUMMARY_PROMPT_VERSION = 1
N_CONTENT_KEYWORDS = 10
N_VOCAB_PALETTE = 8
N_EXEMPLAR_CLUSTERS = 5
EXEMPLARS_PER_CLUSTER = 2

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


# ── step 1: poem summary generation, cached ─────────────────────────────────────

_summary_cache = None


def _load_summary_cache() -> dict:
    global _summary_cache
    if _summary_cache is None:
        if CACHE_PATH.exists():
            _summary_cache = json.loads(CACHE_PATH.read_text(encoding='utf-8'))
        else:
            _summary_cache = {}
    return _summary_cache


def _save_summary_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(_summary_cache, ensure_ascii=False, indent=2), encoding='utf-8'
    )


def _summary_cache_key(model: str, poem_text: str) -> str:
    payload = f'{model}|v{SUMMARY_PROMPT_VERSION}|{poem_text}'
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


SUMMARY_PROMPT = (
    'Summarize the content, imagery, and theme of the following Macedonian poem '
    'in 2-3 sentences. Describe WHAT the poem is about, not its style, form, or '
    'word choice. Reply in Macedonian, with the summary only, no explanation.\n\n'
    '{poem_text}'
)


def summarize_poem(poem_text: str, author: str = None, title: str = None,
                   model: str = DEFAULT_MODEL, backend: str = DEFAULT_BACKEND) -> str:
    """
    A short prose summary of a poem's content/imagery/theme (not style).

    Cached per (model, prompt version, poem text) — deterministic given the
    same input, so a batch run never re-summarizes a poem it already has.
    """
    cache = _load_summary_cache()
    key = _summary_cache_key(model, poem_text)
    if key in cache:
        return cache[key]['summary']

    prompt = SUMMARY_PROMPT.format(poem_text=poem_text)
    summary = call(model, prompt, backend=backend).strip()

    cache[key] = {
        'summary': summary, 'author': author, 'title': title,
        'model': model, 'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    _save_summary_cache()
    return summary


# ── step 6: structural targets ──────────────────────────────────────────────────

def load_profile_rows() -> dict:
    def build():
        with open(DATA / 'author_style_profiles.csv', encoding='utf-8') as f:
            return {r['author']: r for r in csv.DictReader(f)}
    return _cached('profile_rows', build)


STRUCTURAL_COLUMNS = [
    'avg_lines_per_poem', 'avg_tokens_per_line',
    'avg_stanzas_per_poem', 'avg_lines_per_stanza',
]


def structural_targets(target_author: str) -> dict:
    """avg_lines_per_poem, avg_tokens_per_line, avg_stanzas_per_poem,
    avg_lines_per_stanza straight off the target author's profile row."""
    rows = load_profile_rows()
    if target_author not in rows:
        raise KeyError(f'no style profile for author={target_author!r}')
    row = rows[target_author]
    return {col: float(row[col]) for col in STRUCTURAL_COLUMNS}


def format_structural_instruction(targets: dict) -> str:
    return (
        f"approximately {targets['avg_lines_per_poem']:.0f} lines across "
        f"{targets['avg_stanzas_per_poem']:.0f} stanzas "
        f"({targets['avg_lines_per_stanza']:.1f} lines per stanza), "
        f"averaging {targets['avg_tokens_per_line']:.1f} words per line"
    )


# ── step 8: generation + structural validation ──────────────────────────────────

def _tokens(line: str) -> list:
    return re.findall(r'\S+', line)


def validate_structure(generated_text: str, targets: dict) -> dict:
    """
    Actual lines/stanzas/tokens-per-line in generated_text vs. targets.
    Measures fit only — never regenerates.
    """
    stanzas = [s for s in generated_text.strip().split('\n\n') if s.strip()]
    lines = [l for l in generated_text.strip().splitlines() if l.strip()]
    token_counts = [len(_tokens(l)) for l in lines]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0

    actual = {
        'actual_lines': len(lines),
        'actual_stanzas': len(stanzas),
        'actual_tokens_per_line': round(avg_tokens, 2),
        'actual_lines_per_stanza': round(len(lines) / len(stanzas), 2) if stanzas else 0.0,
    }
    return {
        **actual,
        'delta_lines': round(actual['actual_lines'] - targets['avg_lines_per_poem'], 2),
        'delta_stanzas': round(actual['actual_stanzas'] - targets['avg_stanzas_per_poem'], 2),
        'delta_tokens_per_line': round(
            actual['actual_tokens_per_line'] - targets['avg_tokens_per_line'], 2),
        'delta_lines_per_stanza': round(
            actual['actual_lines_per_stanza'] - targets['avg_lines_per_stanza'], 2),
    }


# ── step 5: vocabulary palette ──────────────────────────────────────────────────

def surface_to_lemma_map(pos_rows) -> dict:
    """surface word -> its most frequent corpus lemma (majority vote over all
    occurrences) — tfidf_results.csv is surface forms, word_embeddings.json is
    lemma-keyed, so this bridges the two."""
    votes = defaultdict(Counter)
    for r in pos_rows:
        votes[r['word']][r['lemma']] += 1
    return {word: counts.most_common(1)[0][0] for word, counts in votes.items()}


def vocabulary_palette(target_author: str, content_keywords: list,
                       top_k: int = N_VOCAB_PALETTE) -> list:
    """
    Target author's top-20 TF-IDF surface words, cosine-ranked against the
    embedding centroid of content_keywords (already lemmas — from step 2), so
    the palette leans toward words semantically relevant to THIS poem rather
    than the author's vocabulary in general.

    Falls back to plain TF-IDF rank when nothing is embeddable (unknown
    keywords, or an author absent from tfidf_results.csv) rather than raising —
    a degraded palette beats no palette in a single prompt-assembly path.
    """
    author_scores = load_tfidf().get(target_author, {})
    ranked_words = [w for w, _ in sorted(author_scores.items(), key=lambda ws: -ws[1])][:20]

    embeddings = load_word_embeddings()
    keyword_vectors = [embeddings[k] for k in content_keywords if k in embeddings]

    if not keyword_vectors or not ranked_words:
        log.info('vocabulary_palette author=%s: falling back to plain TF-IDF order '
                 '(no embeddable content keywords)', target_author)
        return ranked_words[:top_k]

    centroid = np.mean(keyword_vectors, axis=0)
    lemma_of = surface_to_lemma_map(load_pos_rows())

    scored = []
    for word in ranked_words:
        lemma = lemma_of.get(word, word)
        vec = embeddings.get(lemma)
        sim = float(vec @ centroid) if vec is not None else -1.0
        scored.append((word, sim))
    scored.sort(key=lambda ws: -ws[1])
    return [w for w, _ in scored[:top_k]]


# ── step 7: prompt assembly ──────────────────────────────────────────────────────

PROMPT_TEMPLATE = {
    'en': (
        'Rewrite this poem\'s content in another poet\'s style.\n\n'
        'Source poem summary: {summary}\n\n'
        'Key content/imagery to preserve: {content_keywords}\n\n'
        'Target author: {target_author}\n\n'
        'Style profile: {style_text}\n\n'
        "Characteristic vocabulary (use where natural, don't force all of them): "
        '{vocab_palette}\n\n'
        "Example lines in {target_author}'s voice (tone/rhythm reference only — "
        'do not reuse their imagery or specific phrasing):\n{exemplar_lines}\n\n'
        f"Instruction: Write a new poem conveying the content above, in "
        "{target_author}'s style as described. Do not reuse the example lines "
        'or their specific images. Aim for {structural_instruction}.'
    ),
}


def assemble_prompt(source_poem_text: str, source_poem_id, source_author: str,
                    source_title: str, target_author: str, model: str = DEFAULT_MODEL,
                    backend: str = DEFAULT_BACKEND, instr_lang: str = DEFAULT_INSTR_LANG,
                    n_keywords: int = N_CONTENT_KEYWORDS,
                    n_vocab: int = N_VOCAB_PALETTE,
                    n_clusters: int = N_EXEMPLAR_CLUSTERS,
                    per_cluster: int = EXEMPLARS_PER_CLUSTER) -> dict:
    """
    Steps 1-7: build the full prompt for one source poem -> target author
    transfer. Returns {'prompt', 'summary', 'content_keywords', 'style_text',
    'vocab_palette', 'exemplars', 'structural_targets'} — the prompt string for
    sending to the model, everything else for logging/audit.

    exemplars keep their song_title for the audit trail (the returned dict),
    but the poem attribution is stripped from what actually goes in the prompt
    text — the model sees tone/rhythm reference lines, not "here's poem X".
    """
    summary = summarize_poem(source_poem_text, author=source_author,
                             title=source_title, model=model, backend=backend)
    content_keywords = poem_tfidf.top_content_words(source_poem_id, n=n_keywords)
    style_text = style_narrator.describe_style(target_author)
    vocab_palette = vocabulary_palette(target_author, content_keywords, top_k=n_vocab)
    exemplars = exemplar_selection.select_exemplars(
        target_author, k=n_clusters, per_cluster=per_cluster)
    targets = structural_targets(target_author)

    exemplar_lines = '\n'.join(f'  {e["line"]}' for e in exemplars)

    prompt = PROMPT_TEMPLATE[instr_lang].format(
        summary=summary,
        content_keywords=', '.join(content_keywords),
        target_author=target_author,
        style_text=style_text,
        vocab_palette=', '.join(vocab_palette),
        exemplar_lines=exemplar_lines,
        structural_instruction=format_structural_instruction(targets),
    )

    return {
        'prompt': prompt,
        'summary': summary,
        'content_keywords': content_keywords,
        'style_text': style_text,
        'vocab_palette': vocab_palette,
        'exemplars': exemplars,
        'structural_targets': targets,
    }


def generate_poem(prompt: str, model: str = DEFAULT_MODEL,
                  backend: str = DEFAULT_BACKEND) -> str:
    """Step 7's second half: actually send the assembled prompt to the LLM."""
    return call(model, prompt, backend=backend).strip()


# ── step 9: orchestration + per-run logging ─────────────────────────────────────

def _load_stripped_songs_indexed() -> dict:
    """poem_id (str) -> {'author', 'song_title', 'song_text'} (that row's dict, as read)."""
    def build():
        with open(DATA / 'stripped_songs.csv', encoding='utf-8') as f:
            return {r['poem_id']: r for r in csv.DictReader(f)}
    return _cached('stripped_songs_by_id', build)


def load_poem_text_by_id(poem_id) -> str:
    """A poem's text by its unambiguous poem_id."""
    rows = _load_stripped_songs_indexed()
    row = rows.get(str(poem_id))
    if row is None:
        raise KeyError(f'no poem with poem_id={poem_id!r}')
    return row['song_text']


def resolve_poem_id(author: str, song_title: str) -> str:
    """
    The single poem_id matching (author, song_title).

    (author, song_title) is NOT a unique key in stripped_songs.csv -- ~15 pairs
    are shared by multiple distinct poems (e.g. Конески has 4 poems titled
    'ПЕСНА'; see CLAUDE.md's Data quality context). Raises rather than guessing
    on an ambiguous title -- the error lists the actual candidate poem_ids so
    the caller can pick one via --source-poem-id.
    """
    by_id = _load_stripped_songs_indexed()
    matches = sorted(
        (pid for pid, r in by_id.items()
         if r['author'] == author and r['song_title'] == song_title),
        key=int,
    )
    if not matches:
        raise KeyError(f'no poem for author={author!r} song_title={song_title!r}')
    if len(matches) > 1:
        raise ValueError(
            f'{len(matches)} poems share author={author!r} song_title={song_title!r} in '
            "stripped_songs.csv — (author, song_title) isn't a unique poem key here. "
            f'Use --source-poem-id to pick one of: {matches}.'
        )
    return matches[0]


def load_poem_text(author: str, song_title: str) -> str:
    """A poem's text by (author, song_title) -- raises if the title is ambiguous;
    see resolve_poem_id(). Use load_poem_text_by_id() when the poem_id is already
    known (e.g. from --source-poem-id)."""
    return load_poem_text_by_id(resolve_poem_id(author, song_title))


def _safe(s: str) -> str:
    return re.sub(r'[^\w-]+', '_', s, flags=re.UNICODE).strip('_')


def log_run(record: dict) -> Path:
    """Write one run's full audit record to LOG_DIR as JSON. Returns the path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = record['timestamp'].replace(':', '').replace('+00:00', 'Z')
    name = f"{ts}__{_safe(record['source_author'])}__{_safe(record['target_author'])}.json"
    path = LOG_DIR / name
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def run_transfer(source_author: str, source_title: str, target_author: str,
                 model: str = DEFAULT_MODEL, backend: str = DEFAULT_BACKEND,
                 instr_lang: str = DEFAULT_INSTR_LANG, source_poem_id=None,
                 **assemble_kw) -> dict:
    """
    Full step 1-9 chain for one source poem -> target author transfer.

    source_poem_id, when given, identifies the source poem directly --
    source_author/source_title are then overwritten from the poem's own record
    (safe to pass placeholders for them in that case). This is what actually
    reaches every one of the corpus's ~15 same-titled poem groups, since
    (author, song_title) alone can't. Without it, source_author/source_title
    resolve through resolve_poem_id(), which raises on an ambiguous title.
    """
    poem_id = source_poem_id if source_poem_id is not None else resolve_poem_id(
        source_author, source_title)
    source_text = load_poem_text_by_id(poem_id)
    source_row = _load_stripped_songs_indexed()[str(poem_id)]
    source_author, source_title = source_row['author'], source_row['song_title']

    assembled = assemble_prompt(source_text, poem_id, source_author, source_title,
                                target_author, model=model, backend=backend,
                                instr_lang=instr_lang, **assemble_kw)
    generated = generate_poem(assembled['prompt'], model=model, backend=backend)
    fit = validate_structure(generated, assembled['structural_targets'])

    record = {
        **assembled,
        'generated_poem': generated,
        'structural_fit': fit,
        'source_poem_id': poem_id,
        'source_author': source_author,
        'source_title': source_title,
        'target_author': target_author,
        'model': model,
        'backend': backend,
        'instr_lang': instr_lang,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    log_path = log_run(record)
    log.info('transfer source=%s (%s, poem_id=%s) -> target=%s  lines=%d/%d stanzas=%d/%d  log=%s',
            source_author, source_title, poem_id, target_author,
            fit['actual_lines'], assembled['structural_targets']['avg_lines_per_poem'],
            fit['actual_stanzas'], assembled['structural_targets']['avg_stanzas_per_poem'],
            log_path)
    return {**record, 'log_path': str(log_path)}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-author', default='Кочо Рацин')
    parser.add_argument('--source-title')
    parser.add_argument('--source-poem-id', default=None)
    parser.add_argument('--target-author', default='Блаже Конески')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--backend', default=DEFAULT_BACKEND, choices=['ollama', 'openrouter'])
    parser.add_argument('--lang', default=DEFAULT_INSTR_LANG, choices=['en', 'mk'],
                       dest='instr_lang')
    args = parser.parse_args()

    if args.source_poem_id is None and not args.source_title:
        # No title or id given: pick this author's first poem in the corpus.
        with open(DATA / 'stripped_songs.csv', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['author'] == args.source_author:
                    args.source_title = row['song_title']
                    break
        if not args.source_title:
            sys.exit(f'No poems found for source author {args.source_author!r}')

    try:
        result = run_transfer(args.source_author, args.source_title, args.target_author,
                              model=args.model, backend=args.backend, instr_lang=args.instr_lang,
                              source_poem_id=args.source_poem_id)
    except KeyError as e:
        sys.exit(e.args[0])  # KeyError.__str__ would otherwise double-quote the message
    except ValueError as e:
        sys.exit(str(e))

    print(f"\n=== {args.source_author} — {args.source_title} → {args.target_author} ===\n")
    print(result['generated_poem'])
    print(f"\n[structural fit] {result['structural_fit']}")
    print(f"[log] {result['log_path']}")
