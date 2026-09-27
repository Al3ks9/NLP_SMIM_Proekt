"""
GRPO rewards for the Qwen style-transfer pipeline (stage 2, after SFT).

For one generated poem, given the source poem and the target author:

- style: the poem's per-poem features (style_features.poem_features -- the
  same definitions as author_style_profiles.csv) scored against the target
  author's per-poem distribution over TRAIN poems (reward_calibration.json),
  plus the stylometric classifier's probability and TF-IDF palette uptake;
  weighted mean over feature groups.
- content: line-level embedding F1 against the SOURCE poem (never the Gemma
  output), rescaled so unrelated poems score 0.
- gates (multiplicative, style_metrics): validity = cyrillic * no markup *
  non-empty on both; line_uniqueness on style; copy_novelty on content.

R = style_weight * R_style_g + content_weight * R_content_g. Every component
is returned (and, during training, logged) separately. See
docs/superpowers/specs/2026-09-27-grpo-stage-design.md §4.

Inputs are only (generated poem, source poem, target author) -- no reference
answer -- so there is no path for the Gemma output or any val/test poem to
leak into the reward.
"""

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import style_features as sf
import style_metrics as sm
from content_similarity import DEFAULT_CONTENT_MODEL, LineEmbedder, line_f1

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIBRATION_PATH = DATA / 'reward_calibration.json'

STYLE_GROUPS = {
    'pos_profile': ('pct_NOUN', 'pct_VERB', 'pct_ADJ', 'pct_ADV', 'pct_PROPN'),
    'lexical_diversity': ('lexical_diversity',),
    'structure': ('avg_tokens_per_line', 'num_lines', 'num_stanzas', 'avg_lines_per_stanza'),
    'pos_bigrams': ('pos_bigrams',),
    'classifier': ('clf_target_prob',),
    'rhyme': ('rhyme_endings',),
    'tfidf': ('tfidf',),
}
DEFAULT_STYLE_WEIGHTS = {'pos_profile': 1.0, 'lexical_diversity': 1.0, 'structure': 1.0,
                         'pos_bigrams': 1.0, 'classifier': 1.0, 'rhyme': 0.5, 'tfidf': 0.5}

# Prefixes pushed to the trainer's logged metrics each step (raw/z/score go
# to the JSONL only -- dozens of per-feature curves would bury the plot).
_LOGGED_PREFIXES = ('R', 'group/', 'gate/', 'content/')


@dataclass
class RewardConfig:
    style_weights: dict = field(default_factory=lambda: dict(DEFAULT_STYLE_WEIGHTS))
    style_weight: float = 0.5
    content_weight: float = 0.5
    copy_overlap_threshold: float = sm.COPY_OVERLAP_THRESHOLD
    sigma_floor_frac: float = 0.5
    content_model: str = DEFAULT_CONTENT_MODEL
    calibration_path: Path = CALIBRATION_PATH
    components_log_path: Path = None

    def __post_init__(self):
        unknown = set(self.style_weights) - set(STYLE_GROUPS)
        if unknown:
            raise ValueError(f'unknown style group(s) {sorted(unknown)}; '
                             f'known: {sorted(STYLE_GROUPS)}')
        if sum(self.style_weights.values()) <= 0:
            raise ValueError('style_weights must have a positive total')


_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def load_calibration(path: Path) -> dict:
    return _cached(('calibration', str(path)),
                   lambda: json.loads(Path(path).read_text(encoding='utf-8')))


def _default_tagger():
    import tagging

    def build():
        with open(DATA / 'pos_tagged.csv', encoding='utf-8') as f:
            verb_lemmas = tagging.load_verb_lemmas(csv.DictReader(f))
        return lambda text: tagging.tag_poem(text, verb_lemmas)
    return _cached('tagger', build)


def _default_embedder(model_name: str) -> LineEmbedder:
    return _cached(('embedder', model_name), lambda: LineEmbedder(model_name))


# ── Scoring primitives ─────────────────────────────────────────────────────────

def gaussian_score(x, mean, std, corpus_std, floor_frac) -> tuple:
    """(z, exp(-z^2/2)); sigma floored at floor_frac * corpus-wide per-poem std
    so a few-poem or unusually uniform author doesn't zero every small miss."""
    sigma = max(std, floor_frac * corpus_std, 1e-6)
    z = (x - mean) / sigma
    return z, math.exp(-0.5 * z * z)


def one_sided_score(c, mean, std) -> float:
    """Distribution similarity: 1 at or above the author's own typical
    self-similarity, Gaussian decay below it."""
    if c >= mean:
        return 1.0
    z = (c - mean) / max(std, 1e-6)
    return math.exp(-0.5 * z * z)


def content_score(f1: float, baseline: float) -> float:
    return min(1.0, max(0.0, (f1 - baseline) / (1.0 - baseline)))


def style_components(features: dict, clf_prob: float, tfidf_hit: float,
                     author_cal: dict, corpus_std: dict, cfg: RewardConfig) -> dict:
    out, scores = {}, {}
    for f in sf.SCALAR_FEATURES:
        st = author_cal['scalar'][f]
        z, s = gaussian_score(features[f], st['mean'], st['std'], corpus_std[f],
                              cfg.sigma_floor_frac)
        out[f'raw/{f}'], out[f'z/{f}'], scores[f] = features[f], z, s
    for f in sf.DIST_FEATURES:
        d = author_cal['dist'][f]
        c = sf.dist_cosine(features[f], d['dist'])
        out[f'raw/{f}'] = c
        scores[f] = one_sided_score(c, d['cos_mean'], d['cos_std'])
    out['raw/clf_target_prob'] = scores['clf_target_prob'] = clf_prob
    out['raw/tfidf'] = tfidf_hit
    scores['tfidf'] = min(1.0, tfidf_hit / author_cal['tfidf_hit_p90'])
    for f, s in scores.items():
        out[f'score/{f}'] = s

    num = den = 0.0
    for group, members in STYLE_GROUPS.items():
        g = sum(scores[m] for m in members) / len(members)
        out[f'group/{group}'] = g
        w = cfg.style_weights.get(group, 0.0)
        num, den = num + w * g, den + w
    out['R_style'] = num / den
    return out


def gates(source: str, text: str, cfg: RewardConfig) -> dict:
    lang = sm.language_metrics(text)
    return {
        'gate/is_cyrillic': lang['is_cyrillic'],
        'gate/no_markup': lang['no_markup'],
        'gate/is_nonempty': lang['is_nonempty'],
        'gate/validity': lang['is_cyrillic'] * lang['no_markup'] * lang['is_nonempty'],
        'gate/line_uniqueness': sm.repetition_metrics(text)['line_uniqueness'],
        'gate/copy_novelty': sm.copy_novelty(source, text, cfg.copy_overlap_threshold),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def score_batch(sources: list, generations: list, targets: list, cfg: RewardConfig = None,
                tagger=None, embedder=None) -> list:
    cfg = cfg or RewardConfig()
    cal = load_calibration(cfg.calibration_path)
    missing = sorted({t for t in targets if t not in cal['authors']})
    if missing:
        raise KeyError(f'no reward calibration for target author(s) {missing} -- '
                       f'rebuild {Path(cfg.calibration_path).name} or pick profiled authors')
    tagger = tagger or _default_tagger()
    embedder = embedder or _default_embedder(cfg.content_model)

    rows = []
    for src, gen, tgt in zip(sources, generations, targets, strict=True):
        feats = sf.poem_features(tagger(gen) if sm.lines(gen) else [], gen)
        match = sm.style_match(gen, tgt)
        style = style_components(feats, match['clf_target_prob'], match['tfidf_hit_rate'],
                                 cal['authors'][tgt], cal['corpus_std'], cfg)
        p, r, f1 = line_f1(embedder.embed_lines(sm.lines(src), cache=True),
                           embedder.embed_lines(sm.lines(gen), cache=False))
        g = gates(src, gen, cfg)
        r_content = content_score(f1, cal['content_baseline'])
        r_style_g = g['gate/validity'] * g['gate/line_uniqueness'] * style['R_style']
        r_content_g = g['gate/validity'] * g['gate/copy_novelty'] * r_content
        rows.append({
            'target_author': tgt, **style, **g,
            'content/precision': p, 'content/recall': r, 'content/f1': f1,
            'R_content': r_content, 'R_style_g': r_style_g, 'R_content_g': r_content_g,
            'R': cfg.style_weight * r_style_g + cfg.content_weight * r_content_g,
        })
    return rows


def score(source: str, generated: str, target: str, cfg: RewardConfig = None,
          tagger=None, embedder=None) -> dict:
    return score_batch([source], [generated], [target], cfg, tagger, embedder)[0]


def _completion_text(completion) -> str:
    """TRL hands plain strings for string prompts, message lists for
    conversational ones; grpo_data uses string prompts."""
    if isinstance(completion, str):
        return completion
    return completion[-1]['content']


def make_reward_funcs(cfg: RewardConfig = None, tagger=None, embedder=None) -> tuple:
    """
    (style_reward, content_reward) with TRL's reward-function signature.
    Both read one memoised score_batch per generation batch, so classla and
    the embedder run once per completion per step, not once per function.
    Dataset columns source_text / target_author / source_poem_id arrive as
    keyword lists; log_metric / trainer_state are TRL's.
    """
    cfg = cfg or RewardConfig()
    memo = {}

    def _rows(completions, source_text, target_author, kwargs):
        texts = [_completion_text(c) for c in completions]
        key = (tuple(texts), tuple(source_text), tuple(target_author))
        if memo.get('key') != key:
            memo.update(key=key, logged=False,
                        rows=score_batch(list(source_text), texts, list(target_author),
                                         cfg, tagger, embedder))
        if not memo['logged']:
            memo['logged'] = True
            _log(memo['rows'], texts, kwargs)
        return memo['rows']

    def _log(rows, texts, kwargs):
        log_metric = kwargs.get('log_metric')
        numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        if log_metric:
            for k in numeric:
                if k.startswith(_LOGGED_PREFIXES):
                    log_metric(k, sum(r[k] for r in rows) / len(rows))
        if cfg.components_log_path:
            state = kwargs.get('trainer_state')
            step = state.global_step if state is not None else None
            ids = kwargs.get('source_poem_id') or [None] * len(rows)
            path = Path(cfg.components_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                for row, text, pid in zip(rows, texts, ids):
                    record = {'step': step, 'source_poem_id': pid,
                              'target_author': row['target_author'], 'completion': text,
                              **{k: float(row[k]) for k in numeric}}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def style_reward(prompts, completions, source_text, target_author, **kwargs):
        return [r['R_style_g'] for r in _rows(completions, source_text, target_author, kwargs)]

    def content_reward(prompts, completions, source_text, target_author, **kwargs):
        return [r['R_content_g'] for r in _rows(completions, source_text, target_author, kwargs)]

    return style_reward, content_reward
