"""
Human evaluation: build the blind rating page, summarise the ratings.

Twenty test pairs (two per target author), each shown once per system and once
as a real-poem control: the target author's train poem closest to the source by
the content reward's line-level F1. The rater sees the source, the poem and the
target's name, never the system; items_key.json maps item ids back to systems.
Each item also carries a panel of three real train poems by the target, so a
rater who doesn't know the author can read their voice first. The control is
never one of the panel poems (the rater would recognise it).

Usage:
    uv run python -m students.shared.human_eval build \\
        --system teacher=data/synthetic/teacher_test.csv \\
        --system qwen-sft=data/validation_generations/qwen3-sft-merged_test.csv \\
        --system qwen-grpo-v2=data/validation_generations/qwen3-grpo-v2_test.csv \\
        --system gemma-sft=data/validation_generations/gemma4-e4b-sft-merged_test.csv \\
        --system gemma-grpo=data/validation_generations/gemma4-e4b-grpo_test.csv \\
        --out-dir data/human_eval
    uv run python -m students.shared.human_eval summarize
"""

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from common.corpus.make_splits import load_stripped_songs, poems_in_split

CONTROL = 'real-poem'
PAGE_DIR = Path(__file__).parent / 'human_eval_page'


def clean_poem(text: str) -> str:
    """One display form for every poem on the page: corpus poems carry leading
    blank lines and runs of empty lines that generated ones don't, which would
    give the real-poem control away."""
    text = text.replace('\r\n', '\n').strip()
    return re.sub(r'\n\s*\n(\s*\n)+', '\n\n', text)


def _lines(text: str) -> list:
    return [l.strip() for l in text.splitlines() if l.strip()]


def load_rows(path) -> list:
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def select_pairs(system_rows: dict, per_author: int = 2, seed: int = 42) -> list:
    """per_author (source_poem_id, target_author) pairs per target, drawn only
    from pairs every system produced."""
    common = None
    for rows in system_rows.values():
        keys = {(r['source_poem_id'], r['target_author']) for r in rows}
        common = keys if common is None else common & keys
    by_target = defaultdict(list)
    for key in sorted(common or ()):
        by_target[key[1]].append(key)
    all_targets = {r['target_author'] for rows in system_rows.values() for r in rows}
    rng = random.Random(seed)
    chosen = []
    for target in sorted(all_targets):
        pool = by_target.get(target, [])
        if len(pool) < per_author:
            raise ValueError(f'only {len(pool)} pairs for {target} are present in every system')
        chosen += rng.sample(pool, per_author)
    return chosen


def panel_poems(author: str, k: int = 3, seed: int = 42,
                min_lines: int = 8, max_lines: int = 30) -> list:
    """k train poems by author, preferring min_lines..max_lines non-empty lines."""
    pool = sorted((p for p in poems_in_split('train') if p['author'] == author),
                  key=lambda p: int(p['poem_id']))
    preferred = [p for p in pool if min_lines <= len(_lines(p['song_text'])) <= max_lines]
    rng = random.Random(seed)
    if len(preferred) >= k:
        return rng.sample(preferred, k)
    rest = [p for p in pool if p not in preferred]
    return preferred + rng.sample(rest, min(k - len(preferred), len(rest)))


def _default_f1():
    from common.evaluation.content_similarity import LineEmbedder, line_f1
    embedder = LineEmbedder()

    def f1(src: str, cand: str) -> float:
        return line_f1(embedder.embed_lines(_lines(src), cache=True),
                       embedder.embed_lines(_lines(cand), cache=True))[2]
    return f1


def control_poem(source_text: str, author: str, exclude_ids: set, f1=None) -> dict:
    """The author's train poem with the highest line-F1 to the source."""
    f1 = f1 or _default_f1()
    pool = [p for p in poems_in_split('train')
            if p['author'] == author and p['poem_id'] not in exclude_ids]
    return max(pool, key=lambda p: (f1(source_text, p['song_text']), -int(p['poem_id'])))


def _item_id(seed: int, system: str, pid: str, target: str) -> str:
    return hashlib.sha1(f'{seed}|{system}|{pid}|{target}'.encode()).hexdigest()[:10]


def page_version(items: list) -> str:
    """Changes when any item, poem or source text changes, so ratings from a
    page built on other outputs are rejected rather than joined."""
    parts = sorted(f"{it['item_id']}|{it['poem']}|{it['source']['text']}" for it in items)
    return hashlib.sha1('\x1f'.join(parts).encode()).hexdigest()[:12]


def build_items(system_rows: dict, systems: list, seed: int = 42, f1=None,
                reveal: bool = False) -> tuple:
    """(items, key, panels). items carry no system; key = {'page_version',
    'items': {item_id: {system, source_poem_id, target_author}}}; panels =
    {author: [{title, text}, ...]}. Each item's `group` is its pair's index,
    so the page can keep a pair's outputs apart. With `reveal`, each item also
    carries the answer the page shows once it is rated (not blind any more)."""
    f1 = f1 or _default_f1()
    poems = {p['poem_id']: p for p in load_stripped_songs()}
    generated = {s: {(r['source_poem_id'], r['target_author']): r['generated_poem']
                     for r in system_rows[s]} for s in systems}
    pairs = select_pairs({s: system_rows[s] for s in systems}, seed=seed)

    panel_rows = {t: panel_poems(t, seed=seed) for t in sorted({t for _, t in pairs})}
    panels = {t: [{'title': p['song_title'], 'text': clean_poem(p['song_text'])} for p in rows]
              for t, rows in panel_rows.items()}
    used_controls = defaultdict(set)

    items, key = [], {}
    for group, (pid, target) in enumerate(pairs):
        src = poems[pid]
        source = {'author': src['author'], 'title': src['song_title'],
                  'text': clean_poem(src['song_text'])}
        # neither a panel poem nor the target's other control: a repeated
        # "generated" poem would give the control away
        control = control_poem(src['song_text'], target,
                               {p['poem_id'] for p in panel_rows[target]} | used_controls[target],
                               f1=f1)
        used_controls[target].add(control['poem_id'])
        outputs = [(s, generated[s][(pid, target)]) for s in systems]
        outputs.append((CONTROL, control['song_text']))
        for system, poem in outputs:
            item_id = _item_id(seed, system, pid, target)
            items.append({'item_id': item_id, 'group': group, 'target_author': target,
                          'source': source, 'poem': clean_poem(poem)})
            key[item_id] = {'system': system, 'source_poem_id': pid, 'target_author': target}
            if system == CONTROL:
                key[item_id]['control_poem_id'] = control['poem_id']
            if reveal:
                items[-1]['reveal'] = (f"Оригинална песна — {target}, „{control['song_title']}“"
                                       if system == CONTROL else f'Генерирана песна — {system}')
    return items, {'page_version': page_version(items), 'items': key}, panels


def _script_safe(text: str) -> str:
    """Text that can sit inside an inline <script> without ending it."""
    return (text.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))


def render_page(items: list, panels: dict, page_version: str) -> str:
    """The self-contained rating page: page.html with the data and page.js inlined."""
    data = json.dumps({'version': page_version, 'items': items, 'panels': panels},
                      ensure_ascii=False)
    js = (PAGE_DIR / 'page.js').read_text(encoding='utf-8').replace('</', '<\\/')
    html = (PAGE_DIR / 'page.html').read_text(encoding='utf-8')
    return html.replace('/*__DATA__*/', _script_safe(data)).replace('/*__JS__*/', js)


def build(system_specs: list, out_dir: Path, seed: int = 42, reveal: bool = False) -> Path:
    """system_specs: ['label=path.csv', ...] in display-independent order."""
    systems, rows = [], {}
    for spec in system_specs:
        label, path = spec.split('=', 1)
        systems.append(label)
        rows[label] = load_rows(path)
    items, key, panels = build_items(rows, systems, seed=seed, reveal=reveal)
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / 'rating_page.html'
    page.write_text(render_page(items, panels, key['page_version']), encoding='utf-8')
    (out_dir / 'items_key.json').write_text(
        json.dumps({**key, 'systems': {s: dict(p.split('=', 1) for p in system_specs)[s]
                                       for s in systems}, 'seed': seed, 'reveal': reveal},
                   ensure_ascii=False, indent=1), encoding='utf-8')
    return page


# ── summarising ratings ──────────────────────────────────────────────────────

SCALES = ['macedonian', 'coherent', 'poetic', 'content', 'author', 'usable']


def krippendorff_alpha_interval(units: list) -> float:
    """Interval-metric Krippendorff's alpha. Each unit is the values the raters
    gave one item; units with fewer than two values are not pairable."""
    units = [u for u in units if len(u) >= 2]
    values = [v for u in units for v in u]
    n = len(values)
    if len(units) < 2:
        return float('nan')
    d_o = sum(sum((a - b) ** 2 for i, a in enumerate(u) for j, b in enumerate(u) if i != j)
              / (len(u) - 1) for u in units) / n
    d_e = sum((a - b) ** 2 for i, a in enumerate(values) for j, b in enumerate(values)
              if i != j) / (n * (n - 1))
    return 1.0 if d_e == 0 else 1 - d_o / d_e


def _bootstrap_ci(xs: list, seed: int, n_boot: int = 2000) -> tuple:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def summarize(ratings_dir: Path, key_path: Path, seed: int = 42) -> list:
    """Per system x scale: mean over items (each item averaged over raters)
    with a bootstrap 95 % interval; with two or more raters, alpha per scale
    over all items. The control's content score is left out: a real poem by
    the target is not a rewrite of the source."""
    key = json.loads(Path(key_path).read_text(encoding='utf-8'))
    # a rater who exports twice leaves ratings_X.json and ratings_X (1).json:
    # one rating session is (rater, started), and its fullest export wins
    sessions = {}
    for f in sorted(Path(ratings_dir).glob('ratings_*.json')):
        data = json.loads(f.read_text(encoding='utf-8'))
        if data.get('page_version') != key['page_version']:
            raise ValueError(f'{f.name} was rated on page version {data.get("page_version")}, '
                             f'not {key["page_version"]}')
        session = (data.get('rater'), data.get('started'))
        if session not in sessions or len(data['ratings']) > len(sessions[session][1]['ratings']):
            sessions[session] = (f, data)
    files = [f for f, _ in sessions.values()]
    by_item = defaultdict(lambda: defaultdict(list))       # item -> scale -> [values per rater]
    for f, data in sessions.values():
        for r in data['ratings']:
            if r['item_id'] not in key['items']:
                raise ValueError(f'{f.name}: unknown item {r["item_id"]}')
            for scale, v in r['scores'].items():
                by_item[r['item_id']][scale].append(v)

    rows = []
    systems = sorted({v['system'] for v in key['items'].values()})
    for system in systems:
        for scale in SCALES:
            if system == CONTROL and scale == 'content':
                continue
            xs = [sum(vals) / len(vals) for item, scales in by_item.items()
                  if key['items'][item]['system'] == system for vals in [scales.get(scale)] if vals]
            if not xs:
                continue
            lo, hi = _bootstrap_ci(xs, seed)
            rows.append({'system': system, 'scale': scale, 'n': len(xs),
                         'mean': sum(xs) / len(xs), 'ci_low': lo, 'ci_high': hi})
    if len(files) >= 2:
        for scale in SCALES:
            units = [scales[scale] for item, scales in by_item.items()
                     if scale in scales and not (scale == 'content' and
                                                 key['items'][item]['system'] == CONTROL)]
            rows.append({'system': 'ALL', 'scale': scale,
                         'alpha': krippendorff_alpha_interval(units)})
    return rows


if __name__ == '__main__':
    import argparse
    from common.paths import ROOT
    parser = argparse.ArgumentParser(description='Build the human-evaluation page or summarise ratings.')
    sub = parser.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--system', action='append', required=True, metavar='LABEL=CSV')
    b.add_argument('--out-dir', type=Path, default=ROOT / 'data' / 'human_eval')
    b.add_argument('--seed', type=int, default=42)
    b.add_argument('--reveal', action='store_true',
                   help='show which system wrote each poem once it is rated (not blind after that)')
    m = sub.add_parser('summarize')
    m.add_argument('--dir', type=Path, default=ROOT / 'data' / 'human_eval')
    args = parser.parse_args()
    if args.cmd == 'build':
        print(build(args.system, args.out_dir, seed=args.seed, reveal=args.reveal))
    else:
        rows = summarize(args.dir, args.dir / 'items_key.json')
        out = args.dir / 'summary.csv'
        with open(out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['system', 'scale', 'n', 'mean', 'ci_low',
                                                   'ci_high', 'alpha'])
            writer.writeheader()
            writer.writerows(rows)
        means = {(r['system'], r['scale']): r['mean'] for r in rows if 'mean' in r}
        for system in sorted({r['system'] for r in rows if 'mean' in r}):
            cells = [f"{means[(system, sc)]:.2f}" if (system, sc) in means else '---' for sc in SCALES]
            print(f"{system} & " + ' & '.join(cells) + r' \\')
        print(out)
