"""
Shared harness for probing LLMs on Macedonian style-transfer tasks.

Not part of the pipeline — this is a throwaway diagnostic that answers one
question before we design anything: can the model actually handle Macedonian
well enough to be the filling stage of a PEGF-style transfer?

Two backends: 'ollama' (local, no limits — the primary target) and 'openrouter'
(remote, quota-limited — kept as a quality ceiling to compare a local 8B against).

Five probes, five API calls per model:
  1. fluency    — can it write Macedonian at all, without drifting to BG/SR?
  2. infill_implicit — PEGF template (a): style words wrapped in [brackets]
  3. infill_explicit — PEGF template (b): style words replaced by [MASK]
  4. vanilla    — PEGF template (c): full rewrite, no masking (the baseline)
  5. morphology — can it inflect a lemma correctly? (i.e. replace morph_lookup.json)

Probes 2/3/4 mirror the paper's own ablation, so we get their central claim
(implicit > explicit > vanilla) tested on our data rather than on Yelp.

The openrouter backend requires OPENROUTER_API_KEY. Uses stdlib only — no new deps.
"""

import re
import time
from pathlib import Path

from llm_client import call

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

# ── Language identity checks ──────────────────────────────────────────────────
# INFORMATIONAL ONLY — never a pass/fail verdict. Much of the corpus predates the
# officialization of Macedonian, and the authors freely used words from neighbouring
# Balkan languages; regional speech carries the same influences (Bulgarian in the
# east, Serbian in the north). So Bulgarian/Serbian lexis is period-authentic, not
# a defect. Only a Macedonian speaker can judge whether output reads well.
# What we *can* check mechanically is Latin-script contamination and whether the
# model produced Macedonian-specific letters at all.

MK_ONLY = set('ѓќѕ')          # Macedonian-specific
FOREIGN = set('ъщюяьыэё') | set('ђћ')  # Bulgarian-only | Serbian-only
CYRILLIC = re.compile(r'[Ѐ-ӿ]')
LATIN = re.compile(r'[A-Za-z]')


def script_report(text: str) -> dict:
    """Character-level evidence for which Cyrillic language this actually is."""
    cyr = len(CYRILLIC.findall(text))
    lat = len(LATIN.findall(text))
    lower = text.lower()
    return {
        'cyrillic': cyr,
        'latin': lat,
        'cyr_ratio': round(cyr / max(cyr + lat, 1), 3),
        'mk_only_found': sorted({c for c in lower if c in MK_ONLY}),
        'foreign_found': sorted({c for c in lower if c in FOREIGN}),
    }


# ── Infilling checks ──────────────────────────────────────────────────────────

def _tokens(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def infill_report(masked_src: str, output: str, mask_token: str = None) -> dict:
    """
    Did the model edit *only* the masked region?

    Measures the core PEGF claim: content outside the editing region should
    survive untouched, and the masked slots should be resolved.
    """
    bracketed = re.findall(r'\[([^\]]+)\]', masked_src)
    stripped = re.sub(r'[\[\]]', ' ', masked_src)
    if mask_token:
        stripped = stripped.replace(mask_token.strip('[]'), ' ')

    frozen = [t for t in _tokens(stripped) if t not in {b.lower() for b in bracketed}]
    out_tokens = set(_tokens(output))
    kept = sum(1 for t in frozen if t in out_tokens)

    return {
        'src_lines': len([l for l in masked_src.splitlines() if l.strip()]),
        'out_lines': len([l for l in output.splitlines() if l.strip()]),
        'content_kept': round(kept / max(len(frozen), 1), 3),
        'markers_left': output.count('[') + output.count('MASK'),
        'preamble': bool(re.match(r'^\s*(here|еве|ова|sure|of course)', output, re.I)),
    }


# ── Test material ─────────────────────────────────────────────────────────────

FALLBACK_STANZA = """Натаму – в поле битолско
чемрее врба проклета –
под врбата незнаен гроб,
в гроб лежи војник непознат."""

# Content words to mask. Chosen as genuine style-bearing words in the stanza,
# standing in for what should_replace() picks in style_transfer.py.
MASK_WORDS = ['чемрее', 'проклета', 'незнаен']

# Конески's TF-IDF top words — the data-derived style spec we feed the model
# instead of a hand-written style label.
TARGET_AUTHOR = 'Блаже Конески'
SOURCE_AUTHOR = 'Кочо Рацин'


def load_stanza() -> str:
    """First stanza of a real Рацин poem from the corpus, or a hardcoded fallback."""
    import csv
    path = DATA / 'stripped_songs.csv'
    if not path.exists():
        return FALLBACK_STANZA
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if 'Рацин' in row['author']:
                stanzas = [s for s in row['song_text'].strip().split('\n\n') if s.strip()]
                for s in stanzas:
                    if len(s.strip().splitlines()) >= 4:
                        return s.strip()
    return FALLBACK_STANZA


def load_target_words(n: int = 15, author: str | None = None) -> list[str]:
    """Author's distinctive vocabulary from our TF-IDF output. Defaults to
    TARGET_AUTHOR (this module's single-author demo) when no author is given —
    callers doing multi-author work (candidate_selection.tier_legacy_fallback)
    must pass one explicitly, or every target silently gets Конески's words."""
    import csv
    author = author or TARGET_AUTHOR
    path = DATA / 'tfidf_results.csv'
    if not path.exists():
        return ['вик', 'мрачен', 'сокол', 'младост', 'самотен', 'тивка']
    words = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['author'] == author and int(row['rank']) <= n:
                words.append(row['word'])
    return words


def _mask(stanza: str, style: str) -> str:
    """Wrap (implicit) or replace (explicit) the style words in the stanza."""
    out = stanza
    for w in MASK_WORDS:
        if w in out:
            out = out.replace(w, f'[{w}]' if style == 'implicit' else '[MASK]', 1)
    return out


def _slots(stanza: str) -> list[dict]:
    """
    Per-slot candidate lists for the masked words (candidate_selection §5).

    The masking step already knows each masked word's surface form and position;
    this adds its corpus lemma+POS and its neighbours' POS, then ranks and
    inflects replacements drawn from the target author's own vocabulary. Emits
    the §6 audit line per slot so a batch run can be scored on how often the
    pipeline was grounded rather than relaxed.
    """
    from candidate_selection import build_slots, log_slot

    slots = build_slots(stanza, MASK_WORDS, SOURCE_AUTHOR, TARGET_AUTHOR)
    for slot in slots:
        print(f'  [candidates] {log_slot(slot)}')
    return slots


# ── Probes ────────────────────────────────────────────────────────────────────

def build_probes(instr_lang: str = 'mk') -> list[dict]:
    """
    Build the battery. The Macedonian *text* is identical either way — only the
    instruction language changes, isolating "does this model follow Macedonian
    instructions worse than English ones?" from its actual Macedonian ability.
    """
    stanza = load_stanza()
    vocab = ', '.join(load_target_words())

    implicit = _mask(stanza, 'implicit')
    explicit = _mask(stanza, 'explicit')

    # The infill probes get a per-slot candidate structure instead of one global
    # vocab string; vanilla keeps the old string, since it has no slots to key off.
    from candidate_selection import render_slot_spec
    slots = _slots(stanza)

    if instr_lang == 'mk':
        style_spec = (
            f'Целниот стил е стилот на македонскиот поет {TARGET_AUTHOR}. '
            f'Карактеристични зборови за неговиот стил се: {vocab}.'
        )
        instr = {
            'fluency': ('Напиши четири стиха на македонски јазик за реката Вардар. '
                        'Одговори само со стиховите, без објаснување.'),
            'implicit': ('Зборовите означени со [загради] се стилски зборови што треба да се заменат. '
                         'Замени ГИ САМО тие зборови со зборови што одговараат на целниот стил. '
                         'Сите други зборови остави ги непроменети. Задржи ја истата структура на стихови. '
                         'Одговори само со новиот текст, без загради и без објаснување.'),
            'explicit': ('Замени го секој [MASK] со збор што одговара на целниот стил. '
                         'Сите други зборови остави ги непроменети. Задржи ја истата структура на стихови. '
                         'Одговори само со новиот текст, без објаснување.'),
            'vanilla': ('Пренапиши го текстот во целниот стил, задржувајќи ја содржината. '
                        'Одговори само со новиот текст, без објаснување.'),
            'morphology': (
                'Одговори на македонски јазик, само со бараните форми, една по ред, '
                'без објаснување.\n'
                '1. Стави го глаголот „оди" во правилна форма: „Јас ___ по патот."\n'
                '2. Стави ја придавката „тивок" во правилна форма: „___ ноќ."\n'
                '3. Стави ја именката „река" во определена форма, множина.\n'
                '4. Стави го глаголот „пее" во минато определено време, трето лице еднина.'),
            'header': 'Еве текст:',
        }
    else:
        style_spec = (
            f'The target style is that of the Macedonian poet {TARGET_AUTHOR}. '
            f'Words characteristic of that style: {vocab}.'
        )
        instr = {
            'fluency': ('Write four lines of verse in Macedonian about the river Vardar. '
                        'Reply with the verse only, no explanation.'),
            'implicit': ('The words marked in [brackets] are style words that must be replaced. '
                         'Replace ONLY those words with words fitting the target style. '
                         'Leave every other word exactly as it is. Keep the same line structure. '
                         'Reply with the new Macedonian text only — no brackets, no explanation.'),
            'explicit': ('Replace each [MASK] with a word fitting the target style. '
                         'Leave every other word exactly as it is. Keep the same line structure. '
                         'Reply with the new Macedonian text only, no explanation.'),
            'vanilla': ('Rewrite the text in the target style, preserving its content. '
                        'Reply with the new Macedonian text only, no explanation.'),
            'morphology': (
                'Answer in Macedonian with the requested forms only, one per line, '
                'no explanation.\n'
                '1. Put the verb "оди" in the correct form: "Јас ___ по патот."\n'
                '2. Put the adjective "тивок" in the correct form: "___ ноќ."\n'
                '3. Put the noun "река" in the definite plural form.\n'
                '4. Put the verb "пее" in the past definite tense, third person singular.'),
            'header': 'Here is a text:',
        }

    # Author sentence without the global vocab list — the slots supply the words now.
    author_spec = style_spec.split('. ')[0] + '.'
    implicit_spec = f'{author_spec}\n{render_slot_spec(slots, instr_lang, "implicit")}'
    explicit_spec = f'{author_spec}\n{render_slot_spec(slots, instr_lang, "explicit")}'

    return [
        {'name': '1. fluency', 'checks': ['script'], 'prompt': instr['fluency']},
        {'name': '2. infill_implicit', 'checks': ['script', 'infill'], 'src': implicit,
         'slots': slots,
         'prompt': f'{instr["header"]}\n{implicit}\n\n{implicit_spec}\n\n{instr["implicit"]}'},
        {'name': '3. infill_explicit', 'checks': ['script', 'infill'], 'src': explicit,
         'mask_token': '[MASK]', 'slots': slots,
         'prompt': f'{instr["header"]}\n{explicit}\n\n{explicit_spec}\n\n{instr["explicit"]}'},
        {'name': '4. vanilla', 'checks': ['script', 'infill'], 'src': stanza,
         'prompt': f'{instr["header"]}\n{stanza}\n\n{style_spec}\n\n{instr["vanilla"]}'},
        {'name': '5. morphology', 'checks': ['script'], 'prompt': instr['morphology']},
    ]


# ── Runner ────────────────────────────────────────────────────────────────────

def run(model: str, backend: str = 'ollama', instr_lang: str = 'mk') -> None:
    """Run the full battery against one model. 5 requests total."""
    probes = build_probes(instr_lang)
    delay = 3.5 if backend == 'openrouter' else 0.0  # only remote needs rate limiting
    print(f'\n{"=" * 72}\nMODEL: {model}  (backend: {backend}, '
          f'instructions: {instr_lang})\n{"=" * 72}')

    elapsed_total = 0.0
    for probe in probes:
        print(f'\n── {probe["name"]} {"─" * (60 - len(probe["name"]))}')
        print(f'\n==PROMPT==\n{probe["prompt"]}')
        t0 = time.time()
        try:
            output = call(model, probe['prompt'], backend=backend)
        except RuntimeError as e:
            print(f'  FAILED: {e}')
            continue
        elapsed = time.time() - t0
        elapsed_total += elapsed

        print(output)
        print(f'\n  [timing] {elapsed:.1f}s')

        if 'script' in probe['checks']:
            r = script_report(output)
            # Only Latin contamination is a real failure; the rest is context.
            verdict = 'OK' if r['cyr_ratio'] > 0.9 else 'LATIN CONTAMINATION'
            print(f'\n  [script {verdict}] cyr_ratio={r["cyr_ratio"]} '
                  f'latin={r["latin"]} mk_only={r["mk_only_found"]} '
                  f'neighbouring={r["foreign_found"]} (fyi, not a defect)')

        if 'infill' in probe['checks']:
            r = infill_report(probe['src'], output, probe.get('mask_token'))
            lines_ok = r['src_lines'] == r['out_lines']
            print(f'  [infill] lines={r["src_lines"]}->{r["out_lines"]}'
                  f'{"" if lines_ok else "  MISMATCH"} '
                  f'content_kept={r["content_kept"]} '
                  f'markers_left={r["markers_left"]} preamble={r["preamble"]}')

        time.sleep(delay)  # stay under 20 req/min on the remote backend

    print(f'\n{"=" * 72}')
    print(f'Done. 5 calls, {elapsed_total:.1f}s total, {elapsed_total / 5:.1f}s/call.')
    # A full PEGF poem costs ~15 calls; extrapolate the corpus-scale cost.
    print(f'At that rate: ~{elapsed_total / 5 * 15 / 60:.1f} min per poem at full PEGF '
          f'(~15 calls), ~{elapsed_total / 5 * 15 * 50 / 3600:.1f} h for a 50-poem eval.\n')
