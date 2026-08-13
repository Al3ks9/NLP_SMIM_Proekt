"""POS-tag the corpus with classla, and audit tags that look wrong.

Writes two files:
  data/pos_tagged.csv  content words only (KEEP_POS), one row per token
  data/pos_flags.csv   every token that tripped a suspicion rule, with context

classla gives us three things spaCy's mk_core_news_lg did not: MULTEXT-East
xpos tags, morphological feats, and lemmas that actually strip inflection
(проклета -> проклет). xpos and feats are appended to pos_tagged.csv; the
original five columns keep their names and meaning so downstream scripts that
read this file with csv.DictReader are unaffected.
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import corrections as C
from tagging import KEEP_POS, RELATIVIZER_XPOS, is_content, gerund_lemma

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

MIN_POEMS = 5

# The first xpos letter each upos should agree with. classla runs the two
# taggers separately, so a clash means neither reading is trustworthy.
XPOS_CATEGORY = {'NOUN': 'N', 'PROPN': 'N', 'VERB': 'V', 'ADJ': 'A', 'ADV': 'R'}

CYRILLIC = re.compile(r'[Ѐ-ӿ]')
LATIN_GREEK = re.compile(r'[A-Za-zͰ-Ͽ]')

# unstable_tag thresholds. A form must occur at least MIN_OCC times before its
# tag distribution says anything, and a tag covering more than MAX_SHARE of
# those occurrences is too common to be a slip. Loosening these (3 / 0.34)
# roughly triples the flags and lets in more genuine ambiguity.
MIN_OCC = 4
MAX_SHARE = 0.20


def flag_reasons(word, lemma, pos, xpos, tag_dist):
    """Rules that catch tags the tagger probably got wrong.

    Returns (reason, detail) pairs. Two families:

    Hard errors — tokens the tagger openly failed on, or that carry characters
    it cannot have handled correctly.

    Suspicions — tags contradicted by other evidence. `unstable_tag` is the
    productive one: it compares a token's tag against every other tagging of
    the same form in the corpus, so a form tagged ADJ eight times and NOUN once
    surfaces that once. It flags the minority reading, which is occasionally
    the correct one and the majority wrong — either way the form needs a look.

    `tag_dist` is the corpus-wide Counter of tags for this lowercase form.
    """
    reasons = []

    # -- hard errors --
    if not pos or pos == 'X' or xpos == 'X':
        reasons.append(('unknown_tag', xpos or 'no tag'))
    if CYRILLIC.search(word) and LATIN_GREEK.search(word):
        # A Cyrillic word carrying a Latin/Greek lookalike codepoint — the
        # tagger sees an OOV string, not the word a reader sees.
        reasons.append(('mixed_script', ''))
    elif pos in KEEP_POS and LATIN_GREEK.search(word) and not CYRILLIC.search(word):
        reasons.append(('non_cyrillic', ''))
    if not lemma.strip():
        reasons.append(('empty_lemma', ''))
    elif CYRILLIC.search(word) and not CYRILLIC.search(lemma):
        reasons.append(('bad_lemma', f'lemma {lemma!r}'))

    # -- suspicions --
    if (pos in XPOS_CATEGORY and xpos and xpos != 'X'
            and xpos[0] != XPOS_CATEGORY[pos]):
        reasons.append(('upos_xpos_clash', f'upos {pos} vs xpos {xpos}'))
    if pos == 'PROPN' and word[:1].islower():
        reasons.append(('lowercase_propn', ''))
    total = sum(tag_dist.values())
    if pos in KEEP_POS and total >= MIN_OCC and tag_dist[pos] / total <= MAX_SHARE:
        majority, count = tag_dist.most_common(1)[0]
        reasons.append(('unstable_tag', f'{tag_dist[pos]}/{total} {pos}, majority {majority} ({count})'))

    return reasons


with open(DATA / 'stripped_songs.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

author_counts = Counter(r['author'] for r in rows)
qualified = {a for a, c in author_counts.items() if c >= MIN_POEMS}
poems = [r for r in rows if r['author'] in qualified]
print(f"Tagging {len(qualified)} authors ({len(poems)} poems) with classla")

from tagging import pipeline
nlp = pipeline()

# Tag everything first, including function words and punctuation. The audit
# needs the tokens KEEP_POS throws away — unknown_tag fires almost entirely on
# tokens that never reach pos_tagged.csv — and unstable_tag needs each form's
# corpus-wide tag distribution, which is only known once every poem is tagged.
tokens = []       # (author, title, word, lemma, pos, xpos, feats, sentence_id)
sentences = []    # sentence text, indexed by sentence_id

for i, row in enumerate(poems):
    doc = nlp(row['song_text'])
    for sent in doc.sentences:
        sentences.append(' '.join(t.text for t in sent.tokens))
        sid = len(sentences) - 1
        for w in sent.words:
            tokens.append((
                row['author'], row['song_title'], w.text, w.lemma or '',
                w.upos or '', w.xpos or '', w.feats or '', sid,
            ))
    if (i + 1) % 250 == 0:
        print(f"  processed {i + 1}/{len(poems)} poems...")

print(f"Tagged {len(tokens)} tokens across {len(sentences)} sentences")

# Hand corrections go on before anything reads the tags, so tag distributions,
# the flag rules and the output file all see the same corrected corpus — and a
# token you have already fixed stops being flagged.
table = C.load()
if table:
    corrected, changed, removed = [], 0, 0
    for author, title, word, lemma, pos, xpos, feats, sid in tokens:
        result = C.apply_to(table, word, pos, lemma, xpos, feats,
                            author, title, sentences[sid])
        if result is None:
            removed += 1
            continue
        if (result[0], result[1]) != (pos, lemma):
            changed += 1
        pos, lemma, xpos, feats = result
        corrected.append((author, title, word, lemma, pos, xpos, feats, sid))
    print(f"Applied {len(table)} hand corrections from {C.CORRECTIONS_CSV.name}: "
          f"{changed} tokens retagged, {removed} dropped")
    tokens = corrected

# Both of these need the whole corpus in hand: unstable_tag compares a token
# against every other tagging of the same form, and gerund_lemma resolves -јќи
# stems against the verb lemmas classla produced anywhere in the corpus.
tag_dists = defaultdict(Counter)   # lowercase form -> Counter of content tags
verb_lemmas = Counter()            # verb lemma -> how often classla produced it
for _, _, word, lemma, pos, xpos, _, _ in tokens:
    if is_content(pos, xpos):
        tag_dists[word.lower()][pos] += 1
    if pos == 'VERB' and lemma:
        verb_lemmas[lemma.lower()] += 1

results = []
flags = []
dropped_relativizers = 0
gerunds_relemmatised = 0
gerunds_guessed = 0
for author, title, word, lemma, pos, xpos, feats, sid in tokens:
    if pos in KEEP_POS and xpos == RELATIVIZER_XPOS:
        dropped_relativizers += 1
    if is_content(pos, xpos):
        out_lemma = lemma.lower()
        if xpos == 'Rv' and word.lower().endswith('јќи'):
            out_lemma, known = gerund_lemma(word.lower(), verb_lemmas)
            gerunds_relemmatised += known
            gerunds_guessed += not known
        results.append({
            'author': author,
            'song_title': title,
            'word': word.lower(),
            'lemma': out_lemma,
            'pos': pos,
            'xpos': xpos,
            'feats': feats,
        })
    if pos == 'PUNCT' or not word.strip() or word.isnumeric():
        continue
    for reason, detail in flag_reasons(word, lemma, pos, xpos, tag_dists[word.lower()]):
        flags.append({
            'reason': reason,
            'word': word,
            'lemma': lemma,
            'pos': pos,
            # Left blank for you to fill in; see src/corrections.py.
            'correction': '',
            'lemma_fix': '',
            'scope': '',
            'xpos': xpos,
            'detail': detail,
            'author': author,
            'song_title': title,
            'context': sentences[sid],
        })

with open(DATA / 'pos_tagged.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f, fieldnames=['author', 'song_title', 'word', 'lemma', 'pos', 'xpos', 'feats'])
    writer.writeheader()
    writer.writerows(results)

# Grouped by rule so the file can be eyeballed one failure mode at a time.
flags.sort(key=lambda r: (r['reason'], r['word'].lower()))
with open(DATA / 'pos_flags.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f, fieldnames=['reason', 'word', 'lemma', 'pos',
                       'correction', 'lemma_fix', 'scope',
                       'xpos', 'detail', 'author', 'song_title', 'context'])
    writer.writeheader()
    writer.writerows(flags)

print(f"\nDone. {len(results)} content tokens -> pos_tagged.csv")
print("POS distribution:", dict(Counter(r['pos'] for r in results).most_common()))
print(f"Dropped {dropped_relativizers} relativizers (xpos {RELATIVIZER_XPOS}: што, кога, како, колку, каде)")
print(f"Re-lemmatised {gerunds_relemmatised + gerunds_guessed} -јќи verbal adverbs "
      f"({gerunds_relemmatised} matched a corpus verb lemma, {gerunds_guessed} fell back to the -и rule)")

print(f"\n{len(flags)} flags ({len(flags) / len(tokens):.2%} of all tokens) -> pos_flags.csv")
by_reason = defaultdict(list)
for r in flags:
    by_reason[r['reason']].append(r['word'].lower())
for reason, words in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
    top = ', '.join(w for w, _ in Counter(words).most_common(8))
    print(f"  {reason:20} {len(words):>6}  ({len(set(words))} unique)  e.g. {top}")
