"""Tagging and tag normalization, shared by the corpus build and inference.

pos_tag_corpus.py does not use raw classla output: it drops bare-Rg
relativizers and re-lemmatizes -јќи verbal adverbs, and pos_tagged.csv
reflects both. Inference that called classla directly would produce a source
'што' tagged ADV/Degree=Pos and a 'барајќи' lemmatized to itself — neither of
which exists in morph_lookup.json, sending every such slot to 'unresolved'.
Defining the normalization once makes that divergence impossible.
"""

from collections import Counter
from dataclasses import dataclass

KEEP_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

# classla tags што/кога/како/колку/каде as ADV, but they are relativizers, not
# manner adverbs ("Зборовите што ти ги дадов"). They carry the bare xpos 'Rg' —
# no degree slot — where real adverbs get Rgp/Rgc/Rgs, so xpos separates them
# cleanly. Left in, these five forms are the most frequent "content" words in
# the corpus and swamp any POS ratio built on it.
RELATIVIZER_XPOS = 'Rg'


def is_content(pos, xpos):
    return pos in KEEP_POS and xpos != RELATIVIZER_XPOS


def gerund_lemma(word, verb_lemmas):
    """Lemmatise a -јќи verbal adverb (xpos Rv) back to its base verb.

    classla tags these ADV and leaves them unlemmatised, so барајќи lemmatises
    to барајќи rather than бара. Stripping -јќи gives the stem, but the ending
    that goes back on is ambiguous: -ајќи always rebuilds an -а verb (барајќи ->
    бара), while -ејќи comes from both и-verbs (велејќи -> вели) and e-verbs
    (знаејќи -> знае). Rather than guess, weigh each candidate by how often
    classla lemmatised a finite verb to it elsewhere in the corpus.

    Frequency rather than first-match matters: бране and гале each occur once as
    a verb lemma, off a single mistagged token, and would beat the correct брани
    (20) and гали (15) on a bare membership test. и-verbs are the larger class,
    so they win when the corpus has nothing to say.
    """
    stem = word[:-len('јќи')]
    candidates = [stem]
    if stem.endswith('е'):
        candidates += [stem[:-1] + 'и', stem + 'е']
    attested = [c for c in candidates if verb_lemmas[c]]
    if attested:
        return max(attested, key=lambda c: verb_lemmas[c]), True
    return (stem[:-1] + 'и' if stem.endswith('е') else stem), False


def load_verb_lemmas(pos_rows):
    """VERB lemma frequencies, the evidence gerund_lemma() weighs candidates on.

    Built from pos_tagged.csv at inference time so a -јќи form resolves to the
    same base verb it did during the corpus build.
    """
    counts = Counter()
    for r in pos_rows:
        if r['pos'] == 'VERB' and r['lemma']:
            counts[r['lemma'].lower()] += 1
    return counts


@dataclass(frozen=True)
class Token:
    """One tagged token. Offsets are relative to the line it came from."""
    text: str
    lemma: str
    pos: str
    xpos: str
    feats: str
    line: int
    start_char: int
    end_char: int


_PIPELINE = None


def pipeline():
    """The classla mk pipeline, loaded once per process."""
    global _PIPELINE
    if _PIPELINE is None:
        import classla
        try:
            _PIPELINE = classla.Pipeline(
                'mk', processors='tokenize,pos,lemma', download_method=None)
        except Exception as e:
            raise SystemExit(
                f"Could not load the classla 'mk' models ({e}).\n"
                "Run this once to fetch them:  "
                "uv run python -c \"import classla; classla.download('mk')\"")
    return _PIPELINE


def tag_lines(text, verb_lemmas):
    """Tag text line by line, returning normalized content tokens.

    Offsets are line-relative because every consumer rebuilds output one line
    at a time. Normalization matches pos_tag_corpus.py exactly: relativizers
    dropped, -јќи forms re-lemmatized, words and lemmas lowercased.
    """
    nlp = pipeline()
    out = []
    for li, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        doc = nlp(line)
        for sent in doc.sentences:
            for w in sent.words:
                pos, xpos = w.upos or '', w.xpos or ''
                if not is_content(pos, xpos):
                    continue
                lemma = (w.lemma or '').lower()
                if xpos == 'Rv' and w.text.lower().endswith('јќи'):
                    lemma, _ = gerund_lemma(w.text.lower(), verb_lemmas)
                start, end = _span(w, line)
                out.append(Token(
                    text=w.text.lower(), lemma=lemma, pos=pos, xpos=xpos,
                    feats=w.feats or '', line=li,
                    start_char=start, end_char=end,
                ))
    return out


def _span(word, line):
    """Character span of a classla word within its line.

    classla exposes offsets as a 'start_char|end_char' string in word.misc on
    most builds; fall back to locating the surface form when it is absent.
    """
    misc = getattr(word, 'misc', None) or ''
    start = end = None
    for part in misc.split('|'):
        if part.startswith('start_char='):
            start = int(part.split('=', 1)[1])
        elif part.startswith('end_char='):
            end = int(part.split('=', 1)[1])
    if start is None or end is None:
        start = line.find(word.text)
        end = start + len(word.text) if start >= 0 else 0
        start = max(start, 0)
    return start, end
