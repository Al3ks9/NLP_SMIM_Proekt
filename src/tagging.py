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

import corrections

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
    at a time. Normalization matches pos_tag_corpus.py exactly: hand
    corrections applied, relativizers dropped, -јќи forms re-lemmatized,
    words and lemmas lowercased.
    """
    nlp = pipeline()
    table = corrections.load()
    out = []
    for li, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        doc = nlp(line)
        cursor = 0
        for sent in doc.sentences:
            for w in sent.words:
                span = _span(w, line, cursor)
                if span is None:
                    # Surface form not found anywhere in the line -- skip the
                    # token rather than fabricate an offset. See _span's
                    # docstring for why (0, 0) would be silent corruption.
                    continue
                start, end = span
                cursor = max(cursor, end)

                pos, xpos = w.upos or '', w.xpos or ''
                lemma, feats = w.lemma or '', w.feats or ''
                # corrections.lookup() tries row scope, then poem scope, then
                # the bare (word.lower(), pos) global key. Inference has no
                # corpus poem_id/context, so passing poem_id='0' simply fails to
                # match the narrow scopes and falls through to the global key --
                # which is correct, since poem/row-scoped fixes are corpus-position
                # -specific and must not fire on arbitrary input. All 86
                # corrections in pos_corrections.csv are global scope, so all
                # 86 apply here, same as in pos_tag_corpus.py.
                result = corrections.apply_to(
                    table, w.text, pos, lemma, xpos, feats, '0', '')
                if result is None:
                    continue
                pos, lemma, xpos, feats = result

                if not is_content(pos, xpos):
                    continue
                lemma = lemma.lower()
                # A corrected POS clears xpos (see apply_to), so a corrected
                # token never carries xpos == 'Rv' here and correctly skips
                # gerund re-lemmatization -- same as the corpus build.
                if xpos == 'Rv' and w.text.lower().endswith('јќи'):
                    lemma, _ = gerund_lemma(w.text.lower(), verb_lemmas)
                out.append(Token(
                    text=w.text.lower(), lemma=lemma, pos=pos, xpos=xpos,
                    feats=feats, line=li,
                    start_char=start, end_char=end,
                ))
    return out


def _span(word, line, cursor):
    """Character span of a classla word within its line, or None if it cannot
    be located.

    Verified live against classla's 'mk' models: word.misc never carries
    start_char/end_char there, only SpaceAfter=No, so the misc-parsing branch
    below is dead for this project -- it exists only as a guard for other
    classla builds that do emit real offsets. The line.find() scan is the
    primary, and in practice the only, path 'mk' takes. `cursor` is how far
    into the line previous tokens' spans already reached, so the scan resumes
    after the last match instead of always returning the first occurrence --
    repeated words are common in this poetry corpus, and a later splice keyed
    on these offsets would otherwise send every repeat to the same position.

    Returns None, never a fabricated (0, 0), when the surface form cannot be
    found anywhere in the line. Callers must skip such a token rather than
    splice it in at column 0 -- that would silently insert a replacement at
    the start of the line instead of leaving it untouched.
    """
    misc = getattr(word, 'misc', None) or ''
    start = end = None
    for part in misc.split('|'):
        if part.startswith('start_char='):
            start = int(part.split('=', 1)[1])
        elif part.startswith('end_char='):
            end = int(part.split('=', 1)[1])
    if start is not None and end is not None:
        return start, end

    start = line.find(word.text, cursor)
    if start < 0:
        start = line.find(word.text)
    if start < 0:
        return None
    return start, start + len(word.text)
