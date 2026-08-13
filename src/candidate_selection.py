"""
Per-slot candidate selection for style transfer.

Replaces the global "top-20 TF-IDF words" style spec with a per-slot,
grammatically valid, thematically grounded, corpus-attested candidate list for
each masked word in the infill probes. This is inference-time logic: it runs at
prompt-build time, not as a one-time offline build like the SKG scripts.

The SKG (skg*.gexf) is deliberately bypassed here. Its 371-word cap and
frequency floors exist for visualisation legibility, not because the underlying
data is missing elsewhere — every signal the graph encodes is read straight from
the ungated JSON/CSV sources instead:

    data/pos_tagged.csv        live co-occurrence, POS + lemma lookups
    data/tfidf_results.csv     distinctiveness boost
    models/author_vocab.json   candidate pool (uncapped)
    models/pos_transitions.json syntactic plausibility
    models/morph_lookup.json   surface realisation
    models/word_embeddings.json semantic narrowing

Every candidate carries the tier that produced it ('primary', 'relaxed_semantic',
'cross_author_similar', 'legacy_fallback') and the morph tier that realised its
surface form ('exact', 'cross_author', 'nearest_attested', 'unresolved'), so a
batch run can be audited for how often the pipeline is actually grounded.
"""

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'

log = logging.getLogger('candidate_selection')

# --- Tunables (§3 step 4, §4). Raw components are logged so these can be retuned
# against evidence rather than intuition. ---
SEM_THRESHOLD = 0.65     # same floor add_embeddings.py used for semantic edges
SEM_FLOOR = 0.45         # lowest threshold the relaxation chain will drop to
SEM_RELAX_STEP = 0.05
MIN_CANDIDATES = 3
TOP_K = 3

W_TFIDF, W_SEM, W_COOC, W_TRANS = 0.4, 0.35, 0.15, 0.10

SUFFIX_LEN = 3           # matches build_morph_lookup.py
JACCARD_THRESHOLD = 0.15  # matches enrich_skg.py's author_similarity edges
MAX_SIMILAR_AUTHORS = 3

# Only these POS tags exist in pos_tagged.csv (pos_tag_corpus.py's KEEP_POS), so
# "the preceding token" always means the preceding *content* token — which is
# also what pos_transitions.json was counted over.
KEEP_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}


# ── Loaders (load once, reuse across every slot in a run) ─────────────────────

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def load_pos_rows() -> list:
    """pos_tagged.csv as a list of dicts, in corpus order (order matters — the
    per-poem token stream is used to read off neighbouring POS tags)."""
    def build():
        with open(DATA / 'pos_tagged.csv', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    return _cached('pos_rows', build)


def load_author_vocab() -> dict:
    """author -> pos -> [(lemma, freq), ...], frequency-sorted."""
    def build():
        with open(MODELS / 'author_vocab.json', encoding='utf-8') as f:
            raw = json.load(f)
        return {
            author: {pos: [(lemma, freq) for lemma, freq in pairs]
                     for pos, pairs in pos_map.items()}
            for author, pos_map in raw.items()
        }
    return _cached('author_vocab', build)


def load_tfidf() -> dict:
    """author -> {word: score}. Note these are *surface* words, top 20 per author."""
    def build():
        tfidf = defaultdict(dict)
        with open(DATA / 'tfidf_results.csv', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                tfidf[row['author']][row['word']] = float(row['tfidf_score'])
        return dict(tfidf)
    return _cached('tfidf', build)


def load_pos_transitions() -> dict:
    """author -> pos -> {next_pos: probability}."""
    def build():
        with open(MODELS / 'pos_transitions.json', encoding='utf-8') as f:
            return json.load(f)
    return _cached('pos_transitions', build)


def load_morph_lookup() -> dict:
    """author -> lemma -> pos -> suffix -> surface form."""
    def build():
        with open(MODELS / 'morph_lookup.json', encoding='utf-8') as f:
            return json.load(f)
    return _cached('morph_lookup', build)


def load_word_embeddings() -> dict:
    """lemma -> 768-dim normalized vector, for every lemma in the corpus."""
    def build():
        with open(MODELS / 'word_embeddings.json', encoding='utf-8') as f:
            raw = json.load(f)
        return {lemma: np.asarray(vec, dtype=np.float32) for lemma, vec in raw.items()}
    return _cached('word_embeddings', build)


def build_cooccurrence_index(pos_rows) -> dict:
    """
    (lemma, pos) -> Counter{(other_lemma, other_pos): shared_poem_count}

    Same logic as build_skg.py's `cooccurrence`, but symmetric and keyed for
    lookup by ANY lemma rather than filtered to graph-node membership. Built
    once per process and reused across all slot queries.

    ~1s and ~300 MB at full corpus scale (1,169 poems, 4.3M pairs).
    """
    poem_tokens = defaultdict(set)
    for r in pos_rows:
        poem_tokens[(r['author'], r['song_title'])].add((r['lemma'], r['pos']))

    index = defaultdict(Counter)
    for tokens in poem_tokens.values():
        tokens = sorted(tokens)
        for i in range(len(tokens)):
            left = index[tokens[i]]
            for j in range(i + 1, len(tokens)):
                left[tokens[j]] += 1
                index[tokens[j]][tokens[i]] += 1
    return index


def load_cooccurrence_index() -> dict:
    return _cached('cooc_index', lambda: build_cooccurrence_index(load_pos_rows()))


def load_resources() -> dict:
    """Everything build_slot() needs, loaded once."""
    return {
        'author_vocab': load_author_vocab(),
        'tfidf': load_tfidf(),
        'transitions': load_pos_transitions(),
        'morph_lookup': load_morph_lookup(),
        'embeddings': load_word_embeddings(),
        'cooc_index': load_cooccurrence_index(),
    }


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embedding(lemma: str, embeddings: dict):
    """
    Direct dict lookup. word_embeddings.json covers every corpus lemma, so this
    should always hit for in-corpus source words. Returns None (not an exception)
    if truly absent — e.g. a lemma introduced only via a hardcoded stanza — and
    the caller skips the semantic filter for that slot rather than crashing.
    """
    return embeddings.get(lemma)


def cosine(a, b) -> float:
    """Plain dot product — embeddings are pre-normalized at generation time."""
    return float(a @ b)


# ── Candidate pool ────────────────────────────────────────────────────────────

def candidate_pool(target_author: str, pos: str, author_vocab: dict) -> list:
    """
    author_vocab[target_author][pos], already frequency-sorted. Returns [] if the
    author/pos combo is absent — the caller handles that via the fallback tiers.
    """
    return list(author_vocab.get(target_author, {}).get(pos, []))


def author_similarity(target_author: str, pos_rows) -> list:
    """
    [(author, jaccard), ...] descending, recomputed directly from pos_tagged.csv
    lemma sets — the same measure enrich_skg.py writes as author_similarity edges,
    but without depending on the graph.
    """
    def build():
        vocab = defaultdict(set)
        for r in pos_rows:
            vocab[r['author']].add(r['lemma'])
        return vocab

    vocab = _cached('author_lemma_sets', build)
    mine = vocab.get(target_author)
    if not mine:
        return []

    scored = []
    for author, theirs in vocab.items():
        if author == target_author:
            continue
        union = mine | theirs
        if union:
            scored.append((author, len(mine & theirs) / len(union)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


# ── Scoring ───────────────────────────────────────────────────────────────────

def _transition_score(target_author, source_pos, prev_pos, next_pos, transitions) -> float:
    """
    Average of the incoming and outgoing POS-bigram probabilities for this slot,
    skipping whichever side is None (line-initial / line-final).

    Candidate-independent by construction — replacements keep the source POS — so
    this scores how at-home the slot's POS is in the target author's syntax, and
    shifts a slot's scores as a block rather than reordering within it.
    """
    author_t = transitions.get(target_author, {})
    parts = []
    if prev_pos is not None:
        parts.append(author_t.get(prev_pos, {}).get(source_pos, 0.0))
    if next_pos is not None:
        parts.append(author_t.get(source_pos, {}).get(next_pos, 0.0))
    return sum(parts) / len(parts) if parts else 0.0


def score_pool(
    pool, source_lemma: str, source_pos: str, target_author: str,
    prev_pos, next_pos, tfidf: dict, embeddings: dict,
    cooc_index: dict, transitions: dict,
) -> list:
    """
    Per-candidate components plus the weighted composite for a pool, before any
    filtering. Split out from rank_candidates so the filter/relaxation tiers can
    be tested against a fixed set of scored candidates.
    """
    source_vec = get_embedding(source_lemma, embeddings)
    author_tfidf = tfidf.get(target_author, {})
    neighbours = cooc_index.get((source_lemma, source_pos), {})
    transition = _transition_score(
        target_author, source_pos, prev_pos, next_pos, transitions
    )

    scored = []
    for lemma, freq in pool:
        cand_vec = get_embedding(lemma, embeddings)
        sem_sim = (cosine(source_vec, cand_vec)
                   if source_vec is not None and cand_vec is not None else None)
        scored.append({
            'lemma': lemma,
            'freq': freq,
            'tfidf': author_tfidf.get(lemma, 0.0),
            'sem_sim': sem_sim,
            'cooc': neighbours.get((lemma, source_pos), 0),
            'transition': transition,
        })
    _composite(scored)
    return scored


def _semantic_filter(scored, threshold: float) -> list:
    """Drop candidates whose similarity to the source word is known and too low.
    Candidates with sem_sim None (no vector) pass — an unknown is not a failure."""
    return [c for c in scored
            if c['sem_sim'] is None or c['sem_sim'] >= threshold]


def _composite(candidates) -> None:
    """
    Attach the weighted composite score, normalising the unbounded components
    (tfidf, cooc) against the maximum within this slot's own pool.
    """
    max_tfidf = max((c['tfidf'] for c in candidates), default=0.0) or 1.0
    max_cooc = max((c['cooc'] for c in candidates), default=0) or 1

    for c in candidates:
        c['score'] = round(
            W_TFIDF * (c['tfidf'] / max_tfidf)
            + W_SEM * (c['sem_sim'] or 0.0)
            + W_COOC * (c['cooc'] / max_cooc)
            + W_TRANS * c['transition'],
            4,
        )


def _emit(candidates, tier: str) -> list:
    """Tag a scored batch with the tier that produced it and order it."""
    out = []
    for c in candidates:
        entry = {k: c[k] for k in ('lemma', 'score', 'tfidf', 'sem_sim', 'cooc', 'transition')}
        entry['sem_sim'] = None if c['sem_sim'] is None else round(c['sem_sim'], 4)
        entry['tfidf'] = round(c['tfidf'], 4)
        entry['transition'] = round(c['transition'], 4)
        entry['tier'] = tier
        for optional in ('sem_threshold', 'donor_author', 'donor_jaccard'):
            if optional in c:
                entry[optional] = c[optional]
        out.append(entry)
    out.sort(key=lambda c: (-c['score'], c['lemma']))
    return out


# ── Fallback / relaxation tiers (§4) — each callable and loggable on its own ───

def tier_primary(
    source_lemma, source_pos, target_author, prev_pos, next_pos,
    author_vocab, tfidf, embeddings, cooc_index, transitions,
) -> tuple:
    """
    Tier 0: the target author's own pool for this POS at the default threshold.
    Returns (scored_pool, kept) — the full pool comes back too, so the relaxation
    tier can re-filter it without re-scoring.
    """
    pool = [(lemma, freq)
            for lemma, freq in candidate_pool(target_author, source_pos, author_vocab)
            if lemma != source_lemma]
    scored = score_pool(pool, source_lemma, source_pos, target_author,
                        prev_pos, next_pos, tfidf, embeddings, cooc_index, transitions)
    kept = _semantic_filter(scored, SEM_THRESHOLD)
    log.info('tier=primary author=%s pos=%s pool=%d kept=%d threshold=%.2f',
             target_author, source_pos, len(pool), len(kept), SEM_THRESHOLD)
    return scored, kept


def tier_relaxed_semantic(scored, already, needed: int) -> list:
    """
    Tier 1: step SEM_THRESHOLD down to SEM_FLOOR, topping up the shortfall with
    whatever the looser threshold admits. Returns only the newly admitted ones.
    """
    added = []
    threshold = SEM_THRESHOLD
    while threshold > SEM_FLOOR and len(added) < needed:
        threshold = round(threshold - SEM_RELAX_STEP, 2)
        admitted = [c for c in _semantic_filter(scored, threshold)
                    if c['lemma'] not in already]
        added = admitted
        log.info('tier=relaxed_semantic threshold=%.2f added=%d', threshold, len(added))
    for c in added:
        c['sem_threshold'] = threshold
    return added


def tier_cross_author_similar(
    source_lemma, source_pos, target_author, prev_pos, next_pos,
    author_vocab, tfidf, embeddings, cooc_index, transitions,
) -> list:
    """
    Tier 2: pull from the most Jaccard-similar authors' pools for the same POS,
    ranked by *their* own tfidf/embedding scores against the same source word.
    """
    similar = [(a, j) for a, j in author_similarity(target_author, load_pos_rows())
               if j >= JACCARD_THRESHOLD][:MAX_SIMILAR_AUTHORS]
    if not similar:
        log.info('tier=cross_author_similar author=%s no similar authors above %.2f',
                 target_author, JACCARD_THRESHOLD)
        return []

    scored = []
    for author, jaccard in similar:
        pool = [(lemma, freq)
                for lemma, freq in candidate_pool(author, source_pos, author_vocab)
                if lemma != source_lemma]
        batch = score_pool(pool, source_lemma, source_pos, author,
                           prev_pos, next_pos, tfidf, embeddings, cooc_index, transitions)
        for c in batch:
            c['donor_author'] = author
            c['donor_jaccard'] = round(jaccard, 4)
        scored.extend(_semantic_filter(batch, SEM_THRESHOLD))

    log.info('tier=cross_author_similar donors=%s kept=%d',
             [a for a, _ in similar], len(scored))
    return scored


def tier_legacy_fallback(source_lemma, source_pos, target_author, prev_pos,
                         next_pos, tfidf, embeddings, cooc_index, transitions) -> list:
    """
    Tier 3, last resort: the global TF-IDF top-N that the probe harness used
    before this pipeline existed. Deliberately noisy in the log — reaching this
    tier means nothing in the corpus was grounded enough to answer the slot.
    """
    # Imported here, not at module scope: llm_probe imports this module, so a
    # top-level import would be circular.
    from llm_probe import TARGET_AUTHOR, load_target_words

    words = load_target_words()
    if target_author != TARGET_AUTHOR:
        log.warning('tier=legacy_fallback load_target_words() is hardcoded to %r, '
                    'but this slot targets %r — words are not the requested author\'s',
                    TARGET_AUTHOR, target_author)

    pool = [(w, 0) for w in words if w != source_lemma]
    scored = score_pool(pool, source_lemma, source_pos, target_author,
                        prev_pos, next_pos, tfidf, embeddings, cooc_index, transitions)
    log.warning('tier=legacy_fallback source=%s pos=%s candidates=%d '
                '(no grounded candidates found — this should be rare)',
                source_lemma, source_pos, len(scored))
    return scored


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_candidates(
    source_lemma: str, source_pos: str,
    target_author: str,
    prev_pos, next_pos,
    author_vocab: dict, tfidf: dict, embeddings: dict,
    cooc_index: dict, transitions: dict,
    top_k: int = TOP_K,
) -> list:
    """
    Rank replacements for one masked word, walking the §4 relaxation chain only
    as far as it has to.

    Returns up to top_k dicts, score-descending:
      {'lemma', 'score', 'tfidf', 'sem_sim', 'cooc', 'transition', 'tier'}
    where tier is 'primary' | 'relaxed_semantic' | 'cross_author_similar' |
    'legacy_fallback'.
    """
    scored, kept = tier_primary(
        source_lemma, source_pos, target_author, prev_pos, next_pos,
        author_vocab, tfidf, embeddings, cooc_index, transitions,
    )
    results = _emit(kept, 'primary')

    if len(results) < MIN_CANDIDATES:
        already = {c['lemma'] for c in results}
        added = tier_relaxed_semantic(scored, already, MIN_CANDIDATES - len(results))
        results += _emit(added, 'relaxed_semantic')

    if len(results) < MIN_CANDIDATES:
        already = {c['lemma'] for c in results}
        donors = [c for c in tier_cross_author_similar(
            source_lemma, source_pos, target_author, prev_pos, next_pos,
            author_vocab, tfidf, embeddings, cooc_index, transitions,
        ) if c['lemma'] not in already]
        results += _emit(donors, 'cross_author_similar')

    if not results:
        results = _emit(tier_legacy_fallback(
            source_lemma, source_pos, target_author, prev_pos, next_pos,
            tfidf, embeddings, cooc_index, transitions,
        ), 'legacy_fallback')

    # Composite scores are normalised per emitted batch; re-rank the merged list
    # so a relaxed or donor candidate cannot outrank a primary one on tier alone.
    tier_rank = {'primary': 0, 'relaxed_semantic': 1,
                 'cross_author_similar': 2, 'legacy_fallback': 3}
    results.sort(key=lambda c: (tier_rank[c['tier']], -c['score'], c['lemma']))
    return results[:top_k]


# ── Surface realisation ───────────────────────────────────────────────────────

def _shared_suffix_len(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def surface_form(
    target_author: str, candidate_lemma: str, pos: str, source_word: str,
    morph_lookup: dict,
) -> tuple:
    """
    Realise a candidate lemma in the source word's grammatical form.

    Returns (surface_form, tier_used) — the tier is required output, not optional
    logging, since it says how much of the inflection the model still has to do.

      1. morph_lookup[target][lemma][pos][suffix]                  -> 'exact'
      2. first other author whose [lemma][pos] has that suffix     -> 'cross_author'
      3. best attested form for [target][lemma][pos], any suffix   -> 'nearest_attested'
      4. the lemma unchanged, flagged for model-side inflection    -> 'unresolved'

    morph_lookup.json collapsed its per-suffix counters to one form each, so tier
    3's "most frequent" is approximated by how many suffixes a form is attested
    under, tie-broken by longest shared suffix with the source word.
    """
    suffix = (source_word[-SUFFIX_LEN:].lower() if len(source_word) >= SUFFIX_LEN
              else source_word.lower())

    target_forms = morph_lookup.get(target_author, {}).get(candidate_lemma, {}).get(pos, {})
    if suffix in target_forms:
        return target_forms[suffix], 'exact'

    for author, lemmas in morph_lookup.items():
        if author == target_author:
            continue
        forms = lemmas.get(candidate_lemma, {}).get(pos, {})
        if suffix in forms:
            return forms[suffix], 'cross_author'

    if target_forms:
        counts = Counter(target_forms.values())
        forms = sorted(set(target_forms.values()))
        best = max(forms, key=lambda f: (counts[f], _shared_suffix_len(f, suffix)))
        return best, 'nearest_attested'

    return candidate_lemma, 'unresolved'


# ── Slot assembly ─────────────────────────────────────────────────────────────

def build_slot(
    source_word: str, source_lemma: str, source_pos: str,
    prev_pos, next_pos,
    target_author: str, resources: dict,
) -> dict:
    """
    One masked word in, one ranked + realised candidate list out.

    {'source_word': 'незнаен', 'pos': 'ADJ', 'candidates': [
        {'surface': 'самотен', 'lemma': 'самотен', 'score': 0.81,
         'morph_tier': 'exact', 'tier': 'primary', ...}, ...]}
    """
    ranked = rank_candidates(
        source_lemma, source_pos, target_author, prev_pos, next_pos,
        resources['author_vocab'], resources['tfidf'], resources['embeddings'],
        resources['cooc_index'], resources['transitions'],
    )

    candidates = []
    for c in ranked:
        surface, morph_tier = surface_form(
            target_author, c['lemma'], source_pos, source_word,
            resources['morph_lookup'],
        )
        candidates.append({'surface': surface, 'morph_tier': morph_tier, **c})

    return {
        'source_word': source_word,
        'lemma': source_lemma,
        'pos': source_pos,
        'prev_pos': prev_pos,
        'next_pos': next_pos,
        'candidates': candidates,
    }


def log_slot(slot: dict) -> str:
    """
    §6's per-slot audit line: source_word, pos, chosen_candidate, composite_score,
    morph_tier, semantic_tier. Returned as well as logged so callers can print it.
    """
    top = slot['candidates'][0] if slot['candidates'] else None
    line = ('slot source_word={} pos={} chosen={} score={} morph_tier={} semantic_tier={}'
            .format(slot['source_word'], slot['pos'],
                    top['surface'] if top else '-',
                    top['score'] if top else '-',
                    top['morph_tier'] if top else '-',
                    top['tier'] if top else 'none'))
    log.info(line)
    return line


# ── Source-side context lookup (§5) ───────────────────────────────────────────

def _words(text: str) -> list:
    return re.findall(r'\w+', text.lower())


def corpus_token_stream(author: str, song_title: str, pos_rows) -> list:
    """The ordered content-token stream of one poem, as tagged in pos_tagged.csv."""
    return [r for r in pos_rows
            if r['author'] == author and r['song_title'] == song_title]


def _align(text: str, tokens: list) -> dict:
    """
    Greedily align a stanza's words to a poem's content-token stream.

    pos_tagged.csv keeps only content words, so the stanza's function words
    simply find no match and are skipped. Returns {(line_idx, word_idx): token_idx}.
    """
    mapping = {}
    p = 0
    for li, line in enumerate(text.splitlines()):
        for wi, word in enumerate(_words(line)):
            for q in range(p, len(tokens)):
                if tokens[q]['word'] == word:
                    mapping[(li, wi)] = q
                    p = q + 1
                    break
    return mapping


def _retag(text: str) -> list:
    """spaCy fallback for a source poem that is not in the corpus (§5)."""
    import spacy
    nlp = _cached('spacy', lambda: spacy.load('mk_core_news_lg'))
    out = []
    for li, line in enumerate(text.splitlines()):
        for token in nlp(line):
            if token.pos_ in KEEP_POS and not token.is_space and not token.is_punct:
                out.append({'line': li, 'word': token.text.lower(),
                            'lemma': token.lemma_.lower(), 'pos': token.pos_})
    return out


def find_source_poem(text: str, author: str, pos_rows) -> tuple:
    """
    Locate the poem a stanza came from, by picking the poem of that author whose
    token stream the stanza aligns into most completely.

    Returns (song_title, coverage) or (None, 0.0).
    """
    wanted = [w for line in text.splitlines() for w in _words(line)]
    if not wanted:
        return None, 0.0

    titles = {r['song_title'] for r in pos_rows if r['author'] == author}
    best, best_cov = None, 0.0
    for title in sorted(titles):
        tokens = corpus_token_stream(author, title, pos_rows)
        cov = len(_align(text, tokens)) / len(wanted)
        if cov > best_cov:
            best, best_cov = title, cov
    return best, best_cov


def slot_contexts(text: str, mask_words: list, source_author: str,
                  pos_rows=None, min_coverage: float = 0.5) -> list:
    """
    For each masked word, the lemma+POS the corpus assigns it and the POS of the
    nearest tagged token on either side *within its own line* (None at the line
    edges — a line-initial word has no incoming transition to score).

    Reads the tagging from pos_tagged.csv when the source poem is in the corpus,
    and re-tags with spaCy when it is not.
    """
    pos_rows = load_pos_rows() if pos_rows is None else pos_rows
    lines = text.splitlines()

    title, coverage = find_source_poem(text, source_author, pos_rows)
    if title is not None and coverage >= min_coverage:
        tokens = corpus_token_stream(source_author, title, pos_rows)
        mapping = _align(text, tokens)
        tagged = {(li, wi): tokens[ti] for (li, wi), ti in mapping.items()}
        log.info('source context from pos_tagged.csv: author=%s poem=%r coverage=%.2f',
                 source_author, title, coverage)
    else:
        log.info('source poem not found in corpus for %r (best coverage %.2f) — '
                 're-tagging with spaCy', source_author, coverage)
        retagged = _retag(text)
        by_line = defaultdict(list)
        for t in retagged:
            by_line[t['line']].append(t)
        tagged = {}
        for li, line in enumerate(lines):
            queue = list(by_line.get(li, []))
            for wi, word in enumerate(_words(line)):
                if queue and queue[0]['word'] == word:
                    tagged[(li, wi)] = queue.pop(0)

    contexts = []
    for target in mask_words:
        target_l = target.lower()
        for li, line in enumerate(lines):
            words = _words(line)
            if target_l not in words:
                continue
            wi = words.index(target_l)
            here = tagged.get((li, wi))
            if here is None:
                continue

            same_line = sorted(k[1] for k in tagged if k[0] == li)
            before = [i for i in same_line if i < wi]
            after = [i for i in same_line if i > wi]
            contexts.append({
                'source_word': target,
                'source_lemma': here['lemma'],
                'source_pos': here['pos'],
                'prev_pos': tagged[(li, before[-1])]['pos'] if before else None,
                'next_pos': tagged[(li, after[0])]['pos'] if after else None,
                'line': li,
            })
            break
    return contexts


def build_slots(text: str, mask_words: list, source_author: str,
                target_author: str, resources: dict = None) -> list:
    """Full §5 path: masked surface words in, per-slot candidate structures out."""
    resources = load_resources() if resources is None else resources
    return [
        build_slot(c['source_word'], c['source_lemma'], c['source_pos'],
                   c['prev_pos'], c['next_pos'], target_author, resources)
        for c in slot_contexts(text, mask_words, source_author)
    ]


def render_slot_spec(slots: list, lang: str = 'mk', style: str = 'implicit') -> str:
    """
    The per-slot replacement for the old single vocab string.

    style='implicit' keys each list by the bracketed source word; style='explicit'
    keys it by mask ordinal, since the explicit template has already removed the
    source word and must not have it handed back in the style spec.

    PROVISIONAL wording — the structure is what matters here; the exact prompt
    template is meant to be tuned once the candidate lists have been eyeballed.
    """
    lines = []
    for i, slot in enumerate(slots, 1):
        options = ', '.join(c['surface'] for c in slot['candidates']) or '—'
        key = f'[{slot["source_word"]}]' if style == 'implicit' else f'[MASK] {i}'
        lines.append(f'  {key} ({slot["pos"]}) → {options}')
    body = '\n'.join(lines)

    if lang == 'mk':
        return ('За секој означен збор, избери еден од предложените заменски '
                'зборови:\n' + body)
    return ('For each marked word, choose one of the suggested replacements:\n'
            + body)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    from llm_probe import MASK_WORDS, SOURCE_AUTHOR, TARGET_AUTHOR, load_stanza

    stanza = load_stanza()
    slots = build_slots(stanza, MASK_WORDS, SOURCE_AUTHOR, TARGET_AUTHOR)
    print(json.dumps(slots, ensure_ascii=False, indent=2))
    for slot in slots:
        print(log_slot(slot))
