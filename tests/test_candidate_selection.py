"""
Acceptance tests for the per-slot candidate selection pipeline (spec §7).

The resource-backed tests share one session-scoped load: the embeddings file is
299 MB and the co-occurrence index is a 4.3M-pair double loop, so loading them
per test would dominate the run.
"""

import logging
import time

import pytest

import candidate_selection as cs

RACIN = 'Кочо Рацин'
KONESKI = 'Блаже Конески'

STANZA = """Натаму – в поле битолско
чемрее врба проклета –
под врбата незнаен гроб,
в гроб лежи војник непознат."""

MASK_WORDS = ['чемрее', 'проклета', 'незнаен']

MORPH_TIERS = {'exact', 'cross_author', 'nearest_attested', 'unresolved'}
SEMANTIC_TIERS = {'primary', 'relaxed_semantic', 'cross_author_similar', 'legacy_fallback'}


@pytest.fixture(scope='session')
def resources():
    return cs.load_resources()


# ── §7: the three-word Рацин→Конески case ─────────────────────────────────────

@pytest.fixture(scope='session')
def racin_slots(resources):
    return cs.build_slots(STANZA, MASK_WORDS, RACIN, KONESKI, resources)


def test_every_masked_word_gets_a_slot(racin_slots):
    assert [s['source_word'] for s in racin_slots] == MASK_WORDS


def test_each_slot_returns_at_least_one_candidate(racin_slots):
    for slot in racin_slots:
        assert slot['candidates'], f'no candidates for {slot["source_word"]}'


def test_slot_pos_matches_the_corpus_tagging(racin_slots):
    # These are the tags pos_tagged.csv actually carries at these positions.
    # spaCy had 'чемрее' as ADJ and 'проклета' as VERB here; classla reads both
    # correctly, so the corpus tagging and the correct reading now agree.
    assert {s['source_word']: s['pos'] for s in racin_slots} == {
        'чемрее': 'VERB', 'проклета': 'ADJ', 'незнаен': 'ADJ',
    }


def test_neighbour_pos_stops_at_line_boundaries(racin_slots):
    by_word = {s['source_word']: s for s in racin_slots}
    # 'чемрее' opens its line, so there is no incoming transition to score.
    assert by_word['чемрее']['prev_pos'] is None
    assert by_word['чемрее']['next_pos'] == 'NOUN'
    # 'проклета' closes its line.
    assert by_word['проклета']['next_pos'] is None


def test_every_candidate_resolves_to_a_surface_form(racin_slots):
    for slot in racin_slots:
        for c in slot['candidates']:
            assert c['surface'], f'empty surface for {c["lemma"]}'
            assert c['morph_tier'] in MORPH_TIERS
            assert c['tier'] in SEMANTIC_TIERS


def test_candidates_are_score_sorted_within_a_tier(racin_slots):
    for slot in racin_slots:
        primary = [c['score'] for c in slot['candidates'] if c['tier'] == 'primary']
        assert primary == sorted(primary, reverse=True)


def test_source_lemma_is_never_its_own_replacement(racin_slots):
    for slot in racin_slots:
        assert slot['lemma'] not in [c['lemma'] for c in slot['candidates']]


def test_log_slot_emits_the_six_required_fields(racin_slots, caplog):
    with caplog.at_level(logging.INFO, logger='candidate_selection'):
        line = cs.log_slot(racin_slots[0])
    for field in ('source_word', 'pos', 'chosen', 'score', 'morph_tier', 'semantic_tier'):
        assert f'{field}=' in line
    assert line in caplog.text


# ── §7: a deliberately sparse pool must relax, loudly ─────────────────────────

# Јосип Коцев has exactly 4 PROPN lemmas -- the thinnest author/POS pool in the
# corpus, so the semantic filter cannot leave MIN_CANDIDATES behind.
SPARSE_AUTHOR, SPARSE_POS = 'Јосип Коцев', 'PROPN'


def test_sparse_pool_is_genuinely_short(resources):
    pool = cs.candidate_pool(SPARSE_AUTHOR, SPARSE_POS, resources['author_vocab'])
    assert len(pool) < cs.MIN_CANDIDATES * 2


def test_sparse_pool_fires_the_relaxation_chain(resources, caplog):
    with caplog.at_level(logging.INFO, logger='candidate_selection'):
        ranked = cs.rank_candidates(
            'вардар', SPARSE_POS, SPARSE_AUTHOR, None, 'NOUN',
            resources['author_vocab'], resources['tfidf'], resources['embeddings'],
            resources['cooc_index'], resources['transitions'],
        )
    assert ranked, 'relaxation chain returned nothing instead of relaxing'
    assert any(c['tier'] == 'relaxed_semantic' for c in ranked)
    assert 'tier=relaxed_semantic' in caplog.text


def test_relaxation_records_the_threshold_it_settled_on(resources):
    ranked = cs.rank_candidates(
        'вардар', SPARSE_POS, SPARSE_AUTHOR, None, 'NOUN',
        resources['author_vocab'], resources['tfidf'], resources['embeddings'],
        resources['cooc_index'], resources['transitions'],
    )
    for c in ranked:
        if c['tier'] == 'relaxed_semantic':
            assert cs.SEM_FLOOR <= c['sem_threshold'] < cs.SEM_THRESHOLD


def test_relaxation_never_drops_below_the_floor(resources):
    scored, _ = cs.tier_primary(
        'вардар', SPARSE_POS, SPARSE_AUTHOR, None, None,
        resources['author_vocab'], resources['tfidf'], resources['embeddings'],
        resources['cooc_index'], resources['transitions'],
    )
    added = cs.tier_relaxed_semantic(scored, already=set(), needed=99)
    for c in added:
        assert c['sem_threshold'] >= cs.SEM_FLOOR


# ── §4 tiers, each exercised on its own ───────────────────────────────────────

def test_cross_author_tier_pulls_from_jaccard_similar_authors(resources):
    donors = cs.tier_cross_author_similar(
        'незнаен', 'ADJ', KONESKI, 'NOUN', 'NOUN',
        resources['author_vocab'], resources['tfidf'], resources['embeddings'],
        resources['cooc_index'], resources['transitions'],
    )
    assert donors
    similar = dict(cs.author_similarity(KONESKI, cs.load_pos_rows()))
    for c in donors:
        assert c['donor_author'] != KONESKI
        assert similar[c['donor_author']] >= cs.JACCARD_THRESHOLD


def test_author_similarity_is_ranked_and_excludes_self():
    ranked = cs.author_similarity(KONESKI, cs.load_pos_rows())
    assert KONESKI not in [a for a, _ in ranked]
    assert [j for _, j in ranked] == sorted((j for _, j in ranked), reverse=True)


def test_legacy_fallback_fires_only_as_last_resort_and_warns(resources, caplog):
    with caplog.at_level(logging.INFO, logger='candidate_selection'):
        ranked = cs.rank_candidates(
            'вардар', 'PROPN', 'НЕПОСТОЕЧКИ АВТОР', None, None,
            resources['author_vocab'], resources['tfidf'], resources['embeddings'],
            resources['cooc_index'], resources['transitions'],
        )
    assert ranked
    assert all(c['tier'] == 'legacy_fallback' for c in ranked)
    # Every earlier tier must have been attempted before falling back.
    for tier in ('tier=primary', 'tier=relaxed_semantic', 'tier=cross_author_similar'):
        assert tier in caplog.text
    assert any(r.levelno >= logging.WARNING for r in caplog.records), 'fallback was silent'


def test_legacy_fallback_uses_load_target_words(resources):
    from llm_probe import load_target_words

    ranked = cs.tier_legacy_fallback(
        'вардар', 'PROPN', KONESKI, None, None,
        resources['tfidf'], resources['embeddings'],
        resources['cooc_index'], resources['transitions'],
    )
    assert {c['lemma'] for c in ranked} <= set(load_target_words())


# ── §7: embeddings degrade gracefully ─────────────────────────────────────────

def test_get_embedding_returns_none_for_an_absent_lemma(resources):
    assert cs.get_embedding('незборкојнепостои', resources['embeddings']) is None


def test_get_embedding_hits_for_an_in_corpus_lemma(resources):
    vec = cs.get_embedding('гроб', resources['embeddings'])
    assert vec is not None and vec.shape == (768,)


def test_absent_source_embedding_skips_the_semantic_filter(resources):
    """An out-of-corpus source word must still produce candidates, unfiltered."""
    scored = cs.score_pool(
        [('темен', 24), ('бел', 23)], 'незборкојнепостои', 'ADJ', KONESKI,
        None, None, resources['tfidf'], resources['embeddings'],
        resources['cooc_index'], resources['transitions'],
    )
    assert all(c['sem_sim'] is None for c in scored)
    assert len(cs._semantic_filter(scored, cs.SEM_THRESHOLD)) == 2


def test_cosine_is_a_plain_dot_product(resources):
    vec = cs.get_embedding('гроб', resources['embeddings'])
    assert cs.cosine(vec, vec) == pytest.approx(1.0, abs=1e-3)


# ── surface_form: one test per morph tier ─────────────────────────────────────

def test_surface_form_exact_tier():
    # 'проклета'[-3:] == 'ета' -- the suffix key build_morph_lookup.py indexes on.
    lookup = {'A': {'самотен': {'ADJ': {'ета': 'самотена'}}}}
    assert cs.surface_form('A', 'самотен', 'ADJ', 'проклета', lookup) == ('самотена', 'exact')


def test_surface_form_falls_through_to_another_author():
    lookup = {
        'A': {'самотен': {'ADJ': {'тен': 'самотен'}}},
        'B': {'самотен': {'ADJ': {'ета': 'самотена'}}},
    }
    assert cs.surface_form('A', 'самотен', 'ADJ', 'проклета', lookup) == ('самотена', 'cross_author')


def test_surface_form_nearest_attested_prefers_the_closest_suffix():
    lookup = {'A': {'самотен': {'ADJ': {'тен': 'самотен', 'ото': 'самотното'}}}}
    form, tier = cs.surface_form('A', 'самотен', 'ADJ', 'проклета', lookup)
    assert tier == 'nearest_attested'
    assert form in {'самотен', 'самотното'}


def test_surface_form_unresolved_returns_the_lemma():
    assert cs.surface_form('A', 'самотен', 'ADJ', 'проклета', {}) == ('самотен', 'unresolved')


def test_surface_form_short_source_word_uses_the_whole_word():
    lookup = {'A': {'сон': {'NOUN': {'ме': 'соне'}}}}
    assert cs.surface_form('A', 'сон', 'NOUN', 'МЕ', lookup) == ('соне', 'exact')


# ── §7: co-occurrence build cost at full corpus scale ─────────────────────────

def test_cooccurrence_index_builds_in_acceptable_time():
    pos_rows = cs.load_pos_rows()
    t0 = time.time()
    index = cs.build_cooccurrence_index(pos_rows)
    elapsed = time.time() - t0
    assert elapsed < 10.0, f'co-occurrence build took {elapsed:.1f}s'
    assert index[('гроб', 'NOUN')][('врба', 'NOUN')] > 0


def test_cooccurrence_index_is_symmetric():
    index = cs.load_cooccurrence_index()
    a, b = ('гроб', 'NOUN'), ('врба', 'NOUN')
    assert index[a][b] == index[b][a]


def test_cooccurrence_index_is_not_capped_to_skg_nodes():
    """The whole point of bypassing the graph: rare lemmas are reachable too."""
    index = cs.load_cooccurrence_index()
    assert len(index) > 10_000


# ── §5 integration: source-side context lookup ────────────────────────────────

def test_slot_contexts_reads_lemma_and_pos_from_the_corpus():
    contexts = cs.slot_contexts(STANZA, MASK_WORDS, RACIN)
    assert [c['source_word'] for c in contexts] == MASK_WORDS
    assert all(c['source_lemma'] and c['source_pos'] in cs.KEEP_POS for c in contexts)


def test_find_source_poem_identifies_the_stanzas_poem():
    title, coverage = cs.find_source_poem(STANZA, RACIN, cs.load_pos_rows())
    assert title == 'Балада за непознатиот'
    assert coverage > 0.5


def test_render_slot_spec_hides_source_words_in_explicit_mode(racin_slots):
    explicit = cs.render_slot_spec(racin_slots, 'en', 'explicit')
    for word in MASK_WORDS:
        assert word not in explicit
    implicit = cs.render_slot_spec(racin_slots, 'en', 'implicit')
    assert all(word in implicit for word in MASK_WORDS)
