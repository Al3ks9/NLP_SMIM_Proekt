"""Offset splicing — the mechanism that makes out-of-region edits impossible."""

import style_transfer as st


def test_splice_replaces_one_span():
    assert st.splice('чемрее врба проклета', [(7, 11, 'бука')]) == 'чемрее бука проклета'


def test_splice_preserves_untouched_text_exactly():
    # 'в поле' must survive: normalising it to 'во полето' destroys period voice.
    line = 'Натаму – в поле битолско'
    assert st.splice(line, []) == line


def test_splice_handles_multiple_edits_in_order():
    out = st.splice('чемрее врба проклета', [(0, 6, 'вене'), (12, 20, 'самотна')])
    assert out == 'вене врба самотна'


def test_splice_is_order_independent():
    edits = [(12, 20, 'самотна'), (0, 6, 'вене')]
    assert st.splice('чемрее врба проклета', edits) == 'вене врба самотна'


# --- Eligibility: which tokens are replacement targets at all ---------------
#
# Criterion (b) used to apply to every content POS, which made it a content
# filter rather than a style one: Конески's NOUN vocabulary is 1,297 lemmas, so
# any noun outside those 1,297 was replaced regardless of meaning. That is what
# selected врба and војник out of the Рацин stanza and swapped in вик and сокол.

SRC_TFIDF = {'гроб', 'огин', 'раца', 'таговен'}
TGT_TFIDF = {'сокол', 'мака', 'младост'}
TGT_VOCAB = {
    'NOUN': {'сокол', 'мака', 'младост', 'оган', 'рака', 'песна'},
    'ADJ': {'мрачен', 'самотен', 'тивок'},
    'VERB': {'вели', 'мине', 'мрзне'},
    'ADV': {'сосем', 'тивко'},
}


def _eligible(lemma, pos):
    return st.should_replace(lemma, pos, SRC_TFIDF, TGT_TFIDF, TGT_VOCAB)


def test_criterion_a_selects_a_source_style_marker_of_any_pos():
    # гроб is distinctive to Рацин and not to Конески — (a) applies to nouns too.
    assert _eligible('гроб', 'NOUN')
    assert _eligible('таговен', 'ADJ')


def test_criterion_a_ignores_a_marker_both_authors_share():
    assert not _eligible('мака', 'NOUN')


def test_criterion_b_selects_an_adjective_absent_from_the_target():
    # Gender is inflection on ADJ, so the replacement is realised into the
    # slot's own bundle and agreement survives.
    assert _eligible('незнаен', 'ADJ')


def test_criterion_b_selects_verbs_and_adverbs_absent_from_the_target():
    assert _eligible('чемрее', 'VERB')
    assert _eligible('натаму', 'ADV')


def test_criterion_b_does_not_select_a_noun_absent_from_the_target():
    # The heart of the gating. врба and војник are simply what the poem is
    # about; Конески not having written them is a fact about his subject
    # matter, not about his style.
    assert not _eligible('врба', 'NOUN')
    assert not _eligible('војник', 'NOUN')


def test_a_noun_the_target_uses_is_never_selected():
    assert not _eligible('песна', 'NOUN')


def test_an_adjective_the_target_uses_is_not_selected_by_b():
    assert not _eligible('тивок', 'ADJ')


def test_criterion_a_still_reaches_a_dialect_noun_variant():
    # огин/оган and раца/рака are real style transfer — same referent, different
    # form. Gating (b) off for nouns must not lose them; (a) is their path.
    assert _eligible('огин', 'NOUN')
    assert _eligible('раца', 'NOUN')


def test_with_no_source_author_only_criterion_b_can_fire():
    # source_author is optional; an empty src_tfidf disables (a) entirely.
    assert not st.should_replace('гроб', 'NOUN', set(), TGT_TFIDF, TGT_VOCAB)
    assert st.should_replace('незнаен', 'ADJ', set(), TGT_TFIDF, TGT_VOCAB)
