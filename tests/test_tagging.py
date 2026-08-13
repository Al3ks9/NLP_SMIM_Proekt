"""Tag normalization shared by the corpus build and the inference path."""

from collections import Counter

import pytest

import tagging


def test_is_content_keeps_a_real_adverb():
    assert tagging.is_content('ADV', 'Rgp') is True


def test_is_content_drops_a_bare_rg_relativizer():
    # што/кога/како carry bare 'Rg' — no degree slot — where adverbs get Rgp.
    assert tagging.is_content('ADV', 'Rg') is False


def test_is_content_drops_a_function_word():
    assert tagging.is_content('ADP', 'Sps') is False


def test_gerund_lemma_rebuilds_an_a_verb():
    assert tagging.gerund_lemma('барајќи', Counter({'бара': 40})) == ('бара', True)


def test_gerund_lemma_prefers_the_frequent_candidate():
    # бране occurs once off a mistagged token; брани is the real lemma.
    verb_lemmas = Counter({'брани': 20, 'бране': 1})
    assert tagging.gerund_lemma('бранејќи', verb_lemmas) == ('брани', True)


def test_gerund_lemma_falls_back_to_the_i_verb_when_unattested():
    lemma, known = tagging.gerund_lemma('велејќи', Counter())
    assert (lemma, known) == ('вели', False)


def test_load_verb_lemmas_counts_only_verbs():
    rows = [
        {'pos': 'VERB', 'lemma': 'бара'},
        {'pos': 'VERB', 'lemma': 'бара'},
        {'pos': 'NOUN', 'lemma': 'град'},
    ]
    assert tagging.load_verb_lemmas(rows) == Counter({'бара': 2})
