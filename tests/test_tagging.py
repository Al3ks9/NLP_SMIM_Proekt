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


class _FakeWord:
    """Stands in for a classla Word: same attributes tag_lines() reads."""

    def __init__(self, text, upos, xpos, lemma=None, feats='', misc=None):
        self.text = text
        self.upos = upos
        self.xpos = xpos
        self.lemma = lemma if lemma is not None else text
        self.feats = feats
        self.misc = misc


class _FakeSentence:
    def __init__(self, words):
        self.words = words


class _FakeDoc:
    def __init__(self, words):
        self.sentences = [_FakeSentence(words)]


def _fake_pipeline(words_by_line):
    """A fake nlp() callable: line text -> the FakeWords to hand back for it."""
    def nlp(line):
        return _FakeDoc(words_by_line[line])
    return nlp


def test_tag_lines_applies_a_global_hand_correction(monkeypatch):
    # 'гора' is hand-corrected PROPN -> NOUN globally in pos_corrections.csv
    # (see data/pos_corrections.csv). classla itself would tag it PROPN here;
    # tag_lines() must apply the same correction pos_tag_corpus.py applied to
    # the corpus, or a token the user already fixed comes back wrong.
    line = 'нечујно гора шумоли'
    fake_word = _FakeWord('гора', upos='PROPN', xpos='Npmsn')
    monkeypatch.setattr(
        tagging, 'pipeline', lambda: _fake_pipeline({line: [fake_word]}))

    tokens = tagging.tag_lines(line, Counter())

    assert len(tokens) == 1
    assert tokens[0].pos == 'NOUN'
    # apply_to() clears xpos/feats on a POS change -- see src/corrections.py.
    assert tokens[0].xpos == ''
    assert tokens[0].feats == ''


def test_span_returns_none_when_the_surface_form_cannot_be_located():
    # classla's mk models never populate word.misc with start_char/end_char --
    # only line.find() locates a span -- and a word whose text is nowhere in
    # the line (e.g. a tokenizer quirk) must not fall back to a fabricated
    # (0, 0): that would later splice a replacement in at column 0 instead of
    # skipping the token.
    word = _FakeWord('фантом', upos='NOUN', xpos='Ncmsn')
    assert tagging._span(word, 'ветер удри', cursor=0) is None


def test_tag_lines_skips_a_token_whose_span_cannot_be_located(monkeypatch):
    line = 'ветер удри'
    words = [
        _FakeWord('фантом', upos='NOUN', xpos='Ncmsn'),  # not present in `line`
        _FakeWord('удри', upos='VERB', xpos='Vmm2s-a-n'),
    ]
    monkeypatch.setattr(
        tagging, 'pipeline', lambda: _fake_pipeline({line: words}))

    tokens = tagging.tag_lines(line, Counter())

    assert len(tokens) == 1
    assert tokens[0].text == 'удри'


def test_tag_lines_gives_repeated_words_on_one_line_distinct_offsets(monkeypatch):
    # _span()'s fallback used line.find(), which always returns the first
    # occurrence -- a line repeating a word (common in this poetry corpus)
    # mapped every occurrence to the same offset, corrupting later splices
    # that key on start_char/end_char.
    line = 'ветер ветер удри'
    words = [
        _FakeWord('ветер', upos='NOUN', xpos='Ncmsn'),
        _FakeWord('ветер', upos='NOUN', xpos='Ncmsn'),
        _FakeWord('удри', upos='VERB', xpos='Vmm2s-a-n'),
    ]
    monkeypatch.setattr(
        tagging, 'pipeline', lambda: _fake_pipeline({line: words}))

    tokens = tagging.tag_lines(line, Counter())

    assert len(tokens) == 3
    first, second, _third = tokens
    assert (first.start_char, first.end_char) != (second.start_char, second.end_char)
    assert line[first.start_char:first.end_char] == 'ветер'
    assert line[second.start_char:second.end_char] == 'ветер'
    assert second.start_char > first.start_char
