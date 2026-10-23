from collections import Counter

import pytest

import style_features as sf

POEM = 'сонце грее над гора\nптица пее\n\nреката тече\nкон морето сина'


def test_line_stats_counts_lines_stanzas_and_whitespace_tokens():
    assert sf.line_stats(POEM) == {'tokens': 11, 'lines': 4, 'stanzas': 2, 'stanza_lines': 4}


def test_poem_features_scalars():
    tokens = [('сонце', 'NOUN'), ('грее', 'VERB'), ('гора', 'NOUN'), ('птица', 'NOUN'),
              ('пее', 'VERB'), ('река', 'NOUN'), ('тече', 'VERB'), ('море', 'NOUN'),
              ('син', 'ADJ')]
    f = sf.poem_features(tokens, POEM)
    assert f['pct_NOUN'] == pytest.approx(5 / 9)
    assert f['pct_VERB'] == pytest.approx(3 / 9)
    assert f['pct_ADJ'] == pytest.approx(1 / 9)
    assert f['pct_ADV'] == 0.0 and f['pct_PROPN'] == 0.0
    assert f['lexical_diversity'] == 1.0          # 9 distinct lemmas, below MATTR window -> TTR
    assert f['avg_tokens_per_line'] == 2.75
    assert f['num_lines'] == 4.0 and f['num_stanzas'] == 2.0
    assert f['avg_lines_per_stanza'] == 2.0
    assert f['pos_bigrams']['NOUN>VERB'] == 3
    assert sum(f['rhyme_endings'].values()) == 4


def test_poem_features_on_empty_input_is_all_zero_not_an_error():
    f = sf.poem_features([], '')
    assert all(f[k] == 0.0 for k in sf.SCALAR_FEATURES)
    assert f['pos_bigrams'] == Counter() and f['rhyme_endings'] == Counter()


def test_dist_cosine():
    assert sf.dist_cosine(Counter({'a': 2}), {'a': 1.0}) == pytest.approx(1.0)
    assert sf.dist_cosine(Counter({'a': 1}), {'b': 1.0}) == 0.0
    assert sf.dist_cosine(Counter(), {'a': 1.0}) == 0.0


def test_normalise_sums_to_one():
    assert sf.normalise(Counter({'a': 1, 'b': 3})) == {'a': 0.25, 'b': 0.75}
