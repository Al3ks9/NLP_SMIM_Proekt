"""Style profile -> descriptive text (step 3): percentile ranking, banding, and
the prose translation of top_pos_bigrams / top_rhyme_endings."""

import style_narrator as sn

# Three synthetic authors, one numeric column, easy to hand-check percentiles for.
ROWS = [
    {'author': 'А', 'pct_ADJ': '0.10'},
    {'author': 'Б', 'pct_ADJ': '0.30'},
    {'author': 'В', 'pct_ADJ': '0.20'},
]


# ── Percentile ranking ──────────────────────────────────────────────────────────

def test_percentile_rank_orders_lowest_value_to_zero():
    pct = sn.author_percentiles(ROWS, ['pct_ADJ'])
    assert pct['А']['pct_ADJ'] == 0.0


def test_percentile_rank_orders_highest_value_to_100():
    pct = sn.author_percentiles(ROWS, ['pct_ADJ'])
    assert pct['Б']['pct_ADJ'] == 100.0


def test_percentile_rank_puts_the_middle_value_at_50():
    pct = sn.author_percentiles(ROWS, ['pct_ADJ'])
    assert pct['В']['pct_ADJ'] == 50.0


def test_percentile_rank_breaks_ties_by_author_name():
    # Equal values must not crash or produce a nondeterministic order.
    tied = [
        {'author': 'Ц', 'pct_ADJ': '0.20'},
        {'author': 'А', 'pct_ADJ': '0.20'},
    ]
    pct = sn.author_percentiles(tied, ['pct_ADJ'])
    assert pct['А']['pct_ADJ'] < pct['Ц']['pct_ADJ']


# ── Tertile banding ──────────────────────────────────────────────────────────────

def test_band_bottom_third_is_low():
    assert sn.band(0.0) == 'low'
    assert sn.band(20.0) == 'low'


def test_band_middle_third_is_mid():
    assert sn.band(50.0) == 'mid'


def test_band_top_third_is_high():
    assert sn.band(100.0) == 'high'
    assert sn.band(70.0) == 'high'


# ── Bigram / rhyme sentences ─────────────────────────────────────────────────────

def test_describe_bigrams_mentions_the_top_pair():
    text = sn.describe_bigrams('ADJ>NOUN|NOUN>VERB|VERB>NOUN', n=1)
    assert 'adjectives directly before nouns' in text


def test_describe_bigrams_handles_an_unmapped_pair_with_a_generic_phrase():
    text = sn.describe_bigrams('PROPN>ADV', n=1)
    assert 'PROPN directly before ADV' in text


def test_describe_bigrams_empty_input_returns_empty_string():
    assert sn.describe_bigrams('') == ''


def test_describe_rhymes_lists_the_endings():
    text = sn.describe_rhymes('ни|ле|и', n=3)
    assert '"-ни"' in text and '"-ле"' in text and '"-и"' in text


def test_describe_rhymes_empty_input_returns_empty_string():
    assert sn.describe_rhymes('') == ''


# ── describe_style() against the real corpus ──────────────────────────────────────

def test_describe_style_returns_nonempty_prose_for_a_real_author():
    text = sn.describe_style('Блаже Конески')
    assert isinstance(text, str)
    assert len(text) > 100
    assert 'Блаже Конески' in text


def test_describe_style_raises_for_an_unknown_author():
    import pytest
    with pytest.raises(KeyError):
        sn.describe_style('Не Постои Автор')
