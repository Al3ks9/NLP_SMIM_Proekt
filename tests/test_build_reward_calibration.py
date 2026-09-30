from collections import Counter

import numpy as np
import pytest

import build_reward_calibration as brc
import style_features as sf


def _feats(noun, bigrams):
    f = {k: 0.0 for k in sf.SCALAR_FEATURES}
    f['pct_NOUN'] = noun
    f['pos_bigrams'] = Counter(bigrams)
    f['rhyme_endings'] = Counter({'ла': 1})
    return f


def test_author_calibration_scalar_mean_std_and_leave_one_out_cosine():
    poems = [_feats(0.4, {'NOUN>VERB': 2}), _feats(0.6, {'NOUN>VERB': 1, 'ADJ>NOUN': 1})]
    cal = brc.author_calibration(poems, hit_rates=[0.0, 0.1], palette_size=20)
    assert cal['n_poems'] == 2
    assert cal['scalar']['pct_NOUN']['mean'] == pytest.approx(0.5)
    assert cal['scalar']['pct_NOUN']['std'] == pytest.approx(0.1)
    assert cal['dist']['pos_bigrams']['dist'] == pytest.approx({'NOUN>VERB': 0.75, 'ADJ>NOUN': 0.25})
    # leave-one-out: poem 1 vs poem 2's dist and vice versa, never itself
    c1 = sf.dist_cosine(Counter({'NOUN>VERB': 2}), {'NOUN>VERB': 0.5, 'ADJ>NOUN': 0.5})
    c2 = sf.dist_cosine(Counter({'NOUN>VERB': 1, 'ADJ>NOUN': 1}), {'NOUN>VERB': 1.0})
    assert cal['dist']['pos_bigrams']['cos_mean'] == pytest.approx((c1 + c2) / 2)


def test_tfidf_p90_is_floored_at_one_palette_word():
    cal = brc.author_calibration([_feats(0.5, {}), _feats(0.5, {})],
                                 hit_rates=[0.0, 0.0], palette_size=20)
    assert cal['tfidf_hit_p90'] == pytest.approx(1 / 20)


class _OneHotEmbedder:
    """Identical lines -> cosine 1, different lines -> 0."""
    def __init__(self):
        self.vocab = {}

    def embed_lines(self, lines, cache):
        idx = [self.vocab.setdefault(l, len(self.vocab)) for l in lines]
        out = np.zeros((len(lines), 64))
        out[np.arange(len(lines)), idx] = 1.0
        return out


def test_content_baseline_only_pairs_different_authors():
    poems = [{'author': 'А', 'song_text': 'ист ред'},
             {'author': 'А', 'song_text': 'ист ред'},
             {'author': 'Б', 'song_text': 'друг ред'}]
    # same-author pairs would score 1.0; cross-author pairs score 0.0
    assert brc.content_baseline(poems, _OneHotEmbedder(), n_pairs=20, seed=1) == 0.0
