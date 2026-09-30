import numpy as np
import pytest

import content_similarity as cs


def _unit(*rows):
    a = np.array(rows, dtype=float)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_line_f1_identical_is_one():
    e = _unit([1, 0], [0, 1])
    assert cs.line_f1(e, e) == pytest.approx((1.0, 1.0, 1.0))


def test_line_f1_partial_coverage_lowers_recall_not_precision():
    src = _unit([1, 0], [0, 1])
    gen = _unit([1, 0])
    p, r, f1 = cs.line_f1(src, gen)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(2 / 3)


def test_line_f1_empty_side_is_zero():
    assert cs.line_f1(np.zeros((0, 2)), _unit([1, 0])) == (0.0, 0.0, 0.0)


@pytest.mark.slow
def test_line_embedder_caches_only_when_asked():
    emb = cs.LineEmbedder()
    out = emb.embed_lines(['сонце грее', 'птица пее'], cache=True)
    assert out.shape[0] == 2
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    assert set(emb._cache) == {'сонце грее', 'птица пее'}
    emb.embed_lines(['нов ред'], cache=False)
    assert 'нов ред' not in emb._cache
    assert emb.embed_lines([], cache=False).shape[0] == 0
