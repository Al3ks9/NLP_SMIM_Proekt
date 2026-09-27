import json
import math

import numpy as np
import pytest

import grpo_rewards as gr
import style_features as sf
import style_metrics as sm

AUTHOR = 'Автор'


def _calibration(tmp_path):
    scalar = {f: {'mean': 0.0, 'std': 1.0} for f in sf.SCALAR_FEATURES}
    scalar['pct_NOUN'] = {'mean': 0.5, 'std': 0.1}
    scalar['num_lines'] = {'mean': 2.0, 'std': 1.0}
    cal = {
        'meta': {}, 'content_baseline': 0.2,
        'corpus_std': {f: 1.0 for f in sf.SCALAR_FEATURES},
        'authors': {AUTHOR: {
            'n_poems': 10, 'scalar': scalar, 'tfidf_hit_p90': 0.1,
            'dist': {'pos_bigrams': {'dist': {'NOUN>VERB': 1.0}, 'cos_mean': 0.8, 'cos_std': 0.1},
                     'rhyme_endings': {'dist': {'ла': 1.0}, 'cos_mean': 0.5, 'cos_std': 0.2}}}},
    }
    path = tmp_path / 'cal.json'
    path.write_text(json.dumps(cal, ensure_ascii=False), encoding='utf-8')
    return path


class OneHotEmbedder:
    def __init__(self):
        self.vocab = {}

    def embed_lines(self, lines, cache):
        idx = [self.vocab.setdefault(l, len(self.vocab)) for l in lines]
        out = np.zeros((len(lines), 64))
        out[np.arange(len(lines)), idx] = 1.0
        return out


def fake_tagger(text):
    return [(w, 'NOUN' if i % 2 == 0 else 'VERB') for i, w in enumerate(sm.tokens(text))]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'style_match', lambda text, author: {
        'clf_target_prob': 0.5, 'clf_top1': 0.0, 'clf_predicted': '', 'tfidf_hit_rate': 0.05})
    gr._CACHE.clear()
    return gr.RewardConfig(calibration_path=_calibration(tmp_path))


SRC = 'сонце грее над гора\nптица пее во шума'
NEW = 'ветер носи лисје низ поле\nмајка чека пред порта'


def _score(cfg, gen, src=SRC):
    return gr.score(src, gen, AUTHOR, cfg, tagger=fake_tagger, embedder=OneHotEmbedder())


def test_gaussian_score_floors_sigma():
    z, s = gr.gaussian_score(1.0, mean=0.0, std=0.0, corpus_std=1.0, floor_frac=0.5)
    assert z == pytest.approx(2.0)
    assert s == pytest.approx(math.exp(-2.0))


def test_one_sided_score_is_one_at_or_above_mean():
    assert gr.one_sided_score(0.9, mean=0.8, std=0.1) == 1.0
    assert gr.one_sided_score(0.7, mean=0.8, std=0.1) == pytest.approx(math.exp(-0.5))


def test_content_score_rescales_and_clips():
    assert gr.content_score(0.2, baseline=0.2) == 0.0
    assert gr.content_score(0.6, baseline=0.2) == pytest.approx(0.5)
    assert gr.content_score(0.1, baseline=0.2) == 0.0


def test_all_components_in_unit_interval(cfg):
    row = _score(cfg, NEW)
    for key, value in row.items():
        if key.startswith(('R', 'group/', 'score/', 'gate/', 'content/')):
            assert 0.0 <= value <= 1.0, key


def test_verbatim_copy_has_zero_gated_content_but_high_raw_content(cfg):
    row = _score(cfg, SRC)
    assert row['R_content'] == 1.0
    assert row['gate/copy_novelty'] == 0.0
    assert row['R_content_g'] == 0.0


def test_combination_uses_weights(cfg):
    row = _score(cfg, NEW)
    assert row['R'] == pytest.approx(0.5 * row['R_style_g'] + 0.5 * row['R_content_g'])


def test_style_weight_zero_removes_a_group(cfg):
    cfg.style_weights = {g: 0.0 for g in gr.STYLE_GROUPS} | {'classifier': 1.0}
    assert _score(cfg, NEW)['R_style'] == pytest.approx(0.5)


def test_unknown_style_group_rejected():
    with pytest.raises(ValueError):
        gr.RewardConfig(style_weights={'no_such_group': 1.0})


# Review Focus 1
def test_unknown_target_author_raises_with_its_name(cfg):
    with pytest.raises(KeyError, match='Непознат'):
        gr.score_batch([SRC], [NEW], ['Непознат'], cfg, tagger=fake_tagger,
                       embedder=OneHotEmbedder())


# Review Focus 2
@pytest.mark.parametrize('gen', ['', '   \n\n  ', '**Poem**\n- a line', 'Here is the poem in English'])
def test_invalid_generations_score_zero_without_crashing(cfg, gen):
    row = _score(cfg, gen)
    assert row['gate/validity'] == 0.0
    assert row['R'] == 0.0


def test_score_batch_matches_score(cfg):
    emb = OneHotEmbedder()
    batch = gr.score_batch([SRC, SRC], [NEW, SRC], [AUTHOR, AUTHOR], cfg,
                           tagger=fake_tagger, embedder=emb)
    single = gr.score(SRC, SRC, AUTHOR, cfg, tagger=fake_tagger, embedder=emb)
    assert batch[1]['R'] == pytest.approx(single['R'])


def test_reward_funcs_score_once_per_batch_and_log(cfg, tmp_path):
    calls = []

    def counting_tagger(text):
        calls.append(text)
        return fake_tagger(text)

    cfg.components_log_path = tmp_path / 'components.jsonl'
    style_fn, content_fn = gr.make_reward_funcs(cfg, tagger=counting_tagger,
                                                embedder=OneHotEmbedder())
    assert style_fn.__name__ == 'style_reward' and content_fn.__name__ == 'content_reward'
    logged = {}
    kwargs = dict(prompts=['p', 'p'], completions=[NEW, SRC], source_text=[SRC, SRC],
                  target_author=[AUTHOR, AUTHOR], source_poem_id=['1', '1'],
                  log_metric=lambda k, v: logged.__setitem__(k, v), trainer_state=None)
    s = style_fn(**kwargs)
    c = content_fn(**kwargs)
    assert len(s) == len(c) == 2
    assert len(calls) == 2                       # classla once per completion, not per reward func
    assert c[1] == 0.0                            # the copy
    assert 'gate/copy_novelty' in logged and 'group/pos_profile' in logged
    lines = (tmp_path / 'components.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(lines) == 2 and json.loads(lines[0])['target_author'] == AUTHOR
