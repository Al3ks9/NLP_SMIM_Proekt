import json

import pytest

import evaluate_reward as er


def test_candidates_cover_every_kind():
    case = json.loads(er.FIXTURES_PATH.read_text(encoding='utf-8'))[0]
    cands = er.build_candidates(case)
    assert set(cands) == {'real_target', 'verbatim_source', 'near_copy', 'gemma',
                          'repetition', 'markdown_english', 'empty'}
    assert cands['empty'] == ''
    assert cands['near_copy'] != cands['verbatim_source']


def test_check_expectations_reports_violations():
    ok = {'R_content_g': 0.0, 'R_style': 0.4, 'R_content': 0.9, 'R': 0.2}
    scores = {
        'verbatim_source': dict(ok),
        'near_copy': dict(ok),
        'real_target': {'R_content_g': 0.1, 'R_style': 0.3, 'R_content': 0.1, 'R': 0.2},  # style too low
        'gemma': {'R_content_g': 0.3, 'R_style': 0.5, 'R_content': 0.3, 'R': 0.4},
        'repetition': {'R_content_g': 0.0, 'R_style': 0.1, 'R_content': 0.0, 'R': 0.05},
        'markdown_english': {'R_content_g': 0.0, 'R_style': 0.0, 'R_content': 0.0, 'R': 0.0},
        'empty': {'R_content_g': 0.0, 'R_style': 0.0, 'R_content': 0.0, 'R': 0.0},
    }
    failures = er.check_expectations(scores)
    assert len(failures) == 1 and 'real_target' in failures[0]


@pytest.mark.slow
def test_fixtures_pass_with_the_real_reward():
    """The point of the whole reward-validation step: real classla, real
    embedder, real calibration. A failure here is a finding to report, not a
    threshold to loosen."""
    assert er.main(['--fixtures']) == 0


def test_summarise_groups_separates_within_and_between_spread():
    groups = [[{'R': 0.1}, {'R': 0.3}], [{'R': 0.5}, {'R': 0.7}]]
    s = er.summarise_groups(groups, ['R'])['R']
    assert s['mean'] == pytest.approx(0.4)
    assert s['within_group_std'] == pytest.approx(0.1)
    assert s['between_prompt_std'] == pytest.approx(0.2)


def test_groups_sample_like_grpo_training(monkeypatch, tmp_path):
    """--groups exists to predict GRPO's within-group reward spread, so it has
    to sample the way GRPOTrainer does (train_grpo defaults), not the way
    validation generation does."""
    import argparse
    import generate_validation as gv
    import grpo_data
    import grpo_rewards as gr
    import train_grpo as tg

    calls = []
    monkeypatch.setattr(gv, 'load_adapter_model', lambda a, base_model=None: (None, None))
    monkeypatch.setattr(gv, 'generate_one', lambda *a, **kw: calls.append(kw) or 'песна')
    monkeypatch.setattr(grpo_data, 'build_prompt_rows', lambda seed: [
        {'source_poem_id': '1', 'source_text': 'извор', 'target_author': 'Б'}])
    monkeypatch.setattr(gr, 'score_batch', lambda s, g, t, cfg=None: [
        {k: 0.0 for k in er.GROUP_KEYS} for _ in g])
    args = er.build_parser().parse_args(['--groups', '--num-prompts', '1',
                                         '--num-generations', '2',
                                         '--groups-out', str(tmp_path / 'g.csv')])
    er.run_groups(args, gr.RewardConfig())
    train = tg.parse_args([])
    assert all((kw['temperature'], kw['top_p'], kw['top_k']) ==
               (train.temperature, train.top_p, train.top_k) for kw in calls)
    assert all(kw['max_new_tokens'] == train.max_completion_length for kw in calls)


@pytest.mark.slow
def test_single_line_scores_well_below_the_full_poem_it_came_from():
    """A one-line output is not gated to zero (real poems are never gated on
    length); structure and content must still rank it well below the full
    poem. Real classla, embedder and calibration."""
    import grpo_rewards as gr
    for case in json.loads(er.FIXTURES_PATH.read_text(encoding='utf-8')):
        source = er._poem(case['source_poem_id'])['song_text']
        full = er._gemma(case['source_poem_id'], case['target_author'])
        first = next(l for l in full.splitlines() if l.strip())
        rows = gr.score_batch([source, source], [full, first], [case['target_author']] * 2)
        assert rows[1]['R'] < rows[0]['R'] - 0.05, case
