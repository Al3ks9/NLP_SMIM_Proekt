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
