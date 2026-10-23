import json

import gemma_compat as compat


def test_check_load_fails_on_missing_language_weights():       # Review Focus 5
    ok, detail = compat.evaluate_load(
        missing_keys=['model.language_model.layers.0.mlp.up_proj.weight'], unexpected_keys=[])
    assert not ok and 'language_model' in detail


def test_check_load_fails_on_missing_text_only_layer_weights():
    ok, _ = compat.evaluate_load(missing_keys=['model.layers.3.self_attn.q_proj.weight'],
                                 unexpected_keys=[])
    assert not ok


def test_check_load_tolerates_dropped_tower_weights():
    # loading text-only from the multimodal checkpoint leaves the towers unused
    ok, _ = compat.evaluate_load(missing_keys=[],
                                 unexpected_keys=['model.vision_tower.encoder.layers.0.mlp.fc1.weight'])
    assert ok


def test_check_lora_targets_rejects_tower_and_per_layer_matches():
    assert not compat.evaluate_lora_matches(['model.vision_tower.encoder.layers.0.self_attn.q_proj'])[0]
    assert not compat.evaluate_lora_matches(['model.layers.0.per_layer_projection'])[0]
    assert not compat.evaluate_lora_matches([])[0]
    assert compat.evaluate_lora_matches(['model.language_model.layers.0.self_attn.q_proj'])[0]


def test_report_exit_code_reflects_any_failed_check(tmp_path):
    assert compat.write_report({'checks': {'a': 'pass', 'b': 'fail: x'}}, tmp_path / 'r.json') == 1
    assert compat.write_report({'checks': {'a': 'pass'}}, tmp_path / 'r.json') == 0
    assert json.loads((tmp_path / 'r.json').read_text(encoding='utf-8')) == {'checks': {'a': 'pass'}}


def test_length_summary_counts_examples_over_each_limit():
    s = compat.length_summary([100, 900, 1100, 2100])
    assert s['n_over_1024'] == 2 and s['n_over_2048'] == 1 and s['max'] == 2100


def test_seq_length_choice_is_the_smallest_that_truncates_nothing():
    assert compat.choose_max_seq_length({'n_over_1024': 0, 'n_over_2048': 0, 'n_over_4096': 0}) == 1024
    assert compat.choose_max_seq_length({'n_over_1024': 3, 'n_over_2048': 0, 'n_over_4096': 0}) == 2048
    assert compat.choose_max_seq_length({'n_over_1024': 9, 'n_over_2048': 1, 'n_over_4096': 1}) is None


def test_check_load_runs_on_a_real_architecture_checkpoint(tmp_path):   # review C2
    import dataclasses
    import model_families as mf
    from test_model_families import _tiny_gemma4_checkpoint
    _tiny_gemma4_checkpoint(tmp_path / 'ckpt')
    family = dataclasses.replace(mf.get_family('gemma4'), hf_name=str(tmp_path / 'ckpt'))
    report = {'checks': {}}
    compat.check_load(report, family)
    assert report['checks']['load'] == 'pass', report
    assert report['loader_used'] == 'gemma4_text' and report['missing_keys'] == []
    json.dumps(report)   # report must be JSON-serialisable (no sets)


def test_merge_agreement_is_judged_per_token():   # review minor 3
    ok, _ = compat.merge_agrees(lp_adapter=-600.0, lp_merged=-603.0, n_tokens=300)   # 0.01/token
    assert ok
    ok, _ = compat.merge_agrees(lp_adapter=-600.0, lp_merged=-630.0, n_tokens=300)   # 0.1/token
    assert not ok


def test_seq_length_choice_extends_to_4096():   # review I2: 10 train rows exceed 2048
    assert compat.choose_max_seq_length({'n_over_1024': 31, 'n_over_2048': 10, 'n_over_4096': 0}) == 4096
