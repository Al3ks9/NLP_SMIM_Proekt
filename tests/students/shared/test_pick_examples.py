"""One (source poem, target) pair across every system's generation CSV."""

import csv

from students.shared import pick_examples as pe


def _csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['source_poem_id', 'target_author', 'generated_poem'])
        w.writeheader()
        w.writerows(rows)
    return path


def test_pick_finds_pair_in_each_csv(tmp_path):
    a = _csv(tmp_path / 'qwen3-sft-merged_test.csv',
             [{'source_poem_id': '43', 'target_author': 'Блаже Конески', 'generated_poem': 'А'},
              {'source_poem_id': '44', 'target_author': 'Блаже Конески', 'generated_poem': 'Б'}])
    b = _csv(tmp_path / 'qwen3-grpo_test.csv',
             [{'source_poem_id': '44', 'target_author': 'Блаже Конески', 'generated_poem': 'В'}])
    assert pe.pick('43', 'Блаже Конески', [a, b]) == [
        {'system': 'qwen3-sft-merged_test', 'generated_poem': 'А'}]


def test_failure_examples_one_per_kind(tmp_path):
    rows = tmp_path / 'rows.csv'
    with open(rows, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'source_poem_id', 'target_author', 'line_uniqueness',
                                          'copy_novelty', 'clf_predicted'])
        w.writeheader()
        w.writerows([
            {'run': 'sft', 'source_poem_id': '1', 'target_author': 'Ристо Лазаров', 'line_uniqueness': '0.5',
             'copy_novelty': '1', 'clf_predicted': 'Ристо Лазаров'},
            {'run': 'grpo', 'source_poem_id': '2', 'target_author': 'Ристо Лазаров', 'line_uniqueness': '1',
             'copy_novelty': '0.9', 'clf_predicted': 'Ристо Лазаров'},
            {'run': 'sft', 'source_poem_id': '3', 'target_author': 'Ристо Лазаров', 'line_uniqueness': '1',
             'copy_novelty': '1', 'clf_predicted': 'Блаже Конески'},
            {'run': 'REAL POEMS', 'source_poem_id': '', 'target_author': 'Ристо Лазаров',
             'line_uniqueness': '0.1', 'copy_novelty': '', 'clf_predicted': 'Ристо Лазаров'},
        ])
    found = pe.failure_examples(rows, seed=0)
    assert found['degenerate']['source_poem_id'] == '1'
    assert found['copied_line']['source_poem_id'] == '2'
    assert found['wrong_voice']['source_poem_id'] == '3'


def test_csv_for_run_matches_stems_and_legacy_labels(tmp_path):
    csvs = [tmp_path / f'{s}_test.csv' for s in ('base-Qwen3-4B', 'qwen3-grpo-v2', 'gemma4-e4b-grpo')]
    assert pe.csv_for_run('grpo-v2', csvs).name == 'qwen3-grpo-v2_test.csv'        # old hand label
    assert pe.csv_for_run('gemma4-e4b-grpo', csvs).name == 'gemma4-e4b-grpo_test.csv'
    assert pe.csv_for_run('nonexistent', csvs) is None
    assert pe.csv_for_run('gemma-teacher', csvs + [tmp_path / 'teacher_test.csv']).name == 'teacher_test.csv'


def test_failure_examples_can_skip_runs(tmp_path):
    rows = tmp_path / 'rows.csv'
    with open(rows, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'source_poem_id', 'target_author', 'line_uniqueness',
                                          'copy_novelty', 'clf_predicted'])
        w.writeheader()
        w.writerow({'run': 'base', 'source_poem_id': '1', 'target_author': 'X', 'line_uniqueness': '0.1',
                    'copy_novelty': '0', 'clf_predicted': 'X'})
    assert pe.failure_examples(rows, skip=lambda run: run.startswith('base')) == {}
