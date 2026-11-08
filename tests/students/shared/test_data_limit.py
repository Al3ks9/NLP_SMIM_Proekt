"""Per-author data counts next to per-author identification rates."""

import csv

from students.shared import data_limit as dl


def test_author_table_counts_and_rates(tmp_path, monkeypatch):
    songs = [{'poem_id': str(i), 'author': a} for i, a in
             enumerate(['A', 'A', 'A', 'B', 'B', 'C'], 1)]
    monkeypatch.setattr(dl, 'load_stripped_songs', lambda: songs)
    monkeypatch.setattr(dl, 'load_splits', lambda: {'1': 'train', '2': 'train', '3': 'test',
                                                    '4': 'train', '5': 'val', '6': 'train'})
    monkeypatch.setattr(dl, 'load_synthetic_rows',
                        lambda: [{'target_author': 'A'}] * 3 + [{'target_author': 'B'}])
    rows_csv = tmp_path / 'rows.csv'
    with open(rows_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'target_author', 'clf_top1', 'judge_top1', 'clf_predicted'])
        w.writeheader()
        w.writerows([
            {'run': 'sft', 'target_author': 'A', 'clf_top1': '1', 'judge_top1': '1', 'clf_predicted': 'A'},
            {'run': 'sft', 'target_author': 'A', 'clf_top1': '0', 'judge_top1': '1', 'clf_predicted': 'B'},
            {'run': 'sft', 'target_author': 'B', 'clf_top1': '0', 'judge_top1': '0', 'clf_predicted': 'A'},
            {'run': 'sft', 'target_author': 'B', 'clf_top1': '0', 'judge_top1': '0', 'clf_predicted': 'A'},
            {'run': 'REAL POEMS', 'target_author': 'A', 'clf_top1': '1', 'judge_top1': '1', 'clf_predicted': 'A'},
        ])

    table = {(r['run'], r['target_author']): r for r in dl.author_table(rows_csv)}

    a = table[('sft', 'A')]
    assert (a['corpus_poems'], a['train_poems'], a['teacher_examples']) == (3, 2, 3)
    assert a['clf_top1'] == 0.5 and a['judge_top1'] == 1.0
    assert a['share_predicted_as_target'] == 0.75            # 3 of sft's 4 outputs read as A
    b = table[('sft', 'B')]
    assert (b['corpus_poems'], b['train_poems'], b['teacher_examples']) == (2, 1, 1)
    assert b['share_predicted_as_target'] == 0.25
    assert ('REAL POEMS', 'A') not in table
