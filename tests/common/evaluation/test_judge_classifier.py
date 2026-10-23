from pathlib import Path

import pytest

import judge_classifier as jc

SRC = Path(__file__).resolve().parent.parent / 'src'

A, B = 'Автор А', 'Автор Б'
A_WORDS = 'ѕвезда месечина ноќ сонце небо'
B_WORDS = 'нива плуг жито селанец орање'


def _poems():
    poems = []
    for i in range(12):
        for author, words in ((A, A_WORDS), (B, B_WORDS)):
            poems.append({'poem_id': f'{author}-{i}', 'author': author,
                          'song_text': f'{words}\n{words} {i}'})
    return poems


def _splits(poems):
    # the last two poems of each author are held out
    return {p['poem_id']: ('val' if p['poem_id'].endswith(('-10', '-11')) else 'train')
            for p in poems}


def test_judge_trains_on_train_split_only():
    poems = _poems()
    texts, labels, ids = jc.training_set(poems, _splits(poems), min_poems=5)
    assert ids and all(not i.endswith(('-10', '-11')) for i in ids)
    assert set(labels) == {A, B}


def test_authors_below_min_train_poems_are_excluded():
    poems = _poems() + [{'poem_id': 'Ц-0', 'author': 'Ц', 'song_text': 'едно'}]
    _, labels, _ = jc.training_set(poems, _splits(poems) | {'Ц-0': 'train'}, min_poems=5)
    assert 'Ц' not in labels


def test_judge_match_reads_author_style():
    poems = _poems()
    model = jc.fit(*jc.training_set(poems, _splits(poems), min_poems=5)[:2])
    hit = jc.judge_match(f'{A_WORDS}\nнебо ноќ', A, model)
    miss = jc.judge_match(f'{B_WORDS}\nжито нива', A, model)
    assert hit['judge_top1'] == 1.0 and hit['judge_predicted'] == A
    assert miss['judge_top1'] == 0.0 and hit['judge_target_prob'] > miss['judge_target_prob']


def test_judge_match_on_empty_or_unknown_author_is_zero():
    poems = _poems()
    model = jc.fit(*jc.training_set(poems, _splits(poems), min_poems=5)[:2])
    assert jc.judge_match('', A, model)['judge_top1'] == 0.0
    assert jc.judge_match(A_WORDS, 'Непознат', model)['judge_target_prob'] == 0.0


def test_judge_is_never_part_of_the_reward():
    # the judge exists because the GRPO policy optimises classifier.pkl; if a
    # reward ever read the judge, it would stop being an independent check
    for name in ('grpo_rewards.py', 'train_grpo.py', 'style_metrics.py'):
        assert 'judge_classifier' not in (SRC / name).read_text(encoding='utf-8'), name
