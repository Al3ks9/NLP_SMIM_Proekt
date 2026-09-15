"""Qwen SFT pipeline, stage 1: train/val/test split."""

import itertools

import make_splits as ms

# poem_id is a globally unique id in the real corpus (add_poem_ids.py assigns
# it by row order across ALL authors) -- a shared counter keeps that true
# across the multiple _poems() calls a single test makes, so two authors
# never accidentally collide on the same poem_id.
_ids = itertools.count()


def _poems(author, n):
    return [{'poem_id': str(next(_ids)), 'author': author} for _ in range(n)]


def test_authors_with_enough_poems_appear_in_all_three_splits():
    poems = _poems('А', 30)
    assignment = ms.compute_splits(poems)
    splits = {assignment[p['poem_id']] for p in poems}
    assert splits == {'train', 'val', 'test'}


def test_split_sizes_are_roughly_80_10_10():
    poems = _poems('А', 100)
    assignment = ms.compute_splits(poems)
    counts = {'train': 0, 'val': 0, 'test': 0}
    for split in assignment.values():
        counts[split] += 1
    assert counts['train'] == 80
    assert counts['val'] == 10
    assert counts['test'] == 10


def test_no_poem_occurs_in_more_than_one_split():
    # trivially true of a dict keyed by poem_id, but pin it down explicitly:
    # every poem_id maps to exactly one split value.
    poems = _poems('А', 17) + _poems('Б', 9)
    assignment = ms.compute_splits(poems)
    assert len(assignment) == len(poems)
    assert set(assignment) == {p['poem_id'] for p in poems}


def test_tiny_authors_go_entirely_to_train():
    poems = _poems('Еднапесна', 1) + _poems('Двепесни', 2)
    assignment = ms.compute_splits(poems)
    assert set(assignment.values()) == {'train'}


def test_minimum_three_poems_gets_one_in_each_split():
    poems = _poems('Три', 3)
    assignment = ms.compute_splits(poems)
    assert sorted(assignment.values()) == ['test', 'train', 'val']


def test_is_reproducible_given_the_same_seed():
    poems = _poems('А', 23) + _poems('Б', 41)
    first = ms.compute_splits(poems, seed=7)
    second = ms.compute_splits(poems, seed=7)
    assert first == second


def test_different_seeds_can_produce_different_assignments():
    poems = _poems('А', 40)
    first = ms.compute_splits(poems, seed=1)
    second = ms.compute_splits(poems, seed=2)
    assert first != second


def test_mixed_corpus_every_eligible_author_covers_all_three_splits():
    poems = (_poems('Голем', 50) + _poems('Среден', 8) +
            _poems('Мал', 3) + _poems('Единствен', 1))
    assignment = ms.compute_splits(poems)

    by_author = {}
    for p in poems:
        by_author.setdefault(p['author'], set()).add(assignment[p['poem_id']])

    assert by_author['Голем'] == {'train', 'val', 'test'}
    assert by_author['Среден'] == {'train', 'val', 'test'}
    assert by_author['Мал'] == {'train', 'val', 'test'}
    assert by_author['Единствен'] == {'train'}
