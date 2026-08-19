"""poem_id migration: the pure helper that resolves a poem/row-scoped correction's
(author, song_title) to exactly one poem_id, refusing rather than guessing."""

import pytest

import add_poem_ids as api


IDS_BY_TITLE = {
    ('А', 'Уникатна'): [5],
    ('А', 'Дупликат'): [12, 40],
}


def test_resolves_an_unambiguous_title_to_its_poem_id():
    row = {'author': 'А', 'song_title': 'Уникатна', 'scope': 'poem', 'word': 'х', 'pos': 'NOUN'}
    assert api.resolve_correction_poem_id(row, IDS_BY_TITLE) == 5


def test_global_scope_rows_are_not_resolved():
    row = {'author': 'А', 'song_title': 'Дупликат', 'scope': '', 'word': 'х', 'pos': 'NOUN'}
    assert api.resolve_correction_poem_id(row, IDS_BY_TITLE) is None


def test_raises_on_an_ambiguous_title_rather_than_guessing():
    row = {'author': 'А', 'song_title': 'Дупликат', 'scope': 'poem', 'word': 'х', 'pos': 'NOUN'}
    with pytest.raises(SystemExit):
        api.resolve_correction_poem_id(row, IDS_BY_TITLE)


def test_raises_when_the_title_matches_no_poem_at_all():
    row = {'author': 'А', 'song_title': 'Не Постои', 'scope': 'row', 'word': 'х', 'pos': 'NOUN'}
    with pytest.raises(SystemExit):
        api.resolve_correction_poem_id(row, IDS_BY_TITLE)
