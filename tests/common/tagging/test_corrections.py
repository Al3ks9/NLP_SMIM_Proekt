"""poem_id-based correction keying — replaces (author, song_title), which is not a
unique poem key in this corpus (~15 shared titles; see CLAUDE.md's Data quality
context)."""

import corrections as C


def test_key_for_global_scope_ignores_poem_id():
    row = {'word': 'гора', 'pos': 'NOUN', 'poem_id': ''}
    assert C.key_for(row, '') == ('гора', 'NOUN')


def test_key_for_poem_scope_uses_poem_id_not_title():
    row = {'word': 'гора', 'pos': 'NOUN', 'poem_id': '42'}
    assert C.key_for(row, 'poem') == ('гора', 'NOUN', 42)


def test_key_for_row_scope_uses_poem_id_and_context():
    row = {'word': 'гора', 'pos': 'NOUN', 'poem_id': '42', 'context': 'некоја реченица'}
    assert C.key_for(row, 'row') == ('гора', 'NOUN', 42, 'некоја реченица')


def test_two_same_titled_poems_do_not_collide_at_poem_scope():
    # The whole point: (author, song_title) alone used to be the key, so two poems
    # sharing a title would collide. poem_id keeps them apart.
    row_a = {'word': 'гора', 'pos': 'NOUN', 'poem_id': '1'}
    row_b = {'word': 'гора', 'pos': 'NOUN', 'poem_id': '2'}
    assert C.key_for(row_a, 'poem') != C.key_for(row_b, 'poem')


def test_lookup_finds_a_poem_scoped_correction_by_poem_id():
    table = {('гора', 'NOUN', 42): ('PROPN', '')}
    assert C.lookup(table, 'гора', 'NOUN', 42, 'иргелава реченица') == ('PROPN', '')


def test_lookup_does_not_match_a_different_poem_id():
    table = {('гора', 'NOUN', 42): ('PROPN', '')}
    assert C.lookup(table, 'гора', 'NOUN', 43, 'иргелава реченица') is None


def test_lookup_row_scope_beats_poem_scope():
    table = {
        ('гора', 'NOUN', 42): ('PROPN', ''),
        ('гора', 'NOUN', 42, 'точна реченица'): ('VERB', ''),
    }
    assert C.lookup(table, 'гора', 'NOUN', 42, 'точна реченица') == ('VERB', '')


def test_apply_to_drops_a_token_on_DROP():
    table = {('гора', 'NOUN', 42): (C.DROP, '')}
    assert C.apply_to(table, 'гора', 'NOUN', 'гора', 'Ncfsn', 'Case=Nom', 42, '') is None


def test_apply_to_clears_xpos_and_feats_on_a_pos_change():
    table = {('гора', 'NOUN', 42): ('PROPN', '')}
    result = C.apply_to(table, 'гора', 'NOUN', 'гора', 'Ncfsn', 'Case=Nom', 42, '')
    assert result == ('PROPN', 'гора', '', '')
