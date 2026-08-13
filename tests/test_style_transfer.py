"""Offset splicing — the mechanism that makes out-of-region edits impossible."""

import style_transfer as st


def test_splice_replaces_one_span():
    assert st.splice('чемрее врба проклета', [(7, 11, 'бука')]) == 'чемрее бука проклета'


def test_splice_preserves_untouched_text_exactly():
    # 'в поле' must survive: normalising it to 'во полето' destroys period voice.
    line = 'Натаму – в поле битолско'
    assert st.splice(line, []) == line


def test_splice_handles_multiple_edits_in_order():
    out = st.splice('чемрее врба проклета', [(0, 6, 'вене'), (12, 20, 'самотна')])
    assert out == 'вене врба самотна'


def test_splice_is_order_independent():
    edits = [(12, 20, 'самотна'), (0, 6, 'вене')]
    assert st.splice('чемрее врба проклета', edits) == 'вене врба самотна'
