"""Feats parsing and the relaxation order (spec §Relaxation order)."""

import morph

NOUN_FEM_SG_DEF = 'Case=Nom|Definite=Def|Gender=Fem|Number=Sing'
VERB_PRES_3SG = 'Aspect=Prog|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres'
ADJ_FEM_SG = 'Definite=Ind|Degree=Pos|Gender=Fem|Number=Sing'


def test_parse_feats_round_trips():
    assert morph.render_feats(morph.parse_feats(NOUN_FEM_SG_DEF)) == NOUN_FEM_SG_DEF


def test_render_feats_is_alphabetical():
    # classla writes keys alphabetically; anything we build must match or keys miss.
    assert morph.render_feats({'Number': 'Sing', 'Case': 'Nom'}) == 'Case=Nom|Number=Sing'


def test_parse_feats_handles_the_empty_string():
    assert morph.parse_feats('') == {}


def test_relaxation_drops_polarity_first_on_a_verb():
    first = next(morph.relax_feats(VERB_PRES_3SG, 'VERB'))
    assert 'Polarity' not in first
    assert 'Person=3' in first and 'Tense=Pres' in first


def test_relaxation_is_cumulative():
    steps = list(morph.relax_feats(VERB_PRES_3SG, 'VERB'))
    widths = [len(morph.parse_feats(s)) for s in steps]
    assert widths == sorted(widths, reverse=True)
    assert len(set(widths)) == len(widths), 'each step must drop exactly one feature'


def test_relaxation_drops_gender_for_a_noun():
    steps = list(morph.relax_feats(NOUN_FEM_SG_DEF, 'NOUN'))
    assert any('Gender' not in s for s in steps)


def test_relaxation_never_drops_gender_for_an_adjective():
    # Gender is agreement on ADJ — dropping it produces 'тивка гроб'.
    for step in morph.relax_feats(ADJ_FEM_SG, 'ADJ'):
        assert 'Gender=Fem' in step


def test_relaxation_never_drops_number_or_definite():
    for step in morph.relax_feats(NOUN_FEM_SG_DEF, 'NOUN'):
        assert 'Number=Sing' in step and 'Definite=Def' in step


def test_relaxation_never_empties_the_bundle():
    # '' is reserved for genuinely featless tokens; matching it from a relaxed
    # lookup would return an arbitrary form while claiming a morphological match.
    # Degree is an ADV's only feature 76.9% of the time and is droppable.
    assert list(morph.relax_feats('Degree=Pos', 'ADV')) == []


def test_relaxation_of_an_empty_bundle_yields_nothing():
    assert list(morph.relax_feats('', 'NOUN')) == []
