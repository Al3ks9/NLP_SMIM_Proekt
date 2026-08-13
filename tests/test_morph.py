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


def test_drop_order_and_never_drop_are_disjoint():
    # Defensive guard invariant: features that are safe-to-drop must never
    # conflict with features that must be preserved.
    assert set(morph.DROP_ORDER) & morph.NEVER_DROP == set()


def test_never_drop_features_never_disappear():
    # NEVER_DROP features in the initial bundle must never disappear from
    # any relaxed output. This verifies the defensive guard works.
    initial = 'Definite=Def|Gender=Fem|Number=Sing'
    for step in morph.relax_feats(initial, 'NOUN'):
        parsed = morph.parse_feats(step)
        # Number and Definite started in the bundle and must still be present
        assert 'Number' in parsed and 'Definite' in parsed


FEM_SG_DEF = 'Case=Nom|Definite=Def|Gender=Fem|Number=Sing'
NEUT_PL_DEF = 'Case=Nom|Definite=Def|Gender=Neut|Number=Plur'

LOOKUP = {
    'by_author': {
        'A': {'жена': {'NOUN': {FEM_SG_DEF: 'жената'}}},
        'B': {'жена': {'NOUN': {NEUT_PL_DEF: 'женине'}}},
    },
    'pooled': {'жена': {'NOUN': {FEM_SG_DEF: 'жената', NEUT_PL_DEF: 'женине'}}},
}


def test_exact_tier_hits_the_target_author():
    assert morph.surface_form('A', 'жена', 'NOUN', FEM_SG_DEF, LOOKUP) == ('жената', 'exact')


def test_the_bug_this_fixes_ata_no_longer_conflates_two_readings():
    # Both surface forms end in 'ата'/'ине' but differ in gender+number. Under
    # the old 3-char suffix key these collided; NOUN+ата covered 2,685 tokens
    # across 8 feats bundles.
    fem, _ = morph.surface_form('A', 'жена', 'NOUN', FEM_SG_DEF, LOOKUP)
    neut, _ = morph.surface_form('A', 'жена', 'NOUN', NEUT_PL_DEF, LOOKUP)
    assert fem != neut


def test_cross_author_tier_borrows_from_the_pooled_index():
    form, tier = morph.surface_form('A', 'жена', 'NOUN', NEUT_PL_DEF, LOOKUP)
    assert (form, tier) == ('женине', 'cross_author')


def test_relaxed_tier_matches_a_stored_superset_key():
    # Query carries Polarity; the stored key does not differ on anything else.
    lookup = {
        'by_author': {'A': {'бара': {'VERB': {
            'Aspect=Prog|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres': 'бара'}}}},
        'pooled': {},
    }
    # Same bundle but Mood=Sub, which no stored key has — relaxation drops Mood.
    query = 'Aspect=Prog|Mood=Sub|Number=Sing|Person=3|Polarity=Pos|Tense=Pres'
    form, tier = morph.surface_form('A', 'бара', 'VERB', query, lookup)
    assert (form, tier) == ('бара', 'relaxed')


def test_relaxed_never_reports_exact():
    lookup = {
        'by_author': {'A': {'бара': {'VERB': {
            'Aspect=Prog|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres': 'бара'}}}},
        'pooled': {},
    }
    query = 'Aspect=Prog|Mood=Sub|Number=Sing|Person=3|Polarity=Pos|Tense=Pres'
    assert morph.surface_form('A', 'бара', 'VERB', query, lookup)[1] != 'exact'


def test_nearest_attested_when_nothing_matches():
    lookup = {'by_author': {'A': {'жена': {'NOUN': {FEM_SG_DEF: 'жената'}}}}, 'pooled': {}}
    other = 'Case=Nom|Definite=Ind|Gender=Masc|Number=Plur'
    form, tier = morph.surface_form('A', 'жена', 'NOUN', other, lookup)
    assert (form, tier) == ('жената', 'nearest_attested')


def test_unresolved_returns_the_bare_lemma():
    empty = {'by_author': {}, 'pooled': {}}
    assert morph.surface_form('A', 'жена', 'NOUN', FEM_SG_DEF, empty) == ('жена', 'unresolved')


def test_featless_token_degrades_to_nearest_attested_not_an_exception():
    lookup = {'by_author': {'A': {'жена': {'NOUN': {FEM_SG_DEF: 'жената'}}}}, 'pooled': {}}
    assert morph.surface_form('A', 'жена', 'NOUN', '', lookup) == ('жената', 'nearest_attested')


def test_target_author_is_preferred_over_the_pooled_index():
    lookup = {
        'by_author': {'A': {'жена': {'NOUN': {FEM_SG_DEF: 'жената'}}}},
        'pooled': {'жена': {'NOUN': {FEM_SG_DEF: 'женава'}}},
    }
    assert morph.surface_form('A', 'жена', 'NOUN', FEM_SG_DEF, lookup)[0] == 'жената'


import json
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / 'models'


def test_morph_lookup_has_both_indices():
    with open(MODELS / 'morph_lookup.json', encoding='utf-8') as f:
        lookup = json.load(f)
    assert set(lookup) == {'by_author', 'pooled'}


def test_morph_lookup_keys_are_feats_not_suffixes():
    with open(MODELS / 'morph_lookup.json', encoding='utf-8') as f:
        lookup = json.load(f)
    keys = [k
            for lemmas in lookup['by_author'].values()
            for pos_map in lemmas.values()
            for forms in pos_map.values()
            for k in forms]
    assert keys, 'lookup is empty'
    # A feats key contains '='; a 3-char suffix never does.
    assert sum('=' in k for k in keys) / len(keys) > 0.9
