"""Morphological feats: parsing, relaxation, and surface realisation.

Kept separate from candidate_selection.py, which is already large and owns
candidate *ranking* rather than candidate *inflection*.
"""

from collections import Counter

# Drop order, most-droppable first, measured over pos_tagged.csv. Polarity is
# 100% present on verbs with a single value and so carries zero information;
# Case is 99.8% Nom; Mood 96.8% Ind; Degree 96.6%/95.1% Pos on ADJ/ADV.
#
# Gender is droppable for NOUN only. Macedonian nouns carry inherent lexical
# gender — град is masculine in every form — so once the lemma is fixed Gender
# is redundant. On ADJ and VERB it is agreement, and dropping it yields the
# 'тивка гроб' class of error.
DROP_ORDER = ['Polarity', 'Case', 'Mood', 'Degree', 'Gender',
              'Aspect', 'Person', 'Tense']

GENDER_DROPPABLE_POS = {'NOUN'}

# Number/Definite are the surface distinction itself. VerbForm, Poss, NumType,
# PronType and AdvType each have a single value but appear on only a fraction
# of their POS — their *presence* is the signal, so single-valuedness must not
# be mistaken for uninformativeness.
NEVER_DROP = {'Number', 'Definite', 'VerbForm', 'Poss',
              'NumType', 'PronType', 'AdvType'}


def parse_feats(feats):
    """'Case=Nom|Number=Sing' -> {'Case': 'Nom', 'Number': 'Sing'}."""
    if not feats:
        return {}
    return dict(kv.split('=', 1) for kv in feats.split('|') if '=' in kv)


def render_feats(d):
    """Inverse of parse_feats. Alphabetical, matching classla's own ordering."""
    return '|'.join(f'{k}={v}' for k, v in sorted(d.items()))


def relax_feats(feats, pos):
    """Yield progressively weaker feats *constraints*, widest match last.

    Each step drops one more feature. Yields constraint sets rather than keys:
    stored keys are always full bundles, so a relaxed bundle is matched by
    superset containment (see surface_form), never by equality.

    Never empties the bundle — '' is reserved for genuinely featless tokens.
    """
    d = parse_feats(feats)
    for feature in DROP_ORDER:
        if feature in NEVER_DROP or feature not in d:
            continue
        if feature == 'Gender' and pos not in GENDER_DROPPABLE_POS:
            continue
        if len(d) <= 1:
            return
        del d[feature]
        yield render_feats(d)
