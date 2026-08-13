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
        # Defensive guard: DROP_ORDER and NEVER_DROP are currently disjoint, so
        # this membership test never fires today. It exists to stop a future edit
        # from making a never-droppable feature droppable.
        if feature in NEVER_DROP or feature not in d:
            continue
        if feature == 'Gender' and pos not in GENDER_DROPPABLE_POS:
            continue
        if len(d) <= 1:
            return
        del d[feature]
        yield render_feats(d)


def _match(forms, constraint):
    """Best stored form whose feats are a superset of `constraint`.

    Stored keys are always full bundles from the corpus, so a relaxed
    constraint is matched by containment rather than equality. Cells hold 1.28
    entries on average, so the scan is cheap. Ties break on frequency then
    alphabetically, so the result is deterministic across runs.
    """
    wanted = parse_feats(constraint).items()
    hits = [surface for key, surface in forms.items()
            if parse_feats(key).items() >= wanted]
    if not hits:
        return None
    counts = Counter(hits)
    return max(sorted(set(hits)), key=lambda f: counts[f])


def surface_form(target_author, candidate_lemma, pos, source_feats, morph_lookup):
    """Realise a candidate lemma in the source token's grammatical form.

    Returns (surface, tier). The tier is required output, not optional logging:
    it says how much of the inflection the model still has to do, and it is what
    a batch run is audited on.

      1. by_author[target][lemma][pos][feats]   -> 'exact'
      2. pooled[lemma][pos][feats]              -> 'cross_author'
      3. relaxed constraint, target then pooled -> 'relaxed'
      4. any attested form for [target][lemma][pos] -> 'nearest_attested'
      5. the lemma unchanged                    -> 'unresolved'

    Tier 2 borrows another author's spelling of an inflection. That is the
    right trade: a wrong-gender word is a visible grammatical error, a
    correctly-inflected word from a neighbouring idiolect is not. Pooling makes
    118% more forms retrievable than the per-author index alone.
    """
    target_forms = (morph_lookup.get('by_author', {})
                    .get(target_author, {}).get(candidate_lemma, {}).get(pos, {}))
    pooled_forms = (morph_lookup.get('pooled', {})
                    .get(candidate_lemma, {}).get(pos, {}))

    if source_feats and source_feats in target_forms:
        return target_forms[source_feats], 'exact'
    if source_feats and source_feats in pooled_forms:
        return pooled_forms[source_feats], 'cross_author'

    for constraint in relax_feats(source_feats, pos):
        hit = _match(target_forms, constraint) or _match(pooled_forms, constraint)
        if hit:
            return hit, 'relaxed'

    if target_forms:
        counts = Counter(target_forms.values())
        return max(sorted(set(target_forms.values())), key=lambda f: counts[f]), 'nearest_attested'

    return candidate_lemma, 'unresolved'
