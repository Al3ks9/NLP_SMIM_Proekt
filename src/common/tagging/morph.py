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
    """Inverse of parse_feats. Sorts keys alphabetically -- which does NOT
    match classla's own ordering (e.g. classla emits Number before NumType in
    193 rows of pos_tagged.csv; alphabetically NumType sorts first). Harmless
    today because render_feats output only ever goes back through parse_feats
    for containment matching in _match, never used as a lookup key against
    corpus-derived feats strings -- but do not start relying on it as one.
    """
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
    entries on average, so the scan is cheap. Ties break on how many matching
    feats bundles produced the same surface form (not corpus token frequency,
    which this function has no access to) then alphabetically, so the result
    is deterministic across runs.
    """
    wanted = parse_feats(constraint).items()
    hits = [surface for key, surface in forms.items()
            if parse_feats(key).items() >= wanted]
    if not hits:
        return None
    counts = Counter(hits)
    return max(sorted(set(hits)), key=lambda f: counts[f])


def _nearest(forms, pos, source_feats):
    """Most common attested form, constrained to agree in gender.

    relax_feats refuses to drop Gender on ADJ and VERB because it is agreement
    there, not a property of the lemma. Picking the most common attested form
    across every stored bundle threw that protection away one tier below, which
    is the 'тивка гроб' error the relaxation order exists to prevent: measured
    over Конески's top-50 ADJ lemmas against real corpus bundles, this tier
    fired on 45.3% of slots and returned the wrong gender in 30.1% of gendered
    ones.

    On NOUN, gender is lexical -- жена is feminine in every form, and the
    candidate lemma brings its own gender rather than agreeing with anything --
    so no constraint applies and the tier behaves as it always did.

    Returns None rather than a wrong-gender form, leaving surface_form to
    report 'unresolved'. The bare lemma is honest about not having been
    inflected and is audited as such; a masculine form dropped into a feminine
    slot silently reads as a successful realisation.
    """
    if not forms:
        return None
    gender = parse_feats(source_feats).get('Gender')
    if gender and pos not in GENDER_DROPPABLE_POS:
        forms = {key: surface for key, surface in forms.items()
                 if parse_feats(key).get('Gender') == gender}
        if not forms:
            return None
    counts = Counter(forms.values())
    return max(sorted(set(forms.values())), key=lambda f: counts[f])


def surface_form(target_author, candidate_lemma, pos, source_feats, morph_lookup):
    """Realise a candidate lemma in the source token's grammatical form.

    Returns (surface, tier). The tier is required output, not optional logging:
    it says how much of the inflection the model still has to do, and it is what
    a batch run is audited on.

      1. by_author[target][lemma][pos][feats]   -> 'exact'
      2. pooled[lemma][pos][feats]              -> 'cross_author'
      3. relaxed constraint, target then pooled -> 'relaxed'
      4. gender-agreeing attested form, target then pooled -> 'nearest_attested'
      5. the lemma unchanged                    -> 'unresolved'

    Tier 2 borrows another author's spelling of an inflection. That is the
    right trade: a wrong-gender word is a visible grammatical error, a
    correctly-inflected word from a neighbouring idiolect is not. Pooling makes
    118% more forms retrievable than the per-author index alone. Tier 4 makes
    the same trade for the same reason — see _nearest, which is also where the
    gender constraint that keeps tier 4 from undoing tier 3 lives.
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

    hit = (_nearest(target_forms, pos, source_feats)
           or _nearest(pooled_forms, pos, source_feats))
    if hit:
        return hit, 'nearest_attested'

    return candidate_lemma, 'unresolved'
