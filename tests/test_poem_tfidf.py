"""Per-poem TF-IDF (step 2): each poem is a document against the rest of the
corpus, over pos_tagged.csv lemmas — distinct from tfidf_authors.py's
author-level, surface-word computation."""

import poem_tfidf as pt


def test_word_unique_to_one_poem_ranks_above_a_word_common_to_all():
    documents = {
        '0': ['нива', 'сокол', 'сокол', 'нива'],
        '1': ['нива', 'река', 'нива'],
        '2': ['нива', 'облак'],
    }
    scores = pt.compute_tfidf(documents)
    top_word_a = scores['0'][0][0]
    assert top_word_a == 'сокол'


def test_scores_within_a_document_are_sorted_descending():
    documents = {
        '0': ['алфа', 'алфа', 'бета', 'гама'],
        '1': ['делта', 'делта', 'делта'],
    }
    scores = pt.compute_tfidf(documents)
    doc_scores = [s for _, s in scores['0']]
    assert doc_scores == sorted(doc_scores, reverse=True)


def test_a_word_absent_from_a_document_does_not_appear_in_its_results():
    documents = {
        '0': ['алфа', 'бета'],
        '1': ['гама'],
    }
    scores = pt.compute_tfidf(documents)
    words_in_0 = {w for w, _ in scores['0']}
    assert 'гама' not in words_in_0


def test_build_documents_keeps_same_titled_poems_separate():
    # The whole point: two distinct poems sharing a title must not blend their
    # vocabulary into one TF-IDF document.
    pos_rows = [
        {'poem_id': '1', 'pos': 'NOUN', 'lemma': 'сокол'},
        {'poem_id': '2', 'pos': 'NOUN', 'lemma': 'камен'},
    ]
    documents = pt.build_documents(pos_rows)
    assert documents == {'1': ['сокол'], '2': ['камен']}
