"""Per-poem TF-IDF (step 2): each poem is a document against the rest of the
corpus, over pos_tagged.csv lemmas — distinct from tfidf_authors.py's
author-level, surface-word computation."""

import poem_tfidf as pt


def test_word_unique_to_one_poem_ranks_above_a_word_common_to_all():
    # 'сокол' only appears in poem A; 'нива' appears in every poem, so its idf
    # is near zero. TF-IDF should rank the distinctive word first in A.
    documents = {
        ('Автор1', 'Песна А'): ['нива', 'сокол', 'сокол', 'нива'],
        ('Автор1', 'Песна Б'): ['нива', 'река', 'нива'],
        ('Автор2', 'Песна В'): ['нива', 'облак'],
    }
    scores = pt.compute_tfidf(documents)
    top_word_a = scores[('Автор1', 'Песна А')][0][0]
    assert top_word_a == 'сокол'


def test_scores_within_a_document_are_sorted_descending():
    documents = {
        ('А', 'X'): ['алфа', 'алфа', 'бета', 'гама'],
        ('Б', 'Y'): ['делта', 'делта', 'делта'],
    }
    scores = pt.compute_tfidf(documents)
    doc_scores = [s for _, s in scores[('А', 'X')]]
    assert doc_scores == sorted(doc_scores, reverse=True)


def test_a_word_absent_from_a_document_does_not_appear_in_its_results():
    documents = {
        ('А', 'X'): ['алфа', 'бета'],
        ('Б', 'Y'): ['гама'],
    }
    scores = pt.compute_tfidf(documents)
    words_in_x = {w for w, _ in scores[('А', 'X')]}
    assert 'гама' not in words_in_x
