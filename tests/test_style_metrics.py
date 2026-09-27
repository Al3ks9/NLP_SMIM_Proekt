import style_metrics as sm

SRC = 'сонце грее над темната гора\nптица пее во зелената шума'


def test_copy_novelty_verbatim_copy_is_zero():
    assert sm.copy_novelty(SRC, SRC) == 0.0


def test_copy_novelty_one_word_changed_per_line_still_counts_as_copy():
    near = 'сонце грее над темната ноќ\nптица пее во зелената ноќ'
    assert sm.copy_novelty(SRC, near) == 0.0


def test_copy_novelty_new_lines_are_novel():
    assert sm.copy_novelty(SRC, 'ветер носи лисје низ полето\nмајка чека пред портата') == 1.0


def test_copy_novelty_short_lines_need_an_exact_match():
    assert sm.copy_novelty('сонце грее', 'сонце гори') == 1.0
    assert sm.copy_novelty('сонце грее', 'Сонце, грее!') == 0.0


def test_copy_novelty_empty_generation_is_zero():
    assert sm.copy_novelty(SRC, '') == 0.0


def test_no_chat_markup_allows_poetic_dashes_and_numbers():
    poem = 'Но, пишувањето е страст -\n- ако не се разголиш нема ни стих\n1. песна за мајка'
    assert sm.language_metrics(poem)['no_chat_markup'] == 1.0
    assert sm.language_metrics(poem)['no_markup'] == 0.0      # reported metric unchanged


def test_no_chat_markup_catches_chat_scaffolding():
    for text in ('**Песна**\nсонце грее', '### Песна\nсонце грее', '```\nсонце\n```'):
        assert sm.language_metrics(text)['no_chat_markup'] == 0.0
