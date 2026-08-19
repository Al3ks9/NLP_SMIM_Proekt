"""LLM-based generative style transfer (steps 1, 5-9). LLM calls are
monkeypatched to canned responses — no dependency on a live Ollama server,
matching the point of llm_client.call() being swappable."""

import json

import numpy as np
import pytest

import exemplar_selection
import llm_style_transfer as lst
import poem_tfidf
import style_narrator

KONESKI = 'Блаже Конески'
RACIN = 'Кочо Рацин'


# ── step 6: structural targets ──────────────────────────────────────────────────

def test_structural_targets_pulls_the_real_profile_row():
    targets = lst.structural_targets(KONESKI)
    assert targets['avg_lines_per_poem'] > 0
    assert targets['avg_tokens_per_line'] > 0
    assert targets['avg_stanzas_per_poem'] > 0
    assert targets['avg_lines_per_stanza'] > 0


def test_structural_targets_raises_for_an_unknown_author():
    with pytest.raises(KeyError):
        lst.structural_targets('Не Постои Автор')


def test_format_structural_instruction_mentions_all_four_numbers():
    targets = {'avg_lines_per_poem': 16.0, 'avg_tokens_per_line': 5.0,
              'avg_stanzas_per_poem': 2.0, 'avg_lines_per_stanza': 8.0}
    text = lst.format_structural_instruction(targets)
    assert '16' in text and '5' in text and '2' in text and '8' in text


# ── step 8: structural validation ───────────────────────────────────────────────

def test_validate_structure_counts_lines_and_stanzas():
    poem = 'ред еден два\nред три четири\n\nред пет шест'
    fit = lst.validate_structure(poem, {'avg_lines_per_poem': 3, 'avg_tokens_per_line': 3,
                                        'avg_stanzas_per_poem': 2, 'avg_lines_per_stanza': 1.5})
    assert fit['actual_lines'] == 3
    assert fit['actual_stanzas'] == 2
    assert fit['actual_tokens_per_line'] == pytest.approx(3.0)


def test_validate_structure_reports_zero_delta_on_exact_match():
    poem = 'а б\nв г'
    targets = {'avg_lines_per_poem': 2, 'avg_tokens_per_line': 2,
              'avg_stanzas_per_poem': 1, 'avg_lines_per_stanza': 2}
    fit = lst.validate_structure(poem, targets)
    assert fit['delta_lines'] == 0
    assert fit['delta_stanzas'] == 0


def test_validate_structure_reports_nonzero_delta_on_mismatch():
    poem = 'а б в'  # 1 line, target wants 5
    targets = {'avg_lines_per_poem': 5, 'avg_tokens_per_line': 3,
              'avg_stanzas_per_poem': 1, 'avg_lines_per_stanza': 5}
    fit = lst.validate_structure(poem, targets)
    assert fit['delta_lines'] == -4


# ── step 1: summary generation + caching ────────────────────────────────────────

def test_summarize_poem_calls_the_llm_and_returns_its_response(monkeypatch, tmp_path):
    monkeypatch.setattr(lst, 'CACHE_PATH', tmp_path / 'cache.json')
    monkeypatch.setattr(lst, '_summary_cache', None)

    def fake_call(model, prompt, backend, **kw):
        return 'Резиме на песната.'

    monkeypatch.setattr(lst, 'call', fake_call)
    summary = lst.summarize_poem('некаков текст', author='А', title='Т')
    assert summary == 'Резиме на песната.'


def test_summarize_poem_is_cached_and_does_not_call_the_llm_twice(monkeypatch, tmp_path):
    monkeypatch.setattr(lst, 'CACHE_PATH', tmp_path / 'cache.json')
    monkeypatch.setattr(lst, '_summary_cache', None)

    calls = []

    def fake_call(model, prompt, backend, **kw):
        calls.append(prompt)
        return 'Резиме.'

    monkeypatch.setattr(lst, 'call', fake_call)
    lst.summarize_poem('истиот текст', author='А', title='Т')
    lst.summarize_poem('истиот текст', author='А', title='Т')
    assert len(calls) == 1


def test_summarize_poem_cache_persists_to_disk(monkeypatch, tmp_path):
    cache_file = tmp_path / 'cache.json'
    monkeypatch.setattr(lst, 'CACHE_PATH', cache_file)
    monkeypatch.setattr(lst, '_summary_cache', None)
    monkeypatch.setattr(lst, 'call', lambda model, prompt, backend, **kw: 'Резиме.')

    lst.summarize_poem('текст', author='А', title='Т')
    assert cache_file.exists()
    saved = json.loads(cache_file.read_text(encoding='utf-8'))
    assert list(saved.values())[0]['summary'] == 'Резиме.'


# ── step 5: vocabulary palette ──────────────────────────────────────────────────

def test_surface_to_lemma_picks_the_most_common_association():
    pos_rows = [
        {'word': 'сокола', 'lemma': 'сокол'},
        {'word': 'сокола', 'lemma': 'сокол'},
        {'word': 'сокола', 'lemma': 'соколче'},  # minority mistag
    ]
    mapping = lst.surface_to_lemma_map(pos_rows)
    assert mapping['сокола'] == 'сокол'


def test_vocabulary_palette_ranks_semantically_closer_words_first(monkeypatch):
    # 'сокол' should outrank 'камен' when the content keyword is 'птица'
    # (bird) and 'сокол' (falcon) sits close to it in embedding space.
    monkeypatch.setattr(lst, 'load_tfidf',
                        lambda: {'А': {'сокол': 0.5, 'камен': 0.9}})
    monkeypatch.setattr(lst, 'load_word_embeddings', lambda: {
        'птица': np.array([1.0, 0.0]),
        'сокол': np.array([0.9, 0.1]),
        'камен': np.array([0.0, 1.0]),
    })
    monkeypatch.setattr(lst, 'surface_to_lemma_map', lambda pos_rows: {
        'сокол': 'сокол', 'камен': 'камен',
    })
    monkeypatch.setattr(lst, 'load_pos_rows', lambda: [])

    palette = lst.vocabulary_palette('А', content_keywords=['птица'], top_k=2)
    assert palette[0] == 'сокол'


def test_vocabulary_palette_falls_back_to_tfidf_order_when_keywords_unembeddable(monkeypatch):
    monkeypatch.setattr(lst, 'load_tfidf', lambda: {'А': {'алфа': 0.9, 'бета': 0.5}})
    monkeypatch.setattr(lst, 'load_word_embeddings', lambda: {})
    monkeypatch.setattr(lst, 'surface_to_lemma_map', lambda pos_rows: {})
    monkeypatch.setattr(lst, 'load_pos_rows', lambda: [])

    palette = lst.vocabulary_palette('А', content_keywords=['непознат'], top_k=2)
    assert palette == ['алфа', 'бета']


# ── step 7: prompt assembly ──────────────────────────────────────────────────────

def _stub_components(monkeypatch):
    monkeypatch.setattr(lst, 'summarize_poem', lambda text, **kw: 'Песна за пролетта.')
    monkeypatch.setattr(poem_tfidf, 'top_content_words',
                        lambda poem_id, n=10: ['пролет', 'цвет'])
    monkeypatch.setattr(style_narrator, 'describe_style',
                        lambda author: f'{author} пишува кратки стихови.')
    monkeypatch.setattr(lst, 'vocabulary_palette',
                        lambda author, content_keywords, top_k=8: ['сон', 'изгрев'])
    monkeypatch.setattr(exemplar_selection, 'select_exemplars',
                        lambda author, **kw: [
                            {'line': 'пример стих еден', 'author': author,
                             'song_title': 'ТАЈНА ПЕСНА', 'cluster': 0},
                        ])
    monkeypatch.setattr(lst, 'structural_targets',
                        lambda author: {'avg_lines_per_poem': 12.0, 'avg_tokens_per_line': 5.0,
                                        'avg_stanzas_per_poem': 3.0, 'avg_lines_per_stanza': 4.0})


def test_assemble_prompt_includes_every_component(monkeypatch):
    _stub_components(monkeypatch)
    result = lst.assemble_prompt('изворен текст', '0', 'Извор Автор', 'Извор Наслов', 'Целен Автор')

    assert 'Песна за пролетта.' in result['prompt']
    assert 'пролет' in result['prompt'] and 'цвет' in result['prompt']
    assert 'пишува кратки стихови' in result['prompt']
    assert 'сон' in result['prompt'] and 'изгрев' in result['prompt']
    assert 'пример стих еден' in result['prompt']
    assert 'Целен Автор' in result['prompt']


def test_assemble_prompt_strips_poem_attribution_from_exemplar_lines(monkeypatch):
    _stub_components(monkeypatch)
    result = lst.assemble_prompt('изворен текст', '0', 'Извор Автор', 'Извор Наслов', 'Целен Автор')
    assert 'ТАЈНА ПЕСНА' not in result['prompt']


def test_assemble_prompt_returns_components_for_logging(monkeypatch):
    _stub_components(monkeypatch)
    result = lst.assemble_prompt('изворен текст', '0', 'Извор Автор', 'Извор Наслов', 'Целен Автор')
    assert result['summary'] == 'Песна за пролетта.'
    assert result['content_keywords'] == ['пролет', 'цвет']
    assert result['vocab_palette'] == ['сон', 'изгрев']
    assert len(result['exemplars']) == 1


# ── step 9: run_transfer + logging ──────────────────────────────────────────────

# ── load_poem_text / load_poem_text_by_id / resolve_poem_id ────────────────────

def test_load_poem_text_returns_the_poem_for_an_unambiguous_title():
    text = lst.load_poem_text(KONESKI, 'Разделба')
    assert isinstance(text, str) and text.strip()


def test_load_poem_text_raises_on_a_real_duplicated_title():
    # Verified in stripped_songs.csv: 4 distinct poems by Конески are titled
    # 'ПЕСНА'. Silently returning one would be a plausible-looking wrong poem.
    with pytest.raises(ValueError):
        lst.load_poem_text(KONESKI, 'ПЕСНА')


def test_resolve_poem_id_error_lists_the_candidate_ids():
    with pytest.raises(ValueError, match=r'--source-poem-id'):
        lst.resolve_poem_id(KONESKI, 'ПЕСНА')


def test_load_poem_text_by_id_resolves_one_of_the_duplicated_pesna_poems():
    import csv
    with open(lst.DATA / 'stripped_songs.csv', encoding='utf-8') as f:
        pesna_ids = [r['poem_id'] for r in csv.DictReader(f)
                    if r['author'] == KONESKI and r['song_title'] == 'ПЕСНА']
    assert len(pesna_ids) > 1, 'fixture assumption: Конески has >1 poem titled ПЕСНА'

    text = lst.load_poem_text_by_id(pesna_ids[0])
    assert isinstance(text, str) and text.strip()


def test_load_poem_text_by_id_raises_for_an_unknown_id():
    with pytest.raises(KeyError):
        lst.load_poem_text_by_id('999999')


def test_log_run_writes_a_json_file_with_the_full_record(tmp_path, monkeypatch):
    monkeypatch.setattr(lst, 'LOG_DIR', tmp_path)
    record = {
        'prompt': 'п', 'generated_poem': 'г', 'structural_fit': {'delta_lines': 0},
        'source_author': 'А', 'source_title': 'Т', 'target_author': 'Ц',
        'model': 'м', 'backend': 'ollama', 'timestamp': '2026-01-01T00:00:00+00:00',
    }
    path = lst.log_run(record)
    assert path.exists()
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved == record


def test_run_transfer_orchestrates_the_full_chain(monkeypatch, tmp_path):
    monkeypatch.setattr(lst, 'LOG_DIR', tmp_path)
    monkeypatch.setattr(lst, 'load_poem_text_by_id', lambda poem_id: 'изворен текст')
    monkeypatch.setattr(lst, '_load_stripped_songs_indexed', lambda: {
        '7': {'author': 'Извор Автор', 'song_title': 'Извор Наслов', 'song_text': 'изворен текст'},
    })
    monkeypatch.setattr(lst, 'assemble_prompt', lambda *a, **kw: {
        'prompt': 'составен промпт', 'summary': 'резиме', 'content_keywords': ['зб'],
        'style_text': 'стил', 'vocab_palette': ['palette'], 'exemplars': [],
        'structural_targets': {'avg_lines_per_poem': 4, 'avg_tokens_per_line': 3,
                               'avg_stanzas_per_poem': 1, 'avg_lines_per_stanza': 4},
    })
    monkeypatch.setattr(lst, 'generate_poem', lambda prompt, **kw: 'нова\nпесна')

    result = lst.run_transfer('Извор Автор', 'Извор Наслов', 'Целен Автор', source_poem_id='7')

    assert result['generated_poem'] == 'нова\nпесна'
    assert result['structural_fit']['actual_lines'] == 2
    assert result['source_author'] == 'Извор Автор'
    assert result['source_poem_id'] == '7'
    assert result['target_author'] == 'Целен Автор'
    logged = list(tmp_path.glob('*.json'))
    assert len(logged) == 1
