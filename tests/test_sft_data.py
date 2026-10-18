"""Qwen SFT pipeline: shared prompt format, target-author selection, and
synthetic-dataset loading. A fake tokenizer stands in for a real Qwen3
tokenizer -- only apply_chat_template's call signature matters here."""

import csv

import pytest

import sft_data


class FakeTokenizer:
    """Mimics enough of a chat tokenizer to exercise render_chat/
    render_inference_prompt without downloading Qwen3."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            enable_thinking):
        assert tokenize is False
        assert enable_thinking is False
        rendered = ''.join(
            f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in messages)
        if add_generation_prompt:
            rendered += '<|im_start|>assistant\n'
        return rendered


# ── build_messages ───────────────────────────────────────────────────────────────

def test_build_messages_without_generated_poem_has_system_and_user_only():
    messages = sft_data.build_messages('извор песна', 'Целен Автор')
    assert [m['role'] for m in messages] == ['system', 'user']
    assert 'извор песна' in messages[1]['content']
    assert 'Целен Автор' in messages[1]['content']


def test_build_messages_with_generated_poem_appends_assistant_turn():
    messages = sft_data.build_messages('извор песна', 'Целен Автор', 'нова песна')
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
    assert messages[2]['content'] == 'нова песна'


def test_build_messages_strips_whitespace():
    messages = sft_data.build_messages('  извор  \n', 'А', '  нова  \n')
    assert 'Source poem:\nизвор\n\nTarget author: А' == messages[1]['content']
    assert messages[2]['content'] == 'нова'


# ── render_chat / render_inference_prompt ───────────────────────────────────────

def test_render_chat_disables_thinking_and_does_not_tokenize():
    tok = FakeTokenizer()
    text = sft_data.render_chat(tok, [{'role': 'user', 'content': 'x'}])
    assert '<|im_start|>user' in text


def test_render_inference_prompt_opens_the_assistant_turn_without_content():
    tok = FakeTokenizer()
    text = sft_data.render_inference_prompt(tok, 'извор', 'Автор')
    assert text.endswith('<|im_start|>assistant\n')


# ── select_target_authors ────────────────────────────────────────────────────────

def _stub_profiles(monkeypatch, rows):
    monkeypatch.setattr(sft_data, 'load_profile_rows', lambda: rows)


def test_select_target_authors_picks_the_most_prolific(monkeypatch):
    _stub_profiles(monkeypatch, [
        {'author': 'А', 'num_poems': '10'},
        {'author': 'Б', 'num_poems': '50'},
        {'author': 'В', 'num_poems': '30'},
    ])
    assert sft_data.select_target_authors(2) == ['Б', 'В']


def test_select_target_authors_breaks_ties_by_name(monkeypatch):
    _stub_profiles(monkeypatch, [
        {'author': 'В', 'num_poems': '10'},
        {'author': 'Б', 'num_poems': '10'},
    ])
    assert sft_data.select_target_authors(1) == ['Б']


def test_select_target_authors_raises_when_n_exceeds_available(monkeypatch):
    _stub_profiles(monkeypatch, [{'author': 'А', 'num_poems': '10'}])
    with pytest.raises(ValueError):
        sft_data.select_target_authors(2)


def test_select_target_authors_explicit_list_is_used_as_is(monkeypatch):
    _stub_profiles(monkeypatch, [{'author': 'А', 'num_poems': '10'},
                                 {'author': 'Б', 'num_poems': '50'}])
    assert sft_data.select_target_authors(1, explicit=['А']) == ['А']


def test_select_target_authors_explicit_list_rejects_unknown_authors(monkeypatch):
    _stub_profiles(monkeypatch, [{'author': 'А', 'num_poems': '10'}])
    with pytest.raises(KeyError):
        sft_data.select_target_authors(1, explicit=['Непостоечки'])


# ── load_synthetic_rows ──────────────────────────────────────────────────────────

_SYNTHETIC_HEADER = ['example_id', 'source_poem_id', 'source_author', 'source_title',
                     'target_author', 'split', 'generated_poem']


def _write_synthetic_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_SYNTHETIC_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_load_synthetic_rows_filters_by_split(tmp_path):
    path = tmp_path / 'synthetic_dataset.csv'
    _write_synthetic_csv(path, [
        {'example_id': '1', 'source_poem_id': '0', 'source_author': 'А',
         'source_title': 'Т', 'target_author': 'Ц', 'split': 'train',
         'generated_poem': 'п1'},
        {'example_id': '2', 'source_poem_id': '1', 'source_author': 'А',
         'source_title': 'Т2', 'target_author': 'Ц', 'split': 'val',
         'generated_poem': 'п2'},
    ])
    train_rows = sft_data.load_synthetic_rows(split='train', path=path)
    assert len(train_rows) == 1
    assert train_rows[0]['example_id'] == '1'


def test_load_synthetic_rows_rejects_an_unknown_split(tmp_path):
    path = tmp_path / 'synthetic_dataset.csv'
    _write_synthetic_csv(path, [])
    with pytest.raises(ValueError):
        sft_data.load_synthetic_rows(split='not-a-split', path=path)


# ── build_synthetic_messages ───────────────────────────────────────────────────────

def test_build_synthetic_messages_looks_up_source_text_by_poem_id(monkeypatch):
    monkeypatch.setattr(sft_data, 'load_stripped_songs', lambda: [
        {'poem_id': '7', 'author': 'Извор Автор', 'song_title': 'Т', 'song_text': 'изворна содржина'},
    ])
    rows = [{'source_poem_id': '7', 'target_author': 'Целен Автор',
            'generated_poem': 'нова содржина'}]
    messages_list = sft_data.build_synthetic_messages(rows)
    assert len(messages_list) == 1
    messages = messages_list[0]
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
    assert 'изворна содржина' in messages[1]['content']
    assert 'Целен Автор' in messages[1]['content']
    assert messages[2]['content'] == 'нова содржина'


def test_build_synthetic_messages_raises_for_an_unknown_source_poem_id(monkeypatch):
    monkeypatch.setattr(sft_data, 'load_stripped_songs', lambda: [])
    rows = [{'source_poem_id': '999', 'target_author': 'А', 'generated_poem': 'п'}]
    with pytest.raises(KeyError):
        sft_data.build_synthetic_messages(rows)


# ── prompt-completion format (Gemma students: no TRL training template) ─────────

TINY = 'trl-internal-testing/tiny-Qwen3ForCausalLM'


@pytest.fixture(scope='module')
def tiny_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TINY)


def test_prompt_is_a_strict_prefix_of_the_full_chat(tiny_tok):
    pc = sft_data.build_prompt_completion(tiny_tok, 'извор\nред', 'Блаже Конески', 'нова песна')
    full = sft_data.render_chat(tiny_tok, sft_data.build_messages('извор\nред', 'Блаже Конески',
                                                                  'нова песна'))
    assert full.startswith(pc['prompt']) and len(pc['prompt']) < len(full)


def test_completion_ends_with_the_templates_end_of_turn(tiny_tok):       # Review Focus 2
    pc = sft_data.build_prompt_completion(tiny_tok, 'извор', 'Блаже Конески', 'нова песна')
    assert sft_data.end_of_turn(tiny_tok) == '<|im_end|>'
    assert pc['completion'] == 'нова песна<|im_end|>'


def _trainable_ids(tokenizer, fmt, tmp_path):
    """Token ids that carry loss after TRL's own dataset preparation."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    source, target, poem = 'извор\nред', 'Блаже Конески', 'нова песна\nвтор стих'
    if fmt == 'messages':
        ds = Dataset.from_dict({'messages': [sft_data.build_messages(source, target, poem)]})
        flags = {'assistant_only_loss': True}
    else:
        ds = Dataset.from_list([sft_data.build_prompt_completion(tokenizer, source, target, poem)])
        flags = {'completion_only_loss': True}
    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(TINY), processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(output_dir=str(tmp_path / fmt), report_to='none', use_cpu=True,
                       bf16=False, max_length=1024, **flags))
    row = trainer.train_dataset[0]
    # TRL 1.13 folds assistant_masks / completion_mask into labels (-100 = no loss)
    return [t for t, label in zip(row['input_ids'], row['labels']) if label != -100]


def test_both_formats_train_on_the_same_poem_and_end_of_turn(tiny_tok, tmp_path):
    # assistant_only_loss on messages vs completion_only_loss on prompt +
    # completion. The poem and its end-of-turn token carry loss in both. The
    # only difference is Qwen3's training template also putting loss on its
    # constant empty-thinking scaffold and the newline after <|im_end|> --
    # fixed tokens, not poem content, and the scaffold is already part of the
    # inference prompt. Pinned exactly, so any other drift fails here.
    a = tiny_tok.decode(_trainable_ids(tiny_tok, 'messages', tmp_path))
    b = tiny_tok.decode(_trainable_ids(tiny_tok, 'prompt_completion', tmp_path))
    assert b == 'нова песна\nвтор стих<|im_end|>'
    assert a == '<think>\n\n</think>\n\n' + b + '\n'


def test_messages_format_is_still_the_default(monkeypatch):
    monkeypatch.setattr(sft_data, 'load_stripped_songs',
                        lambda: [{'poem_id': '1', 'song_text': 'извор'}])
    rows = [{'source_poem_id': '1', 'target_author': 'Блаже Конески', 'generated_poem': 'нова'}]
    assert sft_data.build_hf_dataset(rows).column_names == ['messages']


def test_prompt_completion_dataset_columns(tiny_tok, monkeypatch):
    monkeypatch.setattr(sft_data, 'load_stripped_songs',
                        lambda: [{'poem_id': '1', 'song_text': 'извор'}])
    rows = [{'source_poem_id': '1', 'target_author': 'Блаже Конески', 'generated_poem': 'нова'}]
    ds = sft_data.build_hf_dataset(rows, fmt='prompt_completion', tokenizer=tiny_tok)
    assert ds.column_names == ['prompt', 'completion']
    assert ds[0]['completion'] == 'нова<|im_end|>'
