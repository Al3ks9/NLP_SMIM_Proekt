import grpo_data as gd
from make_splits import load_splits


class FakeTokenizer:
    """Renders to the user text; 'token' count = whitespace words."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False and add_generation_prompt is True
        return messages[1]['content']

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': text.split()}


def test_prompt_rows_are_train_only_and_never_self_transfer():
    splits = load_splits()
    rows = gd.build_prompt_rows()
    assert rows
    assert all(splits[r['source_poem_id']] == 'train' for r in rows)
    assert all(r['source_author'] != r['target_author'] for r in rows)


def test_prompt_rows_are_seeded():
    assert gd.build_prompt_rows(seed=1)[:20] == gd.build_prompt_rows(seed=1)[:20]
    assert gd.build_prompt_rows(seed=1)[:20] != gd.build_prompt_rows(seed=2)[:20]


def test_dataset_columns_and_cap():
    ds, stats = gd.build_grpo_dataset(FakeTokenizer(), num_prompts=5)
    assert len(ds) == 5 and stats['kept'] == 5
    assert set(ds.column_names) == {'prompt', 'source_poem_id', 'source_author',
                                    'source_text', 'target_author'}
    assert ds[0]['source_text'].strip() in ds[0]['prompt']


# Review Focus 3
def test_too_long_prompts_are_dropped_not_truncated():
    ds, stats = gd.build_grpo_dataset(FakeTokenizer(), max_prompt_length=60)
    assert stats['dropped_too_long'] > 0
    for row in ds:
        assert len(row['prompt'].split()) <= 60
        assert row['source_text'].strip() in row['prompt']   # never cut
