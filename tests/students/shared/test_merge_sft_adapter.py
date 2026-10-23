import json

import pytest
import torch

import merge_sft_adapter as msa

TINY = 'trl-internal-testing/tiny-Qwen3ForCausalLM'


@pytest.fixture
def tiny_adapter(tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(TINY)
    peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj'],
                                            task_type='CAUSAL_LM'))
    with torch.no_grad():
        for n, p in peft.named_parameters():
            if 'lora_B' in n:
                p.fill_(0.01)   # non-zero so the merge actually changes weights
    adapter = tmp_path / 'adapter'
    peft.save_pretrained(adapter)
    AutoTokenizer.from_pretrained(TINY).save_pretrained(adapter)
    return adapter, peft


@pytest.mark.slow
def test_merge_matches_the_adapter_model_and_records_provenance(tiny_adapter, tmp_path):
    from transformers import AutoModelForCausalLM
    adapter, peft = tiny_adapter
    out = msa.merge(adapter, tmp_path / 'merged', base_model=TINY)

    info = json.loads((out / 'merge_info.json').read_text())
    assert info['base_model'] == TINY and info['adapter_sha256'] == msa.adapter_sha256(adapter)

    merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.allclose(merged(ids).logits, peft(ids).logits.float(), atol=5e-2)


@pytest.mark.slow
def test_merge_is_idempotent_and_refuses_a_different_adapter(tiny_adapter, tmp_path):
    adapter, _ = tiny_adapter
    out = msa.merge(adapter, tmp_path / 'merged', base_model=TINY)
    assert msa.merge(adapter, out, base_model=TINY) == out   # same adapter: no-op
    info = json.loads((out / 'merge_info.json').read_text())
    info['adapter_sha256'] = 'something-else'
    (out / 'merge_info.json').write_text(json.dumps(info))
    with pytest.raises(SystemExit):
        msa.merge(adapter, out, base_model=TINY)


# ── --family ─────────────────────────────────────────────────────────────────────

import model_families as mf


def test_family_mismatch_is_refused(tmp_path):                        # Review Focus 3
    (tmp_path / 'merge_info.json').write_text('{"family": "qwen3"}')
    with pytest.raises(ValueError, match='--family gemma4 .* qwen3'):
        mf.assert_family_matches(mf.get_family('gemma4'), tmp_path)


def test_family_mismatch_read_from_an_adapter_config(tmp_path):
    (tmp_path / 'adapter_config.json').write_text(
        '{"base_model_name_or_path": "/workspace/models/gemma4-e4b-sft-merged"}')
    with pytest.raises(ValueError, match='--family qwen3 .* gemma4'):
        mf.assert_family_matches(mf.get_family('qwen3'), tmp_path)
    mf.assert_family_matches(mf.get_family('gemma4'), tmp_path)


def test_unknown_provenance_is_not_an_error(tmp_path):
    # pre-family merges and hub ids we don't recognise must keep working
    (tmp_path / 'merge_info.json').write_text('{"adapter_dir": "x"}')
    mf.assert_family_matches(mf.get_family('gemma4'), tmp_path)


def test_merge_records_the_family(tiny_adapter, tmp_path):
    adapter, _ = tiny_adapter
    out = msa.merge(adapter, tmp_path / 'merged', base_model=TINY)
    assert json.loads((out / 'merge_info.json').read_text())['family'] == 'qwen3'


def test_forced_merge_with_the_wrong_family_deletes_nothing(tiny_adapter, tmp_path):   # review minor 2
    adapter, _ = tiny_adapter
    out = msa.merge(adapter, tmp_path / 'merged', base_model=TINY)
    (adapter / 'adapter_config.json').write_text(json.dumps(
        {**json.loads((adapter / 'adapter_config.json').read_text()),
         'base_model_name_or_path': 'google/gemma-4-E4B-it'}))
    (adapter / 'extra').write_text('changes the adapter hash')
    with pytest.raises(ValueError):
        msa.merge(adapter, out, base_model=TINY, family='qwen3', force=True)
    assert (out / 'merge_info.json').exists()
