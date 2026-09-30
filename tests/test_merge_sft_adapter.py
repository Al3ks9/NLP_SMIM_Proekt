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
