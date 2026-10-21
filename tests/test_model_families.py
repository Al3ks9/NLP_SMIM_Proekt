import json
from pathlib import Path

import pytest
import torch

import model_families as mf

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'gemma4_e4b_config.json'

GEMMA_REGEX = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'


def _meta_module_names_from_fixture() -> list:
    """Module names of the real E4B architecture, built on the meta device
    from its config.json -- no weights downloaded."""
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    cfg = Gemma4Config.from_dict(json.loads(FIXTURE.read_text(encoding='utf-8')))
    with torch.device('meta'):
        model = Gemma4ForConditionalGeneration(cfg)
    return [n for n, _ in model.named_modules()]


@pytest.fixture(scope='module')
def e4b_names():
    return _meta_module_names_from_fixture()


def test_qwen_resolves_to_todays_exact_values():
    f = mf.get_family('qwen3')
    assert f.hf_name == 'Qwen/Qwen3-4B'
    assert f.target_modules == ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    assert f.loss_masking == 'assistant_only' and f.generation_overrides == {}
    assert mf.default_dirs(f) == {'sft': MODELS / 'qwen3-lora-sft', 'merged': MODELS / 'qwen3-sft-merged',
                                  'grpo': MODELS / 'qwen3-grpo'}
    assert mf.base_label(f) == 'base-Qwen3-4B'


def test_gemma_family_values():
    f = mf.get_family('gemma4')
    assert f.hf_name == 'google/gemma-4-E4B-it' and f.tag == 'gemma4-e4b'
    assert f.loss_masking == 'completion_only' and f.generation_overrides == {'top_k': 20}
    assert mf.base_label(f) == 'base-gemma-4-E4B-it'


def test_unknown_family_lists_known_keys():
    with pytest.raises(ValueError, match='qwen3.*gemma4|gemma4.*qwen3'):
        mf.get_family('llama')


def test_regex_matches_only_language_model_on_real_e4b_config(e4b_names):
    counts = mf.count_target_matches(GEMMA_REGEX, e4b_names)
    assert counts == {'q_proj': 42, 'o_proj': 42, 'gate_proj': 42, 'up_proj': 42, 'down_proj': 42,
                      'k_proj': 24, 'v_proj': 24}


def test_plain_suffixes_on_multimodal_e4b_would_hit_the_towers(e4b_names):
    suffixes = mf.get_family('qwen3').target_modules
    hits = [n for n in e4b_names if mf.count_target_matches(suffixes, [n])]
    assert any('vision_tower' in n for n in hits) and any('audio_tower' in n for n in hits)


def test_per_layer_projections_are_never_targets(e4b_names):
    for targets in (GEMMA_REGEX, mf.get_family('gemma4').target_modules):
        assert not any('per_layer' in n for n in e4b_names if mf.count_target_matches(targets, [n]))


def test_lora_targets_refuses_a_target_set_that_matches_nothing():   # Review Focus 1
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match='qwen3.*match no module'):
        mf.lora_targets(mf.get_family('qwen3'), model)


def test_lora_targets_returns_the_configured_targets_when_they_match():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 2)

    f = mf.get_family('qwen3')
    assert mf.lora_targets(f, Block()) == f.target_modules


def test_slurm_family_table_matches_the_registry():
    """slurm/grpo_common.sh maps FAMILY -> TAG / BASE_LABEL in bash (the
    jobs need them before the container starts); it must agree with
    FAMILIES, or a job would look for the wrong models/ dir or CSV."""
    import subprocess
    for key, fam in mf.FAMILIES.items():
        out = subprocess.run(
            ['bash', '-c', f'source slurm/grpo_common.sh && FAMILY={key} family_vars && '
                           'echo "$TAG $BASE_LABEL"'],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        assert out == [fam.tag, mf.base_label(fam)], key


def test_slurm_family_table_rejects_unknown_families():
    import subprocess
    r = subprocess.run(['bash', '-c', 'source slurm/grpo_common.sh && FAMILY=llama family_vars'],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and 'llama' in r.stderr


# ── loading a real-architecture Gemma 4 checkpoint (tiny, CPU) ─────────────────

def _tiny_gemma4_checkpoint(path):
    """A ~2M-parameter Gemma4ForConditionalGeneration with the real E4B
    structure (vision + audio towers, PLE, KV sharing), saved the way the hub
    checkpoint is: language weights under model.language_model.*."""
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    c = json.loads(FIXTURE.read_text(encoding='utf-8'))
    c['text_config'].update(
        hidden_size=64, intermediate_size=128, num_hidden_layers=6,
        layer_types=['sliding_attention', 'full_attention'] * 3, num_kv_shared_layers=2,
        vocab_size=512, vocab_size_per_layer_input=512, hidden_size_per_layer_input=16,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32)
    small = {'hidden_size': 64, 'intermediate_size': 128, 'num_hidden_layers': 1,
             'num_attention_heads': 2, 'num_key_value_heads': 2, 'head_dim': 32}
    for tower in ('vision_config', 'audio_config'):
        c[tower].update({k: v for k, v in small.items() if k in c[tower]})
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(Gemma4Config(**c)).eval()
    model.save_pretrained(path)
    return model


@pytest.fixture(scope='module')
def tiny_gemma(tmp_path_factory):
    path = tmp_path_factory.mktemp('gemma4') / 'ckpt'
    return path, _tiny_gemma4_checkpoint(path)


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


IDS = torch.tensor([[2, 17, 42, 99, 7, 300]])


def test_gemma_text_loader_loads_every_language_weight(tiny_gemma):   # review C1
    path, multimodal = tiny_gemma
    text = mf.load_causal_lm(mf.get_family('gemma4'), path)
    assert type(text).__name__ == 'Gemma4ForCausalLM'
    assert torch.allclose(_logits(text.eval(), IDS), _logits(multimodal, IDS), atol=1e-5)


def test_saved_text_model_reloads_through_auto_unchanged(tiny_gemma, tmp_path):   # merged dirs
    from transformers import AutoModelForCausalLM
    path, multimodal = tiny_gemma
    mf.load_causal_lm(mf.get_family('gemma4'), path).save_pretrained(tmp_path / 'merged')
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / 'merged').eval()
    assert type(reloaded).__name__ == 'Gemma4ForCausalLM'
    assert torch.allclose(_logits(reloaded, IDS), _logits(multimodal, IDS), atol=1e-5)
    # and through the family loader too (train_grpo.load_policy's path)
    again = mf.load_causal_lm(mf.get_family('gemma4'), tmp_path / 'merged').eval()
    assert torch.allclose(_logits(again, IDS), _logits(multimodal, IDS), atol=1e-5)


def test_gemma_loader_refuses_a_checkpoint_missing_language_weights(tiny_gemma, tmp_path):
    from safetensors.torch import load_file, save_file
    path, _ = tiny_gemma
    broken = tmp_path / 'broken'
    broken.mkdir()
    for f in path.iterdir():
        if f.suffix != '.safetensors':
            (broken / f.name).write_bytes(f.read_bytes())
    tensors = load_file(path / 'model.safetensors')
    tensors.pop('model.language_model.layers.0.mlp.up_proj.weight')
    save_file(tensors, broken / 'model.safetensors', metadata={'format': 'pt'})
    with pytest.raises(RuntimeError, match='gemma4.*missing'):
        mf.load_causal_lm(mf.get_family('gemma4'), broken)


def test_family_check_reads_a_pre_family_merge_info_base_model(tmp_path):   # review minor 1
    (tmp_path / 'merge_info.json').write_text('{"base_model": "Qwen/Qwen3-4B", "adapter_dir": "x"}')
    with pytest.raises(ValueError, match='--family gemma4 .* qwen3'):
        mf.assert_family_matches(mf.get_family('gemma4'), tmp_path)
