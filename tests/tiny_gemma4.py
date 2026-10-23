"""Shared test helper: a tiny real-architecture Gemma 4 checkpoint.

Used by tests/students/shared/test_model_families.py and
tests/students/gemma/test_gemma_compat.py. Lives at tests/ (on pytest's
pythonpath, see pyproject.toml) so neither test depends on the other having
been collected first."""

import json

import torch

from common.paths import ROOT

FIXTURE = ROOT / 'tests' / 'fixtures' / 'gemma4_e4b_config.json'


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
