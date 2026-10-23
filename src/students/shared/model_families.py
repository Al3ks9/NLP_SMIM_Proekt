"""
Student model families for the SFT -> merge -> GRPO -> eval pipeline.

Everything that differs between Qwen3-4B and Gemma 4 E4B lives here, so the
scripts choose a family with one --family flag instead of each hardcoding
Qwen. See docs/superpowers/specs/2026-10-02-gemma4-student-design.md.

`qwen3` resolves to exactly the values the pipeline used before families
existed (hub name, LoRA targets, assistant_only_loss, models/qwen3-* dirs), so
every existing run stays reproducible.

Gemma 4 E4B facts the entries below rely on (transformers 5.9.0, real E4B
config in tests/fixtures/gemma4_e4b_config.json):

- AutoModelForCausalLM maps the `gemma4` checkpoint to
  Gemma4ForConditionalGeneration, which also loads a vision and an audio
  tower. `loader='gemma4_text'` loads only the language model.
- The towers reuse the names q_proj ... down_proj for Gemma4ClippableLinear
  wrappers PEFT cannot adapt, so a plain suffix target list is only safe on
  the text-only model; on the multimodal one the targets must be the
  `language_model` regex.
- The last 18 of 42 layers share key/value states and have no k_proj/v_proj.
- per_layer_input_gate / per_layer_projection are never targeted, for parity
  with Qwen's attention + MLP target set.
- Gemma ships top_k=64; Qwen3 ships 20. Gemma runs pin 20 so decoding matches.
- TRL 1.13 has no Gemma 4 training chat template, so assistant_only_loss does
  not work; Gemma trains on prompt-completion rows with completion_only_loss.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'

ATTN_MLP = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
GEMMA4_LM_REGEX = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'


@dataclass(frozen=True)
class ModelFamily:
    key: str
    hf_name: str
    tag: str                    # prefix of models/ dirs and generation labels
    target_modules: object      # list = suffix match, str = full-match regex (PEFT's rule)
    loss_masking: str           # 'assistant_only' | 'completion_only'
    loader: str                 # 'auto' | 'gemma4_text'
    generation_overrides: dict = field(default_factory=dict)
    max_seq_length: int = 1024  # SFT; spec §7: smallest that truncates no train example


FAMILIES = {
    'qwen3': ModelFamily('qwen3', 'Qwen/Qwen3-4B', 'qwen3', list(ATTN_MLP),
                         'assistant_only', 'auto'),
    'gemma4': ModelFamily('gemma4', 'google/gemma-4-E4B-it', 'gemma4-e4b', list(ATTN_MLP),
                          'completion_only', 'gemma4_text', {'top_k': 20},
                          max_seq_length=4096),  # 10 train rows exceed 2048 Gemma tokens (max 3,371)
}
DEFAULT_FAMILY = 'qwen3'


def get_family(key: str) -> ModelFamily:
    if key not in FAMILIES:
        raise ValueError(f'unknown model family {key!r}; known: {", ".join(FAMILIES)}')
    return FAMILIES[key]


def default_dirs(family: ModelFamily) -> dict:
    return {'sft': MODELS / f'{family.tag}-lora-sft',
            'merged': MODELS / f'{family.tag}-sft-merged',
            'grpo': MODELS / f'{family.tag}-grpo'}


def base_label(family: ModelFamily) -> str:
    return 'base-' + family.hf_name.split('/')[-1]


def _matches(targets, name: str) -> bool:
    if isinstance(targets, str):
        return re.fullmatch(targets, name) is not None
    return any(name == t or name.endswith('.' + t) for t in targets)


def count_target_matches(targets, module_names: list) -> Counter:
    """{last name component: count} of the modules PEFT would adapt."""
    return Counter(n.rsplit('.', 1)[-1] for n in module_names if _matches(targets, n))


def lora_targets(family: ModelFamily, model, targets=None):
    """The LoRA targets to use (the family's, or an explicit override), after
    checking they match at least one module of the loaded model."""
    targets = family.target_modules if targets is None else targets
    if not count_target_matches(targets, [n for n, _ in model.named_modules()]):
        raise ValueError(f'{family.key}: LoRA targets {targets!r} match no module of '
                         f'{type(model).__name__}')
    return targets


def load_causal_lm(family: ModelFamily, path: str, **kwargs):
    """The family's causal LM from a hub name or local dir.

    gemma4_text: the hub checkpoint is multimodal (model_type gemma4) and
    stores language weights under model.language_model.*, which
    Gemma4ForCausalLM expects at model.*. transformers 5.9 has no conversion
    for that, and a plain from_pretrained silently leaves every language
    weight randomly initialised -- so the prefix is mapped explicitly, the
    recorded conversion is cleared (otherwise save_pretrained re-applies the
    reverse mapping and the merged dir loses every weight on reload), and any
    missing weight raises. A merged Gemma dir is saved text-only
    (model_type gemma4_text) and loads through AutoModelForCausalLM, under
    the same missing-weight check. tests/test_model_families.py verifies all
    of this on a tiny real-architecture checkpoint.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    if family.loader != 'gemma4_text':
        return AutoModelForCausalLM.from_pretrained(str(path), **kwargs)

    config = AutoConfig.from_pretrained(str(path))
    if config.model_type == 'gemma4':
        from transformers import Gemma4ForCausalLM
        model, info = Gemma4ForCausalLM.from_pretrained(
            str(path), config=config.get_text_config(),
            key_mapping={r'^model\.language_model\.': 'model.'},
            output_loading_info=True, **kwargs)
        model._weight_conversions = None
    else:
        model, info = AutoModelForCausalLM.from_pretrained(str(path), output_loading_info=True,
                                                           **kwargs)
    missing = sorted(info['missing_keys'])
    if missing:
        raise RuntimeError(f'{family.key}: {len(missing)} weights missing loading {path}, '
                           f'e.g. {missing[:3]} -- refusing to run on random weights')
    return model


def family_of(name: str):
    """Family key for a hub name or a models/ dir name, None if unknown."""
    base = Path(str(name)).name
    for f in FAMILIES.values():
        if name == f.hf_name or base == f.hf_name.split('/')[-1] or base.startswith(f.tag + '-'):
            return f.key
    return None


def assert_family_matches(family: ModelFamily, model_dir) -> None:
    """Refuse an adapter / merged dir built from the other family -- a clear
    error here instead of a shape mismatch deep inside PEFT. Provenance comes
    from merge_info.json's family (or, for pre-family merges, its base_model),
    else adapter_config.json's base model;
    unknown provenance (pre-family merges, unfamiliar hub ids) passes."""
    model_dir = Path(model_dir)
    built_from = None
    info = model_dir / 'merge_info.json'
    adapter = model_dir / 'adapter_config.json'
    if info.exists():
        record = json.loads(info.read_text(encoding='utf-8'))
        built_from = record.get('family') or (family_of(record['base_model'])
                                              if record.get('base_model') else None)
    if built_from is None and adapter.exists():
        base = json.loads(adapter.read_text(encoding='utf-8')).get('base_model_name_or_path')
        built_from = family_of(base) if base else None
    if built_from is not None and built_from != family.key:
        raise ValueError(f'--family {family.key} but {model_dir} was built from {built_from}')
