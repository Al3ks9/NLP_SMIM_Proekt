"""Qwen SFT pipeline, stages 3-4: LoRA config and training orchestration.
Model/dataset loading and the actual TRL trainer are monkeypatched throughout
-- no model download or GPU needed to exercise the wiring."""

import pytest

import train_sft as ts


# ── build_lora_config ────────────────────────────────────────────────────────────

def test_build_lora_config_uses_the_given_hyperparameters():
    config = ts.build_lora_config(16, 32, 0.05, ts.DEFAULT_TARGET_MODULES)
    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert set(config.target_modules) == set(ts.DEFAULT_TARGET_MODULES)
    assert config.task_type == 'CAUSAL_LM'


def test_default_target_modules_matches_confirmed_qwen3_projections():
    # Confirmed against transformers' actual Qwen3 modeling code -- see
    # train_sft.py's module docstring. q_norm/k_norm are RMSNorm layers, not
    # nn.Linear, and must NOT be here.
    assert set(ts.DEFAULT_TARGET_MODULES) == {
        'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    }
    assert 'q_norm' not in ts.DEFAULT_TARGET_MODULES
    assert 'k_norm' not in ts.DEFAULT_TARGET_MODULES


# ── build_datasets ───────────────────────────────────────────────────────────────

def test_build_datasets_raises_when_train_split_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(ts, 'load_synthetic_rows', lambda split, path: [])
    with pytest.raises(ValueError):
        ts.build_datasets(tmp_path / 'synthetic.csv', eval_split='val')


def test_build_datasets_returns_none_eval_when_eval_split_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(ts, 'load_synthetic_rows',
                        lambda split, path: [{'x': 1}] if split == 'train' else [])
    monkeypatch.setattr(ts, 'build_hf_dataset', lambda rows: rows)
    train_ds, eval_ds = ts.build_datasets(tmp_path / 'synthetic.csv', eval_split='')
    assert train_ds == [{'x': 1}]
    assert eval_ds is None


def test_build_datasets_warns_and_disables_eval_when_split_has_no_rows(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(ts, 'load_synthetic_rows',
                        lambda split, path: [{'x': 1}] if split == 'train' else [])
    monkeypatch.setattr(ts, 'build_hf_dataset', lambda rows: rows)
    with caplog.at_level('WARNING'):
        train_ds, eval_ds = ts.build_datasets(tmp_path / 'synthetic.csv', eval_split='val')
    assert train_ds == [{'x': 1}]
    assert eval_ds is None
    assert 'no split' in caplog.text


def test_build_datasets_builds_eval_when_val_rows_present(monkeypatch, tmp_path):
    rows_by_split = {'train': [{'x': 1}], 'val': [{'x': 2}]}
    monkeypatch.setattr(ts, 'load_synthetic_rows', lambda split, path: rows_by_split[split])
    monkeypatch.setattr(ts, 'build_hf_dataset', lambda rows: rows)
    train_ds, eval_ds = ts.build_datasets(tmp_path / 'synthetic.csv', eval_split='val')
    assert train_ds == [{'x': 1}]
    assert eval_ds == [{'x': 2}]


# ── parse_args defaults ──────────────────────────────────────────────────────────

def test_parse_args_defaults_match_the_documented_starting_point():
    args = ts.parse_args([])
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.lora_dropout == 0.05
    assert args.target_modules == ts.DEFAULT_TARGET_MODULES
    assert args.bf16 is True
    assert args.gradient_checkpointing is True
    assert args.eval_split == 'val'
    assert args.model_name == ts.DEFAULT_MODEL_NAME


def test_parse_args_no_bf16_flag_disables_bf16():
    args = ts.parse_args(['--no-bf16'])
    assert args.bf16 is False


def test_parse_args_no_gradient_checkpointing_flag_disables_it():
    args = ts.parse_args(['--no-gradient-checkpointing'])
    assert args.gradient_checkpointing is False


def test_parse_args_accepts_custom_lora_hyperparameters():
    args = ts.parse_args(['--lora-r', '32', '--lora-alpha', '64', '--lora-dropout', '0.1'])
    assert args.lora_r == 32
    assert args.lora_alpha == 64
    assert args.lora_dropout == 0.1


# ── train() orchestration ────────────────────────────────────────────────────────

class _FakeTrainer:
    instances = []

    def __init__(self, model, args, train_dataset, eval_dataset, peft_config, processing_class):
        self.model, self.args = model, args
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.peft_config, self.processing_class = peft_config, processing_class
        self.saved_to = None
        _FakeTrainer.instances.append(self)

    def train(self, resume_from_checkpoint=None):
        self.resumed_from = resume_from_checkpoint

    def save_model(self, output_dir):
        self.saved_to = output_dir


class _FakeTokenizer:
    def save_pretrained(self, path):
        self.saved_to = path


def test_train_wires_assistant_only_loss_and_lora_config_through(monkeypatch, tmp_path):
    _FakeTrainer.instances.clear()
    monkeypatch.setattr(ts, 'SFTTrainer', _FakeTrainer)
    monkeypatch.setattr(ts, 'load_model_and_tokenizer',
                        lambda *a, **kw: ('MODEL', _FakeTokenizer()))
    monkeypatch.setattr(ts, 'build_datasets', lambda *a, **kw: ('TRAIN_DS', 'EVAL_DS'))

    # --no-bf16: SFTConfig validates bf16 against actual hardware at
    # construction time (raises without a bf16-capable GPU) -- this test
    # exercises wiring, not real bf16 training, and must pass on a CPU-only
    # dev machine too.
    args = ts.parse_args(['--output-dir', str(tmp_path / 'out'), '--no-bf16'])
    ts.train(args)

    trainer = _FakeTrainer.instances[0]
    assert trainer.model == 'MODEL'
    assert trainer.train_dataset == 'TRAIN_DS'
    assert trainer.eval_dataset == 'EVAL_DS'
    assert trainer.args.assistant_only_loss is True
    assert trainer.peft_config.r == args.lora_r
    assert trainer.saved_to == str(tmp_path / 'out')
    assert (tmp_path / 'out' / 'run_config.json').exists()


# ── --family ─────────────────────────────────────────────────────────────────────

from model_families import MODELS, get_family


def test_family_default_leaves_qwen_config_unchanged():
    a = ts.resolve(ts.parse_args([]))
    assert a.family == 'qwen3' and a.model_name == 'Qwen/Qwen3-4B'
    assert a.output_dir == MODELS / 'qwen3-lora-sft'
    assert a.target_modules == ts.DEFAULT_TARGET_MODULES


def test_gemma_family_resolves_its_own_defaults():
    a = ts.parse_args(['--family', 'gemma4'])
    assert a.model_name == 'google/gemma-4-E4B-it' and a.output_dir == MODELS / 'gemma4-e4b-lora-sft'
    assert a.target_modules == get_family('gemma4').target_modules


def test_explicit_paths_beat_family_defaults(tmp_path):
    a = ts.parse_args(['--family', 'gemma4', '--output-dir', str(tmp_path), '--model-name', 'x/y'])
    assert a.output_dir == tmp_path and a.model_name == 'x/y'


def test_gemma_family_uses_completion_only_loss():
    cfg = ts.build_sft_config(ts.parse_args(['--family', 'gemma4', '--no-bf16']))
    assert cfg.completion_only_loss is True and cfg.assistant_only_loss is False


def test_qwen_family_keeps_assistant_only_loss():
    cfg = ts.build_sft_config(ts.parse_args(['--no-bf16']))
    assert cfg.assistant_only_loss is True


def test_gemma_train_builds_prompt_completion_datasets(monkeypatch, tmp_path):
    _FakeTrainer.instances.clear()
    seen = {}
    monkeypatch.setattr(ts, 'SFTTrainer', _FakeTrainer)
    monkeypatch.setattr(ts, 'load_model_and_tokenizer', lambda *a, **kw: ('MODEL', _FakeTokenizer()))

    def fake_build_datasets(path, eval_split='val', fmt='messages', tokenizer=None):
        seen['fmt'], seen['tokenizer'] = fmt, tokenizer
        return 'TRAIN_DS', None
    monkeypatch.setattr(ts, 'build_datasets', fake_build_datasets)
    monkeypatch.setattr(ts, 'end_of_turn', lambda tok: '<turn|>')
    ts.train(ts.parse_args(['--family', 'gemma4', '--output-dir', str(tmp_path / 'o'), '--no-bf16']))
    assert seen['fmt'] == 'prompt_completion' and isinstance(seen['tokenizer'], _FakeTokenizer)
    cfg = (tmp_path / 'o' / 'run_config.json').read_text(encoding='utf-8')
    assert '"family": "gemma4"' in cfg


def test_gemma_sft_stops_on_end_of_turn_not_eos(monkeypatch, tmp_path):   # review I1
    # TRL appends tokenizer.eos_token (<eos>) to every completion unless
    # SFTConfig.eos_token says otherwise; GRPO stops on tokenizer.eos_token.
    # Gemma's turn ends with <turn|>, so that must be the eos for both.
    _FakeTrainer.instances.clear()
    monkeypatch.setattr(ts, 'SFTTrainer', _FakeTrainer)
    monkeypatch.setattr(ts, 'load_model_and_tokenizer', lambda *a, **kw: ('MODEL', _FakeTokenizer()))
    monkeypatch.setattr(ts, 'build_datasets', lambda *a, **kw: ('TRAIN_DS', None))
    monkeypatch.setattr(ts, 'end_of_turn', lambda tok: '<turn|>')
    ts.train(ts.parse_args(['--family', 'gemma4', '--output-dir', str(tmp_path / 'g'), '--no-bf16']))
    ts.train(ts.parse_args(['--output-dir', str(tmp_path / 'q'), '--no-bf16']))
    gemma, qwen = _FakeTrainer.instances
    assert gemma.args.eos_token == '<turn|>'
    assert qwen.args.eos_token is None          # Qwen unchanged


def test_max_seq_length_defaults_per_family():   # review I2
    assert ts.parse_args([]).max_seq_length == 1024
    assert ts.parse_args(['--family', 'gemma4']).max_seq_length == 4096
    assert ts.parse_args(['--family', 'gemma4', '--max-seq-length', '2048']).max_seq_length == 2048
