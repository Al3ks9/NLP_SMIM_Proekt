import json

import pytest

import train_grpo as tg

TINY = 'trl-internal-testing/tiny-Qwen3ForCausalLM'


def test_defaults_match_the_spec():
    a = tg.parse_args([])
    assert (a.num_generations, a.per_device_train_batch_size, a.gradient_accumulation_steps) == (4, 8, 4)
    assert a.learning_rate == 1e-5 and a.beta == 0.04 and a.temperature == 0.9
    assert (a.lora_r, a.lora_alpha, a.lora_dropout) == (16, 32, 0.0)
    assert (a.style_weight, a.content_weight) == (0.5, 0.5)
    assert a.max_steps == 500 and a.seed == 42 and a.max_prompt_length == 1024


def test_grpo_config_carries_reward_weights_and_masks_truncation(tmp_path):
    a = tg.parse_args(['--output-dir', str(tmp_path), '--style-weight', '0.7',
                       '--content-weight', '0.3', '--no-bf16'])
    c = tg.build_grpo_config(a)
    assert c.reward_weights == [0.7, 0.3]
    assert c.mask_truncated_completions is True      # truncated poems get no credit
    assert c.beta == 0.04 and c.num_generations == 4
    off = tg.build_grpo_config(tg.parse_args(['--output-dir', str(tmp_path), '--no-bf16',
                                              '--no-mask-truncated-completions']))
    assert off.mask_truncated_completions is False


def test_smoke_checks_name_total_truncation_explicitly():
    class State:
        log_history = [{'loss': 0.0, 'reward': 0.5, 'reward_std': 0.2,
                        'completions/clipped_ratio': 1.0}]

    class Trainer:
        state = State()
        model = None
        args = type('Args', (), {'mask_truncated_completions': True})()

    msgs = tg._history_failures(Trainer())
    assert any('truncated' in m for m in msgs)


def test_smoke_overrides_are_tiny():
    a = tg.apply_smoke_overrides(tg.parse_args(['--smoke']))
    assert a.max_steps == 3 and a.num_prompts == 4
    # SFT completions have median 291 tokens; a shorter cap would truncate --
    # and, with truncation masking, zero the loss of -- every smoke completion
    assert a.max_completion_length == 512
    assert (a.per_device_train_batch_size * a.gradient_accumulation_steps) % a.num_generations == 0


def test_style_weights_json_merges_with_defaults():
    a = tg.parse_args(['--style-weights', '{"rhyme": 0.0}'])
    assert tg.reward_config(a).style_weights['rhyme'] == 0.0
    assert tg.reward_config(a).style_weights['pos_profile'] == 1.0


def test_missing_merge_info_is_refused(tmp_path):
    with pytest.raises(SystemExit, match='merge_info'):
        tg.load_policy(tmp_path, 'sdpa')


@pytest.mark.slow
def test_plumbing_end_to_end_on_a_tiny_model(tmp_path, monkeypatch):
    """Wiring only: dataset columns reach the reward functions, TRL steps,
    the adapter + run_config are saved. Rewards are stubbed; the real smoke
    test (slurm/grpo_smoke.slurm) runs the real ones on the real model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    merged = tmp_path / 'merged'
    AutoModelForCausalLM.from_pretrained(TINY).save_pretrained(merged)
    AutoTokenizer.from_pretrained(TINY).save_pretrained(merged)
    (merged / 'merge_info.json').write_text(json.dumps({'adapter_dir': 'x', 'adapter_sha256': 'y'}))

    seen = {}

    def fake_make_reward_funcs(cfg=None, **_):
        def style_reward(prompts, completions, source_text, target_author, **kw):
            seen['n'] = len(completions)
            seen['cols'] = (len(source_text), len(target_author))
            return [float(len(c) % 3) for c in completions]

        def content_reward(prompts, completions, source_text, target_author, **kw):
            return [0.5] * len(completions)
        return style_reward, content_reward

    monkeypatch.setattr(tg, 'make_reward_funcs', fake_make_reward_funcs)
    out = tmp_path / 'out'
    tg.train(tg.parse_args([
        '--sft-merged', str(merged), '--output-dir', str(out), '--max-steps', '2',
        '--num-prompts', '4', '--num-generations', '2', '--per-device-train-batch-size', '4',
        '--gradient-accumulation-steps', '1', '--max-completion-length', '8',
        '--no-bf16', '--no-gradient-checkpointing', '--save-steps', '100', '--use-cpu']))

    assert (out / 'adapter_config.json').exists()
    cfg = json.loads((out / 'run_config.json').read_text())
    assert cfg['args']['num_generations'] == 2 and 'calibration_sha256' in cfg
    assert seen['cols'] == (seen['n'], seen['n'])


@pytest.mark.slow
def test_smoke_path_passes_its_own_checks_on_a_tiny_model(tmp_path, monkeypatch):
    """Exercises run_smoke_checks / completion_logprob locally, so the
    cluster smoke job is not the first run of that code."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    merged = tmp_path / 'merged'
    AutoModelForCausalLM.from_pretrained(TINY).save_pretrained(merged)
    AutoTokenizer.from_pretrained(TINY).save_pretrained(merged)
    (merged / 'merge_info.json').write_text(json.dumps({'adapter_dir': 'x', 'adapter_sha256': 'y'}))

    def fake_make_reward_funcs(cfg=None, **_):
        def style_reward(prompts, completions, **kw):
            return [float(len(c) % 5) for c in completions]

        def content_reward(prompts, completions, **kw):
            return [float(len(c) % 2) for c in completions]
        return style_reward, content_reward

    monkeypatch.setattr(tg, 'make_reward_funcs', fake_make_reward_funcs)
    args = tg.apply_smoke_overrides(tg.parse_args([
        '--smoke', '--sft-merged', str(merged), '--output-dir', str(tmp_path / 'out'),
        '--no-bf16', '--no-gradient-checkpointing', '--use-cpu',
        # the random tiny model never emits EOS: without these every completion
        # is truncated and masked, and nothing trains
        '--max-completion-length', '16', '--no-mask-truncated-completions']))
    tg.train(args)   # sys.exit(1) on any failed smoke check
