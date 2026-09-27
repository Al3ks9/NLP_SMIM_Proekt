import compare_generations as cg


def test_rewards_flag_adds_reward_columns(monkeypatch):
    import grpo_rewards as gr
    monkeypatch.setattr(cg, '_poems_by_id', lambda: {'1': {'song_text': 'извор ред'}})
    monkeypatch.setattr(gr, 'score_batch', lambda s, g, t, cfg=None: [
        {'R': 0.4, 'R_style': 0.5, 'R_content': 0.3, 'R_style_g': 0.5,
         'R_content_g': 0.3, 'gate/copy_novelty': 1.0} for _ in g])
    # a real profiled author: score_run's structure_fit needs its style profile
    run = {'label': 'x', 'rows': [{'key': ('1', 'Блаже Конески', '0'), 'source_poem_id': '1',
                                   'target_author': 'Блаже Конески', 'split': 'val',
                                   'generated_poem': 'нова песна'}]}
    row = cg.score_run(run, rewards=True)[0]
    assert row['R'] == 0.4 and row['copy_novelty'] == 1.0
    assert cg.score_run(run, rewards=False)[0].get('R') is None
