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


def test_judge_columns_blank_for_corpus_poems_the_judge_trained_on(monkeypatch):
    import judge_classifier as jc
    monkeypatch.setattr(cg, '_poems_by_id', lambda: {})
    monkeypatch.setattr(cg, 'load_splits', lambda: {'1': 'train', '2': 'test'})
    monkeypatch.setattr(jc, 'judge_match', lambda text, author, model=None: {
        'judge_target_prob': 0.9, 'judge_top1': 1.0, 'judge_predicted': author})
    rows = [{'key': (pid, 'Блаже Конески', '0'), 'source_poem_id': pid,
             'target_author': 'Блаже Конески', 'split': 'corpus',
             'generated_poem': 'нова песна', 'source_text': None} for pid in ('1', '2')]
    seen, held_out = cg.score_run({'label': 'REAL POEMS', 'rows': rows})
    assert seen['judge_top1'] is None and seen['judge_target_prob'] is None
    assert held_out['judge_top1'] == 1.0
