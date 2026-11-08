"""The rating page: safe embedding, no system leak, a per-rater order that keeps
a pair's outputs apart."""

import json
import shutil
import subprocess

import pytest

from students.shared import human_eval as he

PAGE_JS = he.PAGE_DIR / 'page.js'


def _items(poems):
    return [{'item_id': f'id{i}', 'group': i // 2, 'target_author': 'Блаже Конески',
             'source': {'author': 'Кочо Рацин', 'title': 'Балада', 'text': 'извор'},
             'poem': p} for i, p in enumerate(poems)]


def test_render_escapes_script_close():
    html = he.render_page(_items(['</script><b>&', 'ок']), {}, 'v1')
    assert '</script><b>' not in html
    assert html.count('</script>') == html.count('<script')


def test_render_embeds_data_and_version():
    html = he.render_page(_items(['прва', 'втора']), {'Блаже Конески': [{'title': 'Т', 'text': 'Т'}]}, 'v42')
    assert 'id0' in html and 'id1' in html and 'v42' in html
    assert '/*__DATA__*/' not in html and '/*__JS__*/' not in html


def test_render_has_no_system_labels(monkeypatch):
    train = [{'poem_id': str(p), 'author': a, 'song_title': f't{p}',
              'song_text': '\n'.join(f'стих {i}' for i in range(10))}
             for a, ps in (('A', range(10, 16)),) for p in ps]
    corpus = [{'poem_id': str(p), 'author': 'S', 'song_title': 'x', 'song_text': 'извор'} for p in (1, 2)]
    monkeypatch.setattr(he, 'load_stripped_songs', lambda: corpus + train)
    monkeypatch.setattr(he, 'poems_in_split', lambda split: train)
    rows = [{'source_poem_id': p, 'target_author': 'A', 'generated_poem': 'песна'} for p in ('1', '2')]
    items, key, panels = he.build_items({'teacher': rows, 'qwen-sft': rows}, ['teacher', 'qwen-sft'],
                                        seed=1, f1=lambda a, b: 0.0)
    html = he.render_page(items, panels, key['page_version'])
    for label in ('teacher', 'qwen-sft', 'real-poem', 'control_poem_id'):
        assert label not in html


def _order(name, groups):
    script = (f'const {{orderFor}} = require({json.dumps(str(PAGE_JS))});'
              f'console.log(JSON.stringify(orderFor({json.dumps(name)}, {len(groups)}, {json.dumps(groups)})))')
    return json.loads(subprocess.run(['node', '-e', script], capture_output=True, text=True,
                                     check=True).stdout)


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_order_is_permutation_without_adjacent_groups():
    groups = [i // 6 for i in range(120)]            # 20 pairs x 6 outputs
    for name in ['Алекс', 'a', 'b', 'c']:
        order = _order(name, groups)
        assert sorted(order) == list(range(120))
        assert all(groups[x] != groups[y] for x, y in zip(order, order[1:]))
        assert _order(name, groups) == order
    assert _order('a', groups) != _order('b', groups)


def _node(expr):
    script = f'const m = require({json.dumps(str(PAGE_JS))}); console.log(JSON.stringify({expr}))'
    return json.loads(subprocess.run(['node', '-e', script], capture_output=True, text=True,
                                     check=True).stdout)


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_score_key_ignores_modifier_shortcuts():
    assert _node("m.scoreKey({key: '3'})") == 3
    assert _node("m.scoreKey({key: '3', metaKey: true})") is None      # Cmd+3 switches tabs
    assert _node("m.scoreKey({key: '3', ctrlKey: true})") is None
    assert _node("m.scoreKey({key: '7'})") is None


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_unexported_ratings_are_detected():
    assert _node("m.hasUnexported({ratings: {a: {scores: {x: 1}}}}, 0)") is True
    assert _node("m.hasUnexported({ratings: {a: {scores: {x: 1}}}}, 1)") is False
    assert _node("m.hasUnexported({ratings: {}}, 0)") is False
