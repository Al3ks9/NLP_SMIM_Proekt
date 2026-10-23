from pathlib import Path

import llm_style_transfer as lst
import structure
import style_metrics


def test_lst_reexports_structure():
    assert lst.structural_targets is structure.structural_targets
    assert lst.validate_structure is structure.validate_structure
    assert lst.STRUCTURAL_COLUMNS is structure.STRUCTURAL_COLUMNS


def _imported_modules(path):
    import ast
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_style_metrics_does_not_import_teacher():
    # the GRPO reward must not depend on the Gemma teacher's pipeline
    assert not any(m.endswith('llm_style_transfer')
                   for m in _imported_modules(style_metrics.__file__))
