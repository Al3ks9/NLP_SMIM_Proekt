"""The package layout's dependency rule (src/README.md): common imports no
pipeline, a pipeline imports common and earlier pipelines only, and the two
student families share code only through students.shared."""

import ast
import warnings
from pathlib import Path

from common.paths import ROOT

SRC = ROOT / 'src'

RANK = {'common': 0, 'word_substitution': 1, 'llm_infill': 2, 'gemma_teacher': 3, 'students': 4}

# the pre-restructure flat module names; an import of any of these means a
# rewrite was missed
FLAT_MODULES = {
    'add_poem_ids', 'strip_lines', 'make_splits', 'tagging', 'morph', 'corrections',
    'apply_corrections', 'pos_tag_corpus', 'build_morph_lookup', 'classla_test',
    'tfidf_authors', 'style_features', 'style_profiler', 'classifier', 'targets',
    'build_skg', 'enrich_skg', 'add_embeddings', 'graph_analytics', 'visualize_skg',
    'candidate_selection', 'style_metrics', 'content_similarity', 'judge_classifier',
    'structure', 'llm_client', 'style_transfer', 'llm_probe', 'probe_qwen', 'probe_gemma',
    'probe_nemotron', 'probe_google_api', 'show_slots', 'llm_style_transfer',
    'style_narrator', 'exemplar_selection', 'poem_tfidf', 'generate_synthetic',
    'merge_synthetic_parts', 'sft_data', 'model_families', 'train_sft', 'merge_sft_adapter',
    'grpo_data', 'grpo_rewards', 'build_reward_calibration', 'evaluate_reward', 'train_grpo',
    'generate_validation', 'compare_generations', 'diagnose_merge', 'plot_grpo_curves',
    'gemma_compat',
}


def _imports(path: Path) -> list:
    """Every imported module name in the file, at any depth (lazy imports
    inside functions included)."""
    with warnings.catch_warnings():
        # plot_grpo_curves.py carries an invalid escape in a plot label;
        # compiling it here would surface that as a test warning
        warnings.simplefilter('ignore', (DeprecationWarning, SyntaxWarning))
        tree = ast.parse(path.read_text(encoding='utf-8'))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def _sources():
    return sorted(SRC.rglob('*.py'))


def test_dependency_direction():
    violations = []
    for path in _sources():
        parts = path.relative_to(SRC).parts
        own = parts[0]
        if own not in RANK:
            continue
        sub = parts[1] if own == 'students' and len(parts) > 2 else None
        for name in _imports(path):
            head = name.split('.')[0]
            if head not in RANK:
                continue
            rel = path.relative_to(ROOT)
            if RANK[head] > RANK[own]:
                violations.append(f'{rel}: imports later pipeline {name}')
            if head == 'students' and sub is not None:
                target = name.split('.')[1] if '.' in name else None
                if sub == 'shared' and target in ('qwen', 'gemma'):
                    violations.append(f'{rel}: students.shared imports {name}')
                if sub in ('qwen', 'gemma') and target in ('qwen', 'gemma') and target != sub:
                    violations.append(f'{rel}: imports the other student family {name}')
    assert not violations, '\n'.join(violations)


def test_no_flat_imports():
    stale = [f'{p.relative_to(ROOT)}: {name}' for p in _sources()
             for name in _imports(p) if name.split('.')[0] in FLAT_MODULES]
    assert not stale, '\n'.join(stale)


def _shell_files():
    return sorted(set(SRC.rglob('hpc/*')) | set(SRC.rglob('*.sh')))


def test_hpc_scripts_reference_live_modules():
    """Every Slurm/shell file next to a pipeline runs modules that exist,
    names no pre-restructure path, and (if it runs the container) puts
    /workspace/src on PYTHONPATH -- the .sif images don't install the project."""
    import importlib.util
    import re
    files = _shell_files()
    assert files, 'no hpc/ or .sh files under src/'
    problems = []
    for path in files:
        text = path.read_text(encoding='utf-8')
        rel = path.relative_to(ROOT)
        for mod in re.findall(r'-m ([a-z_]+(?:\.[a-z_]+)+)', text):
            if importlib.util.find_spec(mod) is None:
                problems.append(f'{rel}: -m {mod} does not exist')
        for pattern in (r'\bsrc/\w+\.py', r'(?<![\w/])slurm/', r'(?<![\w/])scripts/',
                        r'singularity/(qwen-sft\.def|build\.sh)'):
            for hit in re.findall(pattern, text):
                problems.append(f'{rel}: stale path {hit!r}')
        runs_container = 'singularity exec' in text or 'grpo_common.sh' in text
        if runs_container and path.suffix == '.slurm' and 'grpo_common.sh' not in text \
                and 'PYTHONPATH=/workspace/src' not in text:
            problems.append(f'{rel}: runs the container without PYTHONPATH=/workspace/src')
    common_sh = [p for p in files if p.name == 'grpo_common.sh']
    if common_sh and 'PYTHONPATH=/workspace/src' not in common_sh[0].read_text(encoding='utf-8'):
        problems.append('grpo_common.sh: run_in_container lacks PYTHONPATH=/workspace/src')
    assert not problems, '\n'.join(problems)


def test_docs_run_modules_not_paths():
    """`python src/.../x.py` fails inside the container (the project isn't
    installed there; imports resolve only via -m with PYTHONPATH=src)."""
    import re
    texts = [ROOT / 'CLAUDE.md'] + [p for p in SRC.rglob('*') if p.is_file()
                                     and p.suffix in ('.py', '.sh', '.slurm', '.def', '.md')]
    hits = [f'{p.relative_to(ROOT)}: {m}' for p in texts
            for m in re.findall(r'python3? src/[\w/]+\.py', p.read_text(encoding='utf-8'))]
    assert not hits, '\n'.join(hits)


def test_claude_md_paths_exist():
    import re
    text = (ROOT / 'CLAUDE.md').read_text(encoding='utf-8')
    paths = set(re.findall(r'(?<![\w/])((?:src|tests)/[\w/.-]+\.(?:py|sh|slurm|def))', text))
    missing = sorted(p for p in paths if not (ROOT / p).exists())
    assert not missing, '\n'.join(missing)
