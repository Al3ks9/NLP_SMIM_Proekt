"""
GRPO v1 vs v2 training curves for rl_report.tex, from the runs'
reward_components.jsonl (one row per scored completion; 500 steps x 32).

Plots only components on the same scale in both runs: v2 reweights the
classifier group (5.6) and raises the copy gate to the 3rd power, so R,
R_style and R_content_g are not comparable between runs; the group scores,
ungated content and the gates are. Also prints first-/last-50-step means.

The jsonl files are copied back from the cluster (models/qwen3-grpo*/) and
gitignored (~50 MB each):
    uv run --with matplotlib python scripts/plot_grpo_curves.py \   # plotting only: not a project dep
        --v1 models/reward_components_grpo_v1.jsonl \
        --v2 models/reward_components_grpo_v2.jsonl
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'figures' / 'grpo_training_curves.pdf'

PANELS = [
    ('group/classifier', 'Classifier probability of target'),
    ('group/lexical_diversity', 'Lexical diversity score'),
    ('group/pos_bigrams', 'POS-bigram score'),
    ('R_content', 'Content score (ungated)'),
    ('gate/copy_novelty', 'Copy novelty'),
    ('gate/line_uniqueness', 'Line uniqueness'),
]
# Reference palette, categorical slots 1-2 (validated: CVD dE 24.7, contrast >= 3:1)
COLORS = {'v1': '#2a78d6', 'v2': '#eb6834'}
INK, MUTED, GRID = '#1f1f1e', '#6b6b67', '#e4e4e0'
SMOOTH = 25   # steps


def per_step(path: Path) -> pd.DataFrame:
    cols = ['step'] + [c for c, _ in PANELS]
    rows = [{k: r[k] for k in cols} for r in map(json.loads, path.open(encoding='utf-8'))]
    return pd.DataFrame(rows).groupby('step').mean()


def main(v1: Path, v2: Path, out: Path) -> None:
    runs = {'v1': per_step(v1), 'v2': per_step(v2)}
    for label, df in runs.items():
        first, last = df.iloc[:50].mean(), df.iloc[-50:].mean()
        print(label, ' | '.join(f'{c.split("/")[-1]} {first[c]:.3f}->{last[c]:.3f}' for c, _ in PANELS))

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for ax, (col, title) in zip(axes.flat, PANELS):
        for label, df in runs.items():
            y = df[col].rolling(SMOOTH, min_periods=10).mean()
            ax.plot(df.index, y, color=COLORS[label], linewidth=1.6, label=f'GRPO {label}')
            if ax is axes.flat[0]:   # direct labels once; the legend carries identity elsewhere
                ax.annotate(f'GRPO {label}', (df.index[-1], y.iloc[-1]), xytext=(3, 0),
                            textcoords='offset points', va='center', fontsize=7.5, color=INK)
        ax.set_title(title, fontsize=8.5, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2)
    for ax in axes[1]:
        ax.set_xlabel('training step')
    axes.flat[0].set_xlim(0, 540)   # room for the direct labels
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=160)
    print('wrote', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v1', type=Path, default=ROOT / 'models' / 'reward_components_grpo_v1.jsonl')
    p.add_argument('--v2', type=Path, default=ROOT / 'models' / 'reward_components_grpo_v2.jsonl')
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    main(a.v1, a.v2, a.out)
