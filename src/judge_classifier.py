"""
Held-out author judge: an evaluation-only second opinion on "does this poem
sound like the target author", independent of models/classifier.pkl.

Why it exists: classifier.pkl's target-author probability is a GRPO reward
component (grpo_rewards.py's `classifier` group, weighted up to 5.6 in
GRPO-v2). A policy trained to raise that number will raise it whether or not
the poem really reads as the author -- so on its own it cannot show that
GRPO worked. classifier.pkl was also fitted on the whole corpus, val and test
poems included, so its score on real poems is memorization (top-1 1.0, against
0.883 held out).

The judge differs on all three axes the policy could have exploited:

- data:     train-split poems only (make_splits), so every val/test real poem
            is unseen and its score is an honest ceiling;
- features: word unigrams + bigrams, not character (2,4)-grams;
- model:    multinomial logistic regression, not a calibrated LinearSVC.

If a GRPO run raises clf_top1 but not judge_top1, the gain is the classifier
being gamed, not the author's style being learned. Never import this from a
reward (tests/test_judge_classifier.py enforces it).

Deterministic and fast (seconds), so it is fitted on first use and cached at
models/judge_classifier.pkl (gitignored); `--rebuild` refits and prints its
held-out accuracy on the val+test poems.

Usage:
    uv run python src/judge_classifier.py --rebuild
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from make_splits import load_splits, load_stripped_songs
from style_metrics import lines, tokens

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
DATA = ROOT / 'data'
JUDGE_PATH = MODELS / 'judge_classifier.pkl'
EVAL_PATH = DATA / 'judge_classifier_eval.json'

# Same eligibility floor as classifier.py, but counted on train poems only.
MIN_TRAIN_POEMS = 5

_MODEL = None


def training_set(poems: list, splits: dict, min_poems: int = MIN_TRAIN_POEMS) -> tuple:
    """(texts, labels, poem_ids) for train-split poems of authors with at
    least min_poems of them."""
    train = [p for p in poems if splits.get(str(p['poem_id'])) == 'train']
    counts = Counter(p['author'] for p in train)
    kept = [p for p in train if counts[p['author']] >= min_poems]
    return ([p['song_text'] for p in kept], [p['author'] for p in kept],
            [str(p['poem_id']) for p in kept])


def fit(texts: list, labels: list) -> Pipeline:
    model = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None,
                                  ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ('clf', LogisticRegression(C=10.0, max_iter=5000)),
    ])
    return model.fit(texts, labels)


def held_out_accuracy(model: Pipeline, poems: list, splits: dict, authors: set = None) -> dict:
    """Top-1 on val+test poems of the judge's own authors (optionally only
    `authors`) -- the ceiling a real, unseen poem reaches."""
    known = set(model.classes_) if authors is None else set(model.classes_) & set(authors)
    held = [p for p in poems
            if splits.get(str(p['poem_id'])) in ('val', 'test') and p['author'] in known]
    if not held:
        return {'n': 0, 'top1': None}
    pred = model.predict([p['song_text'] for p in held])
    return {'n': len(held), 'top1': float(np.mean([a == p['author'] for a, p in zip(pred, held)]))}


def build(save: bool = True) -> Pipeline:
    poems, splits = load_stripped_songs(), load_splits()
    model = fit(*training_set(poems, splits)[:2])
    if save:
        JUDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        JUDGE_PATH.write_bytes(pickle.dumps(model))
    return model


def load_judge() -> Pipeline:
    global _MODEL
    if _MODEL is None:
        _MODEL = pickle.loads(JUDGE_PATH.read_bytes()) if JUDGE_PATH.exists() else build()
    return _MODEL


def judge_match(text: str, target_author: str, model: Pipeline = None) -> dict:
    model = model or load_judge()
    classes = list(model.classes_)
    if not lines(text) or target_author not in classes:
        return {'judge_target_prob': 0.0, 'judge_top1': 0.0, 'judge_predicted': ''}
    probs = model.predict_proba([text])[0]
    predicted = classes[int(probs.argmax())]
    return {'judge_target_prob': round(float(probs[classes.index(target_author)]), 4),
            'judge_top1': 1.0 if predicted == target_author else 0.0,
            'judge_predicted': predicted}


if __name__ == '__main__':
    from sft_data import select_target_authors

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rebuild', action='store_true', help='refit and overwrite the cache')
    args = parser.parse_args()

    model = build() if args.rebuild or not JUDGE_PATH.exists() else load_judge()
    poems, splits = load_stripped_songs(), load_splits()
    targets = select_target_authors(10)
    report = {
        'n_authors': len(model.classes_),
        'n_train_poems': len(training_set(poems, splits)[0]),
        'held_out_all_authors': held_out_accuracy(model, poems, splits),
        'held_out_target_authors': held_out_accuracy(model, poems, splits, set(targets)),
    }
    EVAL_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
