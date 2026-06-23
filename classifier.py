import csv
import pickle
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

MIN_POEMS = 5

# Load corpus
songs = []
with open('stripped_songs.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        songs.append(row)

# Filter to eligible authors
author_counts = Counter(r['author'] for r in songs)
eligible = {a for a, c in author_counts.items() if c >= MIN_POEMS}

texts, labels = [], []
for row in songs:
    if row['author'] in eligible:
        texts.append(row['song_text'])
        labels.append(row['author'])

print(f"Dataset: {len(texts)} poems, {len(eligible)} authors")

# Pipeline: char n-grams (2-4) + word unigrams, LinearSVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        sublinear_tf=True,
        min_df=2,
        max_features=50000,
    )),
    ('clf', LinearSVC(max_iter=2000, C=1.0)),
])

# 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(pipeline, texts, labels, cv=cv,
                         scoring=['accuracy', 'f1_macro'], return_train_score=False)

print(f"\n5-fold CV results:")
print(f"  Accuracy:   {results['test_accuracy'].mean():.3f} ± {results['test_accuracy'].std():.3f}")
print(f"  Macro F1:   {results['test_f1_macro'].mean():.3f} ± {results['test_f1_macro'].std():.3f}")

# Train final model on all data (calibrated for probability estimates)
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        sublinear_tf=True,
        min_df=2,
        max_features=50000,
    )),
    ('clf', CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=5)),
])
final_pipeline.fit(texts, labels)

# Save model
with open('classifier.pkl', 'wb') as f:
    pickle.dump(final_pipeline, f)
print("\nModel saved to classifier.pkl")

# Per-author report on last fold for reference
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("\nPer-author F1 (held-out 20%):")
print(classification_report(y_test, y_pred, zero_division=0))


def predict_author(text: str) -> tuple[str, float]:
    """Return (predicted_author, confidence) for a given poem text."""
    with open('classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    idx = int(np.argmax(proba))
    return classes[idx], round(float(proba[idx]), 4)


if __name__ == '__main__':
    # Quick sanity check
    sample = texts[0]
    author, conf = predict_author(sample)
    print(f"\nSanity check — predicted: '{author}' ({conf:.1%}), actual: '{labels[0]}'")
