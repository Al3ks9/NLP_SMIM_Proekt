import csv
from collections import Counter
from pathlib import Path
import spacy

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

MIN_POEMS = 5
KEEP_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

nlp = spacy.load('mk_core_news_lg')

with open(DATA / 'stripped_songs.csv') as f:
    rows = list(csv.DictReader(f))

author_counts = Counter(r['author'] for r in rows)
qualified = {a for a, c in author_counts.items() if c >= MIN_POEMS}
print(f"Tagging {len(qualified)} authors ({sum(1 for r in rows if r['author'] in qualified)} poems)")

results = []
for i, row in enumerate(rows):
    if row['author'] not in qualified:
        continue
    doc = nlp(row['song_text'])
    for token in doc:
        if token.pos_ in KEEP_POS and not token.is_space and not token.is_punct:
            results.append({
                'author': row['author'],
                'song_title': row['song_title'],
                'word': token.text.lower(),
                'lemma': token.lemma_.lower(),
                'pos': token.pos_,
            })
    if (i + 1) % 100 == 0:
        print(f"  processed {i + 1}/{len(rows)} rows...")

with open(DATA / 'pos_tagged.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['author', 'song_title', 'word', 'lemma', 'pos'])
    writer.writeheader()
    writer.writerows(results)

print(f"Done. {len(results)} tagged tokens saved to pos_tagged.csv")

# Quick sanity check
pos_dist = Counter(r['pos'] for r in results)
print("POS distribution:", dict(pos_dist))
