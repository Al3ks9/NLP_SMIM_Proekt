import csv
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

MIN_POEMS = 5
TOP_N = 20

with open('stripped_songs.csv') as f:
    rows = list(csv.DictReader(f))

# Group poems by author
author_poems = defaultdict(list)
for r in rows:
    author_poems[r['author']].append(r['song_text'])

# Filter to authors with enough poems
qualified = {a: poems for a, poems in author_poems.items() if len(poems) >= MIN_POEMS}
print(f"Authors with >={MIN_POEMS} poems: {len(qualified)}")

authors = sorted(qualified.keys())
# One document per author = all their poems concatenated
documents = [' '.join(qualified[a]) for a in authors]

def tokenize(text):
    # Lowercase, keep only Cyrillic and basic Latin letters
    text = text.lower()
    return re.findall(r'[а-шѓќўџa-z]+', text)

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    token_pattern=None,
    max_df=0.85,   # drop words present in >85% of author corpora (function words)
    min_df=2,      # drop words appearing in only one author's corpus
    sublinear_tf=True,
)

tfidf_matrix = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(vocab)}")
print()

results = []
for i, author in enumerate(authors):
    row = tfidf_matrix[i].toarray().flatten()
    top_idx = np.argsort(row)[::-1][:TOP_N]
    top_words = [(vocab[j], round(float(row[j]), 4)) for j in top_idx if row[j] > 0]
    results.append((author, top_words))
    print(f"{author} ({len(qualified[author])} poems)")
    print("  " + ", ".join(f"{w}({s})" for w, s in top_words[:10]))
    print()

# Save full results to CSV
with open('tfidf_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['author', 'rank', 'word', 'tfidf_score'])
    for author, words in results:
        for rank, (word, score) in enumerate(words, 1):
            writer.writerow([author, rank, word, score])

print("Saved tfidf_results.csv")
