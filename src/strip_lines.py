"""
Regenerate data/stripped_songs.csv from the raw songs.csv, assigning poem_id by row
order -- see docs/superpowers/specs/2026-08-19-poem-id-migration-design.md.

Not part of CLAUDE.md's regular pipeline order (data/stripped_songs.csv is the
checked-in base input that order starts from) -- only needed if songs.csv itself
changes and stripped_songs.csv must be regenerated from scratch. Uses the csv module
(the previous version read songs.csv line-by-line without parsing it as CSV, which
can't survive multi-line quoted fields safely and can't insert a column).
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
in_file = ROOT / 'songs.csv'
out_file = ROOT / 'data' / 'stripped_songs.csv'

with open(in_file, encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# Filter out rows with missing required fields (data quality issues in raw file)
# Also remove None key which can appear in malformed CSV rows
valid_rows = []
for r in rows:
    if None in r:  # Skip rows with extra/malformed columns
        continue
    if r.get('author') and r.get('song_title') and r.get('song_text'):
        valid_rows.append(r)

for poem_id, row in enumerate(valid_rows):
    row['author'] = row['author'].strip()
    row['song_title'] = row['song_title'].strip()
    # Strip each physical line of song_text individually, matching the original
    # script's per-line behavior, rather than only the field's outer whitespace.
    row['song_text'] = '\n'.join(line.strip() for line in row['song_text'].split('\n'))
    row['poem_id'] = poem_id

with open(out_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['poem_id', 'author', 'song_title', 'song_text'])
    writer.writeheader()
    writer.writerows(valid_rows)

print(f'Wrote {len(valid_rows)} poems to {out_file}')
