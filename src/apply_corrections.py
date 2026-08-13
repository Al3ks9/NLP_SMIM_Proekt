"""Harvest hand corrections from pos_flags.csv and apply them to pos_tagged.csv.

Run after filling in the `correction` / `lemma_fix` / `scope` columns of
`data/pos_flags.csv`. See src/corrections.py for the workflow and scope rules.
"""

import csv
import shutil

import corrections as C

FLAGS = C.DATA / 'pos_flags.csv'
TAGGED = C.DATA / 'pos_tagged.csv'

with open(FLAGS, encoding='utf-8') as f:
    flag_rows = list(csv.DictReader(f))

edited = [r for r in flag_rows
          if r.get('correction', '').strip() or r.get('lemma_fix', '').strip()]
if not edited:
    raise SystemExit(
        f"No edits found in {FLAGS.name}. Fill in the 'correction' column "
        "(a POS tag, or DROP) and/or 'lemma_fix', then run this again.")

# Reject the whole batch on a bad row rather than half-applying it.
problems = [(r, err) for r in edited if (err := C.validate(r))]
if problems:
    print(f"{len(problems)} unusable correction(s) — nothing was applied:\n")
    for r, err in problems:
        print(f"  {r['word']!r} ({r['pos']}): {err}")
    raise SystemExit(1)

# Merge into the durable file. A re-edit of the same word+tag+scope replaces the
# earlier decision instead of stacking a second conflicting rule.
existing = {}
if C.CORRECTIONS_CSV.exists():
    with open(C.CORRECTIONS_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            existing[C.key_for(row, row.get('scope', '').strip().lower())] = row

was = {k: (v['correction'], v['lemma_fix']) for k, v in existing.items()}
for r in edited:
    scope = r['scope'].strip().lower()
    row = {
        'word': r['word'], 'pos': r['pos'],
        'correction': r['correction'].strip(), 'lemma_fix': r['lemma_fix'].strip(),
        'scope': scope, 'author': r['author'], 'song_title': r['song_title'],
        'context': r['context'],
    }
    existing[C.key_for(row, scope)] = row

# Counted over rules, not edited rows: 30 rows correcting гора all collapse into
# the one global rule, and reporting them as 30 updates would be noise.
added = sum(1 for k in existing if k not in was)
updated = sum(1 for k, v in existing.items()
              if k in was and was[k] != (v['correction'], v['lemma_fix']))

with open(C.CORRECTIONS_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=C.FIELDS)
    writer.writeheader()
    writer.writerows(existing.values())
print(f"{len(edited)} edited row(s) -> {added} new rule(s), {updated} changed, "
      f"{len(existing)} in {C.CORRECTIONS_CSV.name}")

# Patch pos_tagged.csv in place, keeping a backup of what we overwrite.
with open(TAGGED, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    tagged = list(reader)

table = C.load()
shutil.copy(TAGGED, TAGGED.with_suffix('.csv.bak'))

out, changed, dropped, touched = [], 0, 0, set()
for row in tagged:
    before = (row['pos'], row['lemma'])
    result = C.apply_to(
        table, row['word'], row['pos'], row['lemma'], row['xpos'], row['feats'],
        row['author'], row['song_title'], '')
    if result is None:
        dropped += 1
        touched.add((row['word'].lower(), before[0]))
        continue
    row['pos'], row['lemma'], row['xpos'], row['feats'] = result
    if (row['pos'], row['lemma']) != before:
        changed += 1
        touched.add((row['word'].lower(), before[0]))
    out.append(row)

with open(TAGGED, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out)

print(f"{TAGGED.name}: {changed} tokens retagged, {dropped} dropped, "
      f"{len(out)} rows remain (backup: {TAGGED.with_suffix('.csv.bak').name})")

# Row-scoped corrections need the sentence text, which pos_tagged.csv does not
# carry, and corrections on tokens classla never gave a content tag to cannot be
# added by patching. Both resolve on the next full tagging run.
KEEP = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
deferred = [r for r in edited
            if r['scope'].strip().lower() == 'row'
            or r['pos'] not in KEEP or r['xpos'] == 'Rg']
if deferred:
    print(f"\n{len(deferred)} correction(s) stored but not applied here — they need "
          "a full re-tag:\n  uv run python src/pos_tag_corpus.py")
    for r in deferred[:10]:
        why = ('row-scoped' if r['scope'].strip().lower() == 'row'
               else 'not a content token until corrected')
        print(f"  {r['word']!r} ({r['pos'] or 'no tag'}) — {why}")
    if len(deferred) > 10:
        print(f"  ... and {len(deferred) - 10} more")
