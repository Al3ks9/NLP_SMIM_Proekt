"""
One-time migration: add a stable poem_id to stripped_songs.csv and (retroactively)
to pos_corrections.csv.

poem_id is a plain sequential integer, assigned by row order in stripped_songs.csv --
see docs/superpowers/specs/2026-08-19-poem-id-migration-design.md for why (not a UUID,
not partial coverage). Idempotent: running it again on an already-migrated file prints
a message and does nothing.

Run once:
    uv run python src/add_poem_ids.py
"""

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'


def resolve_correction_poem_id(row: dict, ids_by_title: dict):
    """
    The single poem_id a poem/row-scoped correction's (author, song_title) resolves
    to, or None for a global-scope row (which never used author/song_title to begin
    with). Raises SystemExit -- refuses rather than guessing -- if the title matches
    zero or more than one poem; there's no way to recover which physical poem a
    hand-written correction meant in that case.
    """
    scope = row['scope'].strip().lower()
    if scope not in ('poem', 'row'):
        return None
    matches = ids_by_title.get((row['author'], row['song_title']), [])
    if len(matches) != 1:
        raise SystemExit(
            f"Cannot migrate: {row['word']!r} ({row['pos']}) is scope={scope!r} for "
            f"author={row['author']!r} song_title={row['song_title']!r}, which matches "
            f"{len(matches)} poems. Disambiguate this correction by hand before migrating."
        )
    return matches[0]


def migrate_stripped_songs() -> dict:
    """Add poem_id to stripped_songs.csv. Returns {(author, song_title): [poem_id, ...]}
    for migrate_pos_corrections() to resolve corrections against."""
    path = DATA / 'stripped_songs.csv'
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames[0] == 'poem_id':
        print(f'{path.name} already has poem_id -- skipping')
        ids_by_title = {}
        for row in rows:
            ids_by_title.setdefault((row['author'], row['song_title']), []).append(
                int(row['poem_id']))
        return ids_by_title

    ids_by_title = {}
    for poem_id, row in enumerate(rows):
        row['poem_id'] = poem_id
        ids_by_title.setdefault((row['author'], row['song_title']), []).append(poem_id)

    shutil.copy(path, path.with_suffix('.csv.bak'))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['poem_id'] + fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{path.name}: added poem_id to {len(rows)} rows (backup: {path.with_suffix(".csv.bak").name})')
    return ids_by_title


def migrate_pos_corrections(ids_by_title: dict) -> None:
    """Add poem_id to pos_corrections.csv, resolving each poem/row-scoped row's
    (author, song_title) via resolve_correction_poem_id()."""
    path = DATA / 'pos_corrections.csv'
    if not path.exists():
        print(f'{path.name} does not exist -- nothing to migrate')
        return

    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if 'poem_id' in fieldnames:
        print(f'{path.name} already has poem_id -- skipping')
        return

    for row in rows:
        poem_id = resolve_correction_poem_id(row, ids_by_title)
        row['poem_id'] = poem_id if poem_id is not None else ''

    out_fields = ['word', 'pos', 'correction', 'lemma_fix', 'scope', 'poem_id',
                 'author', 'song_title', 'context']
    shutil.copy(path, path.with_suffix('.csv.bak'))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{path.name}: added poem_id to {len(rows)} rows (backup: {path.with_suffix(".csv.bak").name})')


if __name__ == '__main__':
    ids_by_title = migrate_stripped_songs()
    migrate_pos_corrections(ids_by_title)
