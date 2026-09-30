"""
Merge per-target-author part CSVs back into the canonical
data/synthetic/synthetic_dataset.csv / synthetic_errors.csv.

Companion to scripts/generate_synthetic_parallel.sh: running one
generate_synthetic.py process per target author in parallel means each must
write to its own --output-path/--errors-path (concurrent appends to one CSV
aren't safe once a row -- a generated poem -- can exceed the OS's atomic
write size). This module does the two things that split needs:

- seed_part_from_main: before a per-author process starts, copy that
  author's already-done rows from the main dataset into its part file, so
  its own load_done_keys() (unchanged, unaware of parallelism) correctly
  skips pairs a previous non-parallel run already generated.
- merge_dataset_parts / merge_error_parts: after all parts finish, fold them
  back into the main files, deduping by (source_poem_id, target_author,
  sample_index) so re-running a merge (e.g. after adding more parts) never
  duplicates a row already present.
"""

import argparse
import csv
from pathlib import Path

from generate_synthetic import ERRORS_PATH, FIELDNAMES, OUTPUT_PATH, _open_append, load_done_keys

ROOT = Path(__file__).resolve().parent.parent

ERROR_FIELDNAMES = ['source_poem_id', 'source_author', 'target_author', 'sample_index', 'error']


def _error_key(row: dict) -> tuple:
    return (row['source_poem_id'], row['target_author'], row['sample_index'])


def _load_error_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, encoding='utf-8') as f:
        return {_error_key(r) for r in csv.DictReader(f)}


def _merge(part_paths: list, output_path: Path, fieldnames: list,
          done: set, key_fn) -> dict:
    out_f, writer = _open_append(output_path, fieldnames)
    merged = skipped = 0
    try:
        for part_path in part_paths:
            part_path = Path(part_path)
            if not part_path.exists():
                continue
            with open(part_path, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    key = key_fn(row)
                    if key in done:
                        skipped += 1
                        continue
                    done.add(key)
                    writer.writerow(row)
                    merged += 1
    finally:
        out_f.close()
    return {'merged': merged, 'skipped_duplicates': skipped}


def merge_dataset_parts(part_paths: list, output_path: Path = OUTPUT_PATH) -> dict:
    output_path = Path(output_path)
    done = load_done_keys(output_path)
    key_fn = lambda row: (row['source_poem_id'], row['target_author'], int(row['sample_index']))
    return _merge(part_paths, output_path, FIELDNAMES, done, key_fn)


def merge_error_parts(part_paths: list, output_path: Path = ERRORS_PATH) -> dict:
    output_path = Path(output_path)
    done = _load_error_keys(output_path)
    return _merge(part_paths, output_path, ERROR_FIELDNAMES, done, _error_key)


def seed_part_from_main(target_author: str, main_path: Path, part_path: Path) -> int:
    """Write target_author's already-done rows from main_path into a fresh
    part_path, so a per-author generate_synthetic.py run pointed at part_path
    skips them via its own unmodified load_done_keys(). Always (re)creates
    part_path, even with zero matching rows, so a stale part from an earlier
    seed never lingers."""
    main_path = Path(main_path)
    rows = []
    if main_path.exists():
        with open(main_path, encoding='utf-8') as f:
            rows = [r for r in csv.DictReader(f) if r['target_author'] == target_author]

    part_path = Path(part_path)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    with open(part_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def paths_for_split(split: str) -> dict:
    """Where the parallel generator writes for a split. 'val' (the Gemma
    teacher comparison run) gets its own main file and parts dir, so a val row
    can never be merged into synthetic_dataset.csv, which train_sft.py reads."""
    synthetic = OUTPUT_PATH.parent
    if split == 'train':
        main, errors, parts = OUTPUT_PATH, ERRORS_PATH, synthetic / 'parts'
    elif split == 'val':
        main, errors = synthetic / 'teacher_val.csv', synthetic / 'teacher_val_errors.csv'
        parts = synthetic / 'parts_val'
    else:
        raise ValueError(f'no synthetic output location for split {split!r}')
    return {'main': main, 'errors': errors, 'parts_dir': parts,
            'error_parts_dir': parts / 'errors', 'log_dir': parts / 'logs'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    seed_p = subparsers.add_parser('seed', help="seed one author's part file from the main dataset")
    seed_p.add_argument('--author', required=True)
    seed_p.add_argument('--main', type=Path, default=OUTPUT_PATH)
    seed_p.add_argument('--part', type=Path, required=True)

    merge_p = subparsers.add_parser('merge', help='merge part files back into the main dataset/errors')
    merge_p.add_argument('--kind', choices=['dataset', 'errors'], default='dataset')
    merge_p.add_argument('--output-path', type=Path, default=None)
    merge_p.add_argument('parts', nargs='+', type=Path)

    paths_p = subparsers.add_parser('paths', help='print KEY=path lines for a split (shell use)')
    paths_p.add_argument('--split', default='train')

    args = parser.parse_args()

    if args.command == 'paths':
        for key, path in paths_for_split(args.split).items():
            print(f'{key.upper()}={path.relative_to(ROOT)}')
    elif args.command == 'seed':
        n = seed_part_from_main(args.author, args.main, args.part)
        print(f'seeded {n} already-done row(s) for {args.author!r} into {args.part}')
    elif args.command == 'merge':
        if args.kind == 'dataset':
            result = merge_dataset_parts(args.parts, args.output_path or OUTPUT_PATH)
        else:
            result = merge_error_parts(args.parts, args.output_path or ERRORS_PATH)
        print(f"merged={result['merged']} skipped_duplicates={result['skipped_duplicates']}")
