"""Merging per-target-author part CSVs (from parallel generate_synthetic.py
runs, see scripts/generate_synthetic_parallel.sh) back into the canonical
data/synthetic/synthetic_dataset.csv / synthetic_errors.csv."""

import csv

import merge_synthetic_parts as msp
from generate_synthetic import FIELDNAMES


def _row(poem_id, target_author, sample_index=0, poem='песна'):
    return {**{k: '' for k in FIELDNAMES},
           'source_poem_id': poem_id, 'target_author': target_author,
           'sample_index': sample_index, 'generated_poem': poem}


def _write_dataset(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_dataset(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ── merge_dataset_parts ────────────────────────────────────────────────────────

def test_merge_dataset_parts_combines_rows_from_multiple_parts(tmp_path):
    part_a = tmp_path / 'a.csv'
    part_b = tmp_path / 'b.csv'
    _write_dataset(part_a, [_row('0', 'Ц1')])
    _write_dataset(part_b, [_row('1', 'Ц2')])
    output = tmp_path / 'out.csv'

    result = msp.merge_dataset_parts([part_a, part_b], output)

    rows = _read_dataset(output)
    assert {(r['source_poem_id'], r['target_author']) for r in rows} == {('0', 'Ц1'), ('1', 'Ц2')}
    assert result == {'merged': 2, 'skipped_duplicates': 0}


def test_merge_dataset_parts_skips_rows_already_in_the_output(tmp_path):
    output = tmp_path / 'out.csv'
    _write_dataset(output, [_row('0', 'Ц1')])
    part = tmp_path / 'a.csv'
    _write_dataset(part, [_row('0', 'Ц1'), _row('1', 'Ц1')])

    result = msp.merge_dataset_parts([part], output)

    rows = _read_dataset(output)
    assert len(rows) == 2
    assert result == {'merged': 1, 'skipped_duplicates': 1}


def test_merge_dataset_parts_skips_the_same_key_repeated_across_parts(tmp_path):
    part_a = tmp_path / 'a.csv'
    part_b = tmp_path / 'b.csv'
    _write_dataset(part_a, [_row('0', 'Ц1')])
    _write_dataset(part_b, [_row('0', 'Ц1')])  # same key, e.g. re-run overlap
    output = tmp_path / 'out.csv'

    result = msp.merge_dataset_parts([part_a, part_b], output)

    assert len(_read_dataset(output)) == 1
    assert result == {'merged': 1, 'skipped_duplicates': 1}


def test_merge_dataset_parts_ignores_a_missing_part_file(tmp_path):
    output = tmp_path / 'out.csv'
    result = msp.merge_dataset_parts([tmp_path / 'does_not_exist.csv'], output)
    assert result == {'merged': 0, 'skipped_duplicates': 0}


def test_merge_dataset_parts_is_idempotent(tmp_path):
    part = tmp_path / 'a.csv'
    _write_dataset(part, [_row('0', 'Ц1'), _row('1', 'Ц1')])
    output = tmp_path / 'out.csv'

    msp.merge_dataset_parts([part], output)
    second = msp.merge_dataset_parts([part], output)

    assert len(_read_dataset(output)) == 2
    assert second == {'merged': 0, 'skipped_duplicates': 2}


# ── merge_error_parts ────────────────────────────────────────────────────────

def _error_row(poem_id, target_author, sample_index=0, error='boom'):
    return {'source_poem_id': poem_id, 'source_author': 'А', 'target_author': target_author,
           'sample_index': sample_index, 'error': error}


def test_merge_error_parts_combines_and_dedupes(tmp_path):
    part_a = tmp_path / 'a_errors.csv'
    part_b = tmp_path / 'b_errors.csv'
    with open(part_a, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=msp.ERROR_FIELDNAMES)
        writer.writeheader()
        writer.writerow(_error_row('0', 'Ц1'))
    with open(part_b, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=msp.ERROR_FIELDNAMES)
        writer.writeheader()
        writer.writerow(_error_row('0', 'Ц1'))  # duplicate of part_a
        writer.writerow(_error_row('1', 'Ц2'))

    output = tmp_path / 'errors.csv'
    result = msp.merge_error_parts([part_a, part_b], output)

    with open(output, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert result == {'merged': 2, 'skipped_duplicates': 1}


# ── seed_part_from_main ────────────────────────────────────────────────────────

def test_seed_part_from_main_copies_only_the_matching_authors_rows(tmp_path):
    main = tmp_path / 'main.csv'
    _write_dataset(main, [_row('0', 'Ц1'), _row('1', 'Ц2'), _row('2', 'Ц1')])
    part = tmp_path / 'Ц1.csv'

    msp.seed_part_from_main('Ц1', main, part)

    rows = _read_dataset(part)
    assert {r['source_poem_id'] for r in rows} == {'0', '2'}


def test_seed_part_from_main_writes_a_header_only_file_when_main_is_missing(tmp_path):
    main = tmp_path / 'does_not_exist.csv'
    part = tmp_path / 'Ц1.csv'

    msp.seed_part_from_main('Ц1', main, part)

    assert _read_dataset(part) == []


def test_seeded_part_makes_generate_synthetic_skip_the_authors_done_pairs(tmp_path):
    # The actual point of seeding: a fresh generate_synthetic.py process
    # pointed at the seeded part file must treat the author's prior successes
    # as already done, via the same load_done_keys it always uses.
    from generate_synthetic import load_done_keys

    main = tmp_path / 'main.csv'
    _write_dataset(main, [_row('0', 'Ц1'), _row('1', 'Ц2')])
    part = tmp_path / 'Ц1.csv'

    msp.seed_part_from_main('Ц1', main, part)

    assert load_done_keys(part) == {('0', 'Ц1', 0)}
