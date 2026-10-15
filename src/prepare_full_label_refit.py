"""Prepare source-safe, resumable features and a grouped development comparison.

Only the three labeled development files are accepted. No test-set input exists.
Group membership is a deterministic content heuristic, NOT a label feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from predict_submission import RAW_COLUMNS, TOKEN_LENGTH, audit_input, encode_raw_batch
from prepare_features import sql_path
from soc_threat import LABELS
from soc_threat.refit_utils import bind_manifest, sha256, source_hashes, verify_artifacts, write_json

DATE = re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?', re.I)
SPACE = re.compile(r'\s+')
OUTPUTS = ['control_train.parquet', 'expanded_train.parquet', 'holdout.parquet', 'full_train.parquet']


def group_key(row: dict) -> str:
    """Keep same-source-family repeated bodies together; never read labels or IDs.

    Normalize whitespace/explicit timestamps, but retain IPs, users, ports, paths,
    actions and event codes. Empty bodies fall back to all non-ID/time fields.
    This intentionally does not claim to find every possible near-duplicate.
    """
    text = SPACE.sub(' ', DATE.sub('<TIME>', str(row.get('message_sanitized') or ''))).strip()
    if text:
        payload = [str(row.get(k) or '').strip() for k in ('pipeline', 'vendor_name', 'product_name')]
        payload.append(text)
    else:
        payload = [str(row.get(k) or '').strip() for k in RAW_COLUMNS if k not in ('event_id', 'timestamp')]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def audit_labels(db, relation: str) -> None:
    counts = dict(db.execute(f'SELECT label_binary,count(*) FROM {relation} GROUP BY 1').fetchall())
    if set(counts) - set(LABELS) or not counts:
        raise ValueError(f'Invalid labels in {relation}: {counts}')


def validate_sources(db, data_dir: Path) -> dict:
    train = data_dir / 'train.parquet'
    valid = data_dir / 'valid_input.parquet'
    answer = data_dir / 'valid_answer_private.parquet'
    sizes = {'train': audit_input(db, train), 'valid': audit_input(db, valid)}
    if 'label_binary' in pq.ParquetFile(valid).schema_arrow.names:
        raise ValueError('valid_input must not contain an embedded label column')
    for name, path in [('train_labels', train), ('answers', answer)]:
        db.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT event_id,label_binary FROM read_parquet('{sql_path(path)}')")
        audit_labels(db, name)
    n, unique, empty = db.execute("SELECT count(*),count(DISTINCT event_id),count(*) FILTER(WHERE event_id IS NULL OR trim(event_id)='') FROM answers").fetchone()
    if n != unique or empty or n != sizes['valid']:
        raise ValueError('Validation answers have missing/duplicate/extra IDs')
    missing = db.execute(f"SELECT count(*) FROM read_parquet('{sql_path(valid)}') i ANTI JOIN answers a USING(event_id)").fetchone()[0]
    if missing:
        raise ValueError('Validation answer IDs do not exactly match validation input')
    return sizes


def prepare_source(db, path: Path, source: str, output: Path, run_id: str, shard_size: int) -> list[Path]:
    directory = output / 'shards' / source
    directory.mkdir(parents=True, exist_ok=True)
    columns = RAW_COLUMNS + (['label_binary'] if source == 'train' else [])
    paths, done = [], 0
    started = time.monotonic()
    for index, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=shard_size, columns=columns)):
        raw = pa.Table.from_batches([batch])
        path_out = directory / f'part-{index:06d}.parquet'
        marker = path_out.with_suffix('.json')
        paths.append(path_out)
        reusable = False
        if marker.exists() and path_out.exists():
            info = json.loads(marker.read_text('utf-8'))
            reusable = (info.get('run_id') == run_id and info.get('rows') == len(raw)
                        and info.get('sha256') == sha256(path_out))
        if not reusable:
            if source == 'train':
                labels = raw['label_binary'].to_pylist()
            else:
                db.register('batch_ids', pa.table({'event_id': raw['event_id'], 'pos': range(len(raw))}))
                labels = [r[0] for r in db.execute('SELECT a.label_binary FROM batch_ids i JOIN answers a USING(event_id) ORDER BY i.pos').fetchall()]
            if len(labels) != len(raw):
                raise ValueError('Validation label join changed row count')
            frame, tokens = encode_raw_batch(raw, db, buckets=65536)
            ids = frame['event_id'].astype(str).tolist()
            frame['original_event_id'] = ids
            frame['event_id'] = [source + ':' + value for value in ids]
            frame['source_split'] = source
            frame['label_binary'] = labels
            frame['group_key'] = [group_key(row) for row in raw.select(RAW_COLUMNS).to_pylist()]
            table = pa.Table.from_pandas(frame, preserve_index=False)
            table = table.append_column('raw_token_ids', pa.FixedSizeListArray.from_arrays(
                pa.array(tokens.astype(np.uint16).reshape(-1)), TOKEN_LENGTH))
            temporary = path_out.with_suffix('.tmp')
            pq.write_table(table, temporary, compression='zstd')
            os.replace(temporary, path_out)
            write_json(marker, {'run_id': run_id, 'rows': len(raw), 'sha256': sha256(path_out)})
        done += len(raw)
        print(json.dumps({'stage': 'refit_features', 'source': source, 'rows': done,
                          'reused': reusable, 'seconds': round(time.monotonic()-started, 2)}), flush=True)
    return paths


def make_splits(db, paths: list[Path], output: Path, fraction: float, seed: int) -> dict:
    db.read_parquet([str(p.resolve()) for p in paths]).create_view('all_features', replace=True)
    # Stratify group counts by their entire label set, not by individual rows.
    # A group with conflicting labels remains intact and is reported below.
    db.execute('''CREATE OR REPLACE TEMP TABLE valid_groups AS
        SELECT group_key,string_agg(DISTINCT label_binary, ',' ORDER BY label_binary) label_set,
        count(*) n FROM all_features WHERE source_split='valid' GROUP BY group_key''')
    db.execute(f'''CREATE OR REPLACE TEMP TABLE holdout_groups AS
        SELECT group_key FROM (
            SELECT *,row_number() OVER(PARTITION BY label_set ORDER BY md5('{seed}:'||group_key)) rank,
            count(*) OVER(PARTITION BY label_set) total FROM valid_groups
        ) WHERE rank <= CASE WHEN total<2 THEN 0
          ELSE least(total-1,greatest(1,floor(total*{fraction}))) END''')
    queries = {
        'control_train.parquet': "SELECT f.* FROM all_features f ANTI JOIN holdout_groups h USING(group_key) WHERE source_split='train'",
        'expanded_train.parquet': 'SELECT f.* FROM all_features f ANTI JOIN holdout_groups h USING(group_key)',
        'holdout.parquet': "SELECT f.* FROM all_features f JOIN holdout_groups h USING(group_key) WHERE source_split='valid'",
        'full_train.parquet': 'SELECT * FROM all_features',
    }
    report = {'splits': {}, 'holdout_group_fraction': fraction, 'split_seed': seed}
    for name, query in queries.items():
        temporary = output / (name + '.tmp')
        # Never expose a partially written Parquet as a completed dataset.
        db.execute(f"COPY ({query}) TO '{sql_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)")
        n, unique = db.execute(f"SELECT count(*),count(DISTINCT event_id) FROM read_parquet('{sql_path(temporary)}')").fetchone()
        counts = dict(db.execute(f"SELECT label_binary,count(*) FROM read_parquet('{sql_path(temporary)}') GROUP BY 1").fetchall())
        if n != unique or set(counts) != set(LABELS):
            raise ValueError(f'{name}: require unique source-prefixed IDs and all three classes, got {counts}')
        os.replace(temporary, output / name)
        report['splits'][name] = {'rows': n, 'labels': counts}
    report['excluded_train_rows_matching_holdout'] = db.execute("SELECT count(*) FROM all_features JOIN holdout_groups USING(group_key) WHERE source_split='train'").fetchone()[0]
    report['group_label_conflicts'] = db.execute('SELECT count(*) FROM (SELECT group_key FROM all_features GROUP BY group_key HAVING count(DISTINCT label_binary)>1)').fetchone()[0]
    report['largest_valid_group_rows'] = db.execute('SELECT max(n) FROM valid_groups').fetchone()[0]
    for name in ['control_train.parquet', 'expanded_train.parquet']:
        overlap = db.execute(f"SELECT count(*) FROM read_parquet('{sql_path(output/name)}') JOIN holdout_groups USING(group_key)").fetchone()[0]
        if overlap:
            raise ValueError('Group leakage into development training')
    report['group_overlap_with_holdout'] = 0
    report['holdout_row_fraction'] = report['splits']['holdout.parquet']['rows'] / db.execute("SELECT count(*) FROM all_features WHERE source_split='valid'").fetchone()[0]
    report['note'] = ('Development holdout only; historical validation was already used in development. '
                      'Grouping removes repeated normalized bodies, not every near-duplicate. '
                      'Group fraction is NOT guaranteed to equal row fraction. All labels remain in full_train.')
    report['artifact_sha256'] = {name: sha256(output/name) for name in OUTPUTS}
    write_json(output / 'split_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--holdout-fraction', type=float, default=0.2)
    parser.add_argument('--split-seed', type=int, default=20261015)
    parser.add_argument('--shard-size', type=int, default=20000)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.holdout_fraction < 1 or min(args.shard_size, args.threads) < 1:
        raise ValueError('Invalid split or batch settings')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {n: args.data_dir/n for n in ['train.parquet', 'valid_input.parquet', 'valid_answer_private.parquet']}
    identity = {'format': 1, 'files': {n: sha256(p) for n,p in files.items()},
                'code': source_hashes(), 'split_seed': args.split_seed,
                'holdout_fraction': args.holdout_fraction, 'shard_size': args.shard_size,
                'duckdb': duckdb.__version__, 'pyarrow': pa.__version__}
    run_id = bind_manifest(args.output_dir / 'input_manifest.json', identity)
    report_path = args.output_dir / 'split_report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text('utf-8'))
        if verify_artifacts(args.output_dir, OUTPUTS, report):
            print(json.dumps({'stage': 'prepared_reused', **report}, ensure_ascii=False), flush=True)
            return
    with duckdb.connect() as db:
        db.execute(f'SET threads={args.threads}')
        db.execute("SET TimeZone='UTC'")
        db.execute('SET preserve_insertion_order=true')
        sizes = validate_sources(db, args.data_dir)
        print(json.dumps({'stage': 'input_audit', 'rows': sizes}), flush=True)
        paths = []
        for source, filename in [('train','train.parquet'), ('valid','valid_input.parquet')]:
            paths += prepare_source(db, files[filename], source, args.output_dir, run_id, args.shard_size)
        report = make_splits(db, paths, args.output_dir, args.holdout_fraction, args.split_seed)
        print(json.dumps({'stage': 'prepared', **report}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
