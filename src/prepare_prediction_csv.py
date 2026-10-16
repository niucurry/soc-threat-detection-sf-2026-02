"""Validate raw CSV and preserve its values/order in a shared inference parquet."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from soc_threat.refit_utils import sha256, write_json

RAW_COLUMNS = ['event_id', 'timestamp', 'pipeline', 'src_ip', 'dst_ip', 'src_port',
               'src_host', 'dst_host', 'username', 'message_sanitized',
               'product_name', 'vendor_name']
SCHEMA = pa.schema([(c, pa.float64() if c == 'timestamp' else pa.string()) for c in RAW_COLUMNS])


def prepare(input_path: Path, output_dir: Path) -> dict:
    input_path = input_path.resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / 'test_input.parquet'
    manifest = output_dir / 'input_audit.json'
    temporary = output_dir / 'test_input.parquet.tmp'
    if input_path in {p.resolve() for p in [target, manifest, temporary]}:
        raise ValueError('Input and output paths must be different')
    if input_path.suffix.lower() != '.csv':
        raise ValueError('This converter accepts raw .csv only; do not rename parquet files')
    identity = {'input_sha256': sha256(input_path), 'converter_sha256': sha256(Path(__file__)),
                'schema_version': 1}
    if manifest.exists():
        previous = json.loads(manifest.read_text('utf-8'))
        if previous['identity'] != identity:
            raise ValueError('Input/converter changed; use a NEW output directory')
        if not target.is_file() or sha256(target) != previous['parquet_sha256']:
            raise ValueError('Prepared input is missing/corrupt; use a NEW output directory')
        return dict(previous, reused=True)
    if target.exists():
        raise FileExistsError('Unowned prepared input exists; use a NEW output directory')
    # Arrow's CSV parser misreads some unquoted log fields containing NUL bytes.
    # Python's CSV parser preserves them, along with embedded newlines and quotes.
    # Never delete control characters or infer identifiers/log text as null markers.
    old_field_limit=csv.field_size_limit()
    csv.field_size_limit(64*1024*1024)
    try:
        with input_path.open('r',encoding='utf-8-sig',newline='') as stream:
            reader=csv.reader(stream,strict=True)
            header=next(reader,[])
            if len(header)!=len(RAW_COLUMNS) or set(header)!=set(RAW_COLUMNS):
                raise ValueError(f'Expected exactly the 12 raw columns, no labels; received {header}')
            indices=[header.index(c) for c in RAW_COLUMNS]
            with pq.ParquetWriter(temporary,SCHEMA,compression='zstd') as writer:
                columns=[[] for _ in RAW_COLUMNS]
                for record_number,values in enumerate(reader,1):
                    if len(values)!=len(RAW_COLUMNS):
                        raise ValueError(f'Invalid CSV record {record_number}: expected 12 columns, got {len(values)}')
                    ordered=[values[i] for i in indices]
                    try:
                        ordered[1]=float(ordered[1])
                    except ValueError as exc:
                        raise ValueError(f'Invalid raw input timestamp at CSV record {record_number}') from exc
                    for dest,value in zip(columns,ordered):dest.append(value)
                    if len(columns[0])>=10000:
                        writer.write_table(pa.Table.from_arrays([pa.array(v,type=f.type) for v,f in zip(columns,SCHEMA)],schema=SCHEMA))
                        columns=[[] for _ in RAW_COLUMNS]
                if columns[0]:
                    writer.write_table(pa.Table.from_arrays([pa.array(v,type=f.type) for v,f in zip(columns,SCHEMA)],schema=SCHEMA))
        with duckdb.connect() as db:
            db.read_parquet(str(temporary)).create_view('raw')
            n, unique, invalid_ids, invalid_time = db.execute('''SELECT count(*), count(DISTINCT event_id),
                count(*) FILTER (WHERE event_id IS NULL OR trim(event_id)='' OR event_id<>trim(event_id)),
                count(*) FILTER (WHERE timestamp IS NULL OR NOT isfinite(timestamp)) FROM raw''').fetchone()
            if not n or unique != n or invalid_ids or invalid_time:
                raise ValueError(f'Invalid raw input: {n=}, {unique=}, {invalid_ids=}, {invalid_time=}')
            min_time,max_time = db.execute('SELECT min(timestamp),max(timestamp) FROM raw').fetchone()
            # Original model contract is Unix seconds, not ms/us/ns or date strings.
            if min_time < 0 or max_time >= 100000000000:
                raise ValueError('Expected Unix timestamp in seconds; no automatic unit conversion')
            pipelines = dict(db.execute('SELECT pipeline,count(*) FROM raw GROUP BY 1 ORDER BY 1').fetchall())
        if sha256(input_path) != identity['input_sha256']:
            raise ValueError('CSV changed while being read; rerun with an immutable input')
        report = {'identity': identity, 'source': str(input_path), 'rows': n, 'distinct_event_ids': unique,
                  'invalid_ids': invalid_ids, 'invalid_timestamps': invalid_time,
                  'timestamp_min': min_time, 'timestamp_max': max_time, 'pipelines': pipelines,
                  'columns': RAW_COLUMNS, 'text_empty_fields': 'preserved as empty strings',
                  'timestamp_type': 'float64 Unix seconds; no rounding or unit conversion',
                  'parquet_sha256': sha256(temporary), 'has_ground_truth': False}
        os.replace(temporary, target)
        write_json(manifest, report)
        return dict(report, reused=False)
    finally:
        csv.field_size_limit(old_field_limit)
        # Only this function's unpublished intermediate, never the source or a finished output.
        temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps({'stage':'input_ready', **prepare(args.input,args.output_dir)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
