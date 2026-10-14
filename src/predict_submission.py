"""Frozen v5.2-exp01 inference; no training, label input, or threshold search."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import time

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from prepare_features import FEATURE_QUERY, sql_path
from prepare_content_features import _encoding_table
from soc_threat import LABELS
from soc_threat.content_features import encode_log_content
from soc_threat.hierarchical_features import transform_hierarchical_inputs
from soc_threat.hierarchical_model import HierarchicalContentModel
from soc_threat.anchored_residual_model import AnchoredConflictResidualModel
from train_hierarchical_content import hierarchical_predictions, hierarchical_probabilities
from train_npu_tabular import choose_device

RAW_COLUMNS = ['event_id', 'timestamp', 'pipeline', 'src_ip', 'dst_ip', 'src_port',
               'src_host', 'dst_host', 'username', 'message_sanitized',
               'product_name', 'vendor_name']
TOKEN_LENGTH = 96  # Frozen training recipe, not inferred from test data.
FINAL_CHECKPOINT_SHA256 = '60e2f47a85d0b1f472a8fcf74ea0a8b73df97982817abcd2ebab09141f347080'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def restore_checkpoint(archive: Path, output: Path) -> Path:
    """Check the original checksum and copy one regular member, never extract paths."""
    checksum = Path(str(archive) + '.sha256')
    expected = checksum.read_text('utf-8-sig').split()[0].lower()
    if not re.fullmatch(r'[0-9a-f]{64}', expected) or sha256(archive) != expected:
        raise ValueError('Archive SHA256 mismatch; do not use this backup')
    with tarfile.open(archive, 'r:gz') as bundle:
        matches = [m for m in bundle.getmembers() if m.isfile()
                   and Path(m.name).name == 'model.pt'
                   and Path(m.name).parent.name in {'cr1_anchor_content', 'exp01_anchor_content'}]
        if len(matches) != 1:
            raise ValueError('Expected exactly one exp01 anchor-content checkpoint in the archive')
        temporary = output.with_suffix('.tmp')
        with bundle.extractfile(matches[0]) as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target)
        if output.exists() and sha256(output) != sha256(temporary):
            temporary.unlink()
            raise ValueError('Output contains a different checkpoint; choose a new output directory')
        os.replace(temporary, output)
    return output


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = dict(checkpoint['model_config'])
    anchor_config = dict(config.pop('anchor_config'))
    if (config['residual_input_mode'] != 'anchor' or config['evidence_source'] != 'content'
            or config['max_conflict_gap'] != 2.0):
        raise ValueError('Expected v5.2-exp01: anchor input, content evidence, gap=2')
    if list(checkpoint['labels']) != list(LABELS) or checkpoint['threat_threshold'] != 0.5:
        raise ValueError('Unexpected checkpoint labels or threat threshold')
    if anchor_config.get('content_input_mode') != 'raw':
        raise ValueError('Expected the frozen raw-content anchor')
    anchor = HierarchicalContentModel(**anchor_config)
    model = AnchoredConflictResidualModel(
        anchor=anchor, **config,
        hash_buckets=anchor_config['hash_buckets'],
        content_embedding_dim=anchor_config['content_embedding_dim'],
        content_output_dim=anchor_config['content_output_dim'],
    )
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict_table(raw: pa.Table, model, checkpoint: dict, device: torch.device,
                  batch_size: int, connection) -> pa.Table:
    raw = raw.select(RAW_COLUMNS)
    connection.register('source_data', raw.append_column(
        'label_binary', pa.nulls(len(raw), type=pa.string())))
    frame = connection.execute(FEATURE_QUERY).fetchdf()
    if frame['event_id'].tolist() != raw['event_id'].to_pylist():
        raise ValueError('Feature extraction changed input row order')
    buckets = int(checkpoint['model_config']['anchor_config']['hash_buckets'])
    encodings = [encode_log_content(row, buckets=buckets, max_tokens=TOKEN_LENGTH)
                 for row in raw.select(['event_id', 'pipeline', 'product_name',
                                        'vendor_name', 'message_sanitized']).to_pylist()]
    content = _encoding_table(frame['event_id'].tolist(), encodings, TOKEN_LENGTH)
    for name in content.column_names:
        if name not in {'event_id', 'raw_token_ids', 'field_token_ids'}:
            frame[name] = content[name].to_pandas().to_numpy()
    arrays = transform_hierarchical_inputs(frame, checkpoint['preprocessor'])
    tokens = np.asarray([e.raw_token_ids for e in encodings], dtype=np.int64)
    inputs = [(arrays.metadata_categorical, torch.long),
              (arrays.metadata_numeric, torch.float32),
              (arrays.semantic_categorical, torch.long),
              (arrays.semantic_numeric, torch.float32), (tokens, torch.long)]
    labels, probabilities = [], []
    for start in range(0, len(raw), batch_size):
        end = start + batch_size
        tensors = [torch.as_tensor(values[start:end], dtype=dtype, device=device)
                   for values, dtype in inputs]
        result = model(*tensors, None,
                       torch.as_tensor(arrays.novelty_gate[start:end], device=device),
                       torch.as_tensor(arrays.combo_counts[start:end], device=device))
        threat = result.threat_logits.softmax(1).cpu().numpy()
        subtype = result.subtype_logits.softmax(1).cpu().numpy()
        if not np.isfinite(threat).all() or not np.isfinite(subtype).all():
            raise ValueError('Nonfinite model output')
        predicted, _, _ = hierarchical_predictions(
            threat, subtype, threshold=float(checkpoint['threat_threshold']))
        labels.extend(np.asarray(LABELS)[predicted].tolist())
        probabilities.append(hierarchical_probabilities(threat, subtype))
    probability = np.concatenate(probabilities)
    return pa.table({'event_id': raw['event_id'], 'pred_label': labels,
                     **{'prob_' + name: probability[:, i] for i, name in enumerate(LABELS)}})


def audit_input(connection, path: Path) -> int:
    missing = set(RAW_COLUMNS) - set(pq.ParquetFile(path).schema_arrow.names)
    if missing:
        raise ValueError(f'Missing raw columns: {sorted(missing)}')
    rows, unique, empty = connection.execute(f"""
        SELECT count(*), count(DISTINCT event_id),
        count(*) FILTER (WHERE event_id IS NULL OR trim(event_id)='')
        FROM read_parquet('{sql_path(path)}')""").fetchone()
    if rows == 0 or rows != unique or empty:
        raise ValueError(f'Invalid input IDs: rows={rows}, unique={unique}, empty={empty}')
    return rows


def audit_submission(connection, input_path: Path, csv_path: Path) -> dict:
    connection.execute(f"""CREATE OR REPLACE TEMP VIEW submission AS
        SELECT * FROM read_csv('{sql_path(csv_path)}', header=true, all_varchar=true)""")
    columns = [row[0] for row in connection.execute('DESCRIBE submission').fetchall()]
    if columns != ['event_id', 'pred_label']:
        raise ValueError(f'Invalid submission columns: {columns}')
    rows, unique, invalid = connection.execute("""
        SELECT count(*),count(DISTINCT event_id),count(*) FILTER (
          WHERE event_id IS NULL OR trim(event_id)='' OR pred_label IS NULL
          OR pred_label NOT IN ('benign','malicious','suspicious')) FROM submission
        """).fetchone()
    missing = connection.execute(f"""SELECT count(*) FROM read_parquet('{sql_path(input_path)}') i
        ANTI JOIN submission s USING (event_id)""").fetchone()[0]
    extra = connection.execute(f"""SELECT count(*) FROM submission s
        ANTI JOIN read_parquet('{sql_path(input_path)}') i USING (event_id)""").fetchone()[0]
    if not rows or rows != unique or invalid or missing or extra:
        raise ValueError(f'Invalid submission: {rows=}, {unique=}, {invalid=}, {missing=}, {extra=}')
    return {'rows': rows, 'distinct_event_ids': unique, 'invalid_rows': invalid,
            'missing_ids': missing, 'extra_ids': extra,
            'label_counts': dict(connection.execute(
                'SELECT pred_label,count(*) FROM submission GROUP BY pred_label').fetchall())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive', type=Path)
    source.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.chunk_size, args.batch_size, args.threads) <= 0:
        raise ValueError('Batch sizes and threads must be positive')
    torch.set_num_threads(args.threads)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'res.csv').exists() and not (output / 'manifest.json').exists():
        raise FileExistsError('Unowned res.csv exists; choose a new output directory')
    checkpoint_path = (restore_checkpoint(args.archive, output / 'frozen_model.pt')
                       if args.archive else args.checkpoint)
    if sha256(checkpoint_path) != FINAL_CHECKPOINT_SHA256:
        raise ValueError('This is not the verified final v5.2-exp01 checkpoint; refusing to predict')
    if str(args.device).startswith('npu'):
        import torch_npu  # noqa: F401
    device = choose_device(args.device)
    model, checkpoint = load_model(checkpoint_path, device)
    connection = duckdb.connect()
    connection.execute(f'SET threads TO {args.threads}')
    connection.execute("SET TimeZone='UTC'")
    connection.execute('SET preserve_insertion_order=true')
    rows = audit_input(connection, args.input)
    src = Path(__file__).resolve().parent
    identity = {'input_sha256': sha256(args.input), 'checkpoint_sha256': sha256(checkpoint_path),
                'rows': rows, 'chunk_size': args.chunk_size, 'batch_size': args.batch_size,
                'token_length': TOKEN_LENGTH, 'device': str(device), 'torch': str(torch.__version__),
                'source_hashes': {str(p.relative_to(src)): sha256(p)
                                  for p in sorted(src.rglob('*.py'))}}
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text('utf-8')) != identity:
        raise ValueError('Input/model/code/settings changed; use a NEW output directory')
    write_json(manifest, identity)
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest().encode()
    shards = output / 'prediction_shards'
    shards.mkdir(exist_ok=True)
    expected_paths, completed = [], 0
    started = time.monotonic()
    print(json.dumps({'stage': 'loaded', 'model': 'v5.2-exp01', 'rows': rows,
                      'device': str(device), 'checkpoint_sha256': identity['checkpoint_sha256']}), flush=True)
    for index, batch in enumerate(pq.ParquetFile(args.input).iter_batches(
            batch_size=args.chunk_size, columns=RAW_COLUMNS)):
        raw = pa.Table.from_batches([batch])
        path = shards / f'part-{index:06d}.parquet'
        expected_paths.append(str(path.resolve()))
        reused = False
        if path.exists():
            try:
                cached = pq.read_table(path)
                reused = ((cached.schema.metadata or {}).get(b'run_id') == run_id
                          and cached['event_id'].equals(raw['event_id'])
                          and set(cached['pred_label'].to_pylist()) <= set(LABELS))
            except (OSError, pa.ArrowException, KeyError):
                reused = False
        if not reused:
            result = predict_table(raw, model, checkpoint, device, args.batch_size, connection)
            result = result.replace_schema_metadata({b'run_id': run_id})
            temporary = path.with_suffix('.tmp')
            pq.write_table(result, temporary, compression='zstd')
            os.replace(temporary, path)
        completed += len(raw)
        print(json.dumps({'stage': 'predict', 'rows': completed, 'total': rows,
                          'reused': reused, 'seconds': round(time.monotonic()-started, 2)}), flush=True)
    connection.read_parquet(expected_paths).create_view('predictions')
    temporary_csv = output / 'res.tmp.csv'
    connection.execute(f"""COPY (SELECT event_id,pred_label FROM predictions)
        TO '{sql_path(temporary_csv)}' (FORMAT CSV, HEADER true)""")
    audit = audit_submission(connection, args.input, temporary_csv)
    connection.close()
    csv_path = output / 'res.csv'
    os.replace(temporary_csv, csv_path)
    audit.update({'model': 'v5.2-exp01', 'checkpoint_sha256': identity['checkpoint_sha256'],
                  'res_sha256': sha256(csv_path), 'output': str(csv_path.resolve()),
                  'has_ground_truth': False})
    write_json(output / 'submission_audit.json', audit)
    print(json.dumps({'stage': 'complete', **audit}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
