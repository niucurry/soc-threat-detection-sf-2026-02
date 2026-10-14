from pathlib import Path
import sys

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from predict_submission import RAW_COLUMNS, audit_input, audit_submission


def raw_file(tmp_path, ids):
    path = tmp_path / 'input.parquet'
    pq.write_table(pa.table({name: ids if name == 'event_id' else [None]*len(ids)
                            for name in RAW_COLUMNS}), path)
    return path


def test_submission_exact_coverage_in_any_order(tmp_path):
    path = raw_file(tmp_path, ['a', 'b', 'c'])
    csv = tmp_path / 'res.csv'
    csv.write_text('event_id,pred_label\nc,suspicious\na,benign\nb,malicious\n')
    with duckdb.connect() as db:
        assert audit_input(db, path) == 3
        assert audit_submission(db, path, csv)['missing_ids'] == 0


@pytest.mark.parametrize('rows', [
    'a,benign\na,benign\n', 'a,benign\n', 'a,benign\nx,malicious\n',
    'a,unknown\nb,benign\n', 'a,benign\nb,\n'])
def test_reject_invalid_submissions(tmp_path, rows):
    path = raw_file(tmp_path, ['a', 'b'])
    csv = tmp_path / 'res.csv'
    csv.write_text('event_id,pred_label\n' + rows)
    with duckdb.connect() as db, pytest.raises(ValueError):
        audit_submission(db, path, csv)


def test_reject_extra_columns(tmp_path):
    path = raw_file(tmp_path, ['a'])
    csv = tmp_path / 'res.csv'
    csv.write_text('event_id,pred_label,prob\na,benign,1\n')
    with duckdb.connect() as db, pytest.raises(ValueError):
        audit_submission(db, path, csv)


@pytest.mark.parametrize('ids', [['a','a'], ['a',None], ['a','']])
def test_reject_invalid_input_ids(tmp_path, ids):
    with duckdb.connect() as db, pytest.raises(ValueError):
        audit_input(db, raw_file(tmp_path, ids))
