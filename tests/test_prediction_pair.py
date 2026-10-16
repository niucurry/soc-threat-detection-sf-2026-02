import csv
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_prediction_csv import RAW_COLUMNS, SCHEMA, prepare
from compare_prediction_files import compare
from soc_threat.refit_utils import sha256, write_json
from predict_submission import encode_raw_batch, RAW_COLUMNS as PREDICT_COLUMNS


def raw_row(i=1):
    row={k:'' for k in RAW_COLUMNS}
    row.update(event_id=f'EVT-{i:010}',timestamp=1722019019.8346298,pipeline='syslog',
        src_port='0053',username='NA',product_name='NULL',
        message_sanitized='line one, "quoted"\n日志第二行\\n literal\x00')
    return row


def csv_input(tmp_path,rows=None,header=None,bom=False):
    path=tmp_path/'raw.csv'
    with path.open('w',encoding='utf-8-sig' if bom else 'utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=header or RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows if rows is not None else [raw_row(1),raw_row(2)])
    return path


def test_csv_contract_text_timestamp_and_features_preserved(tmp_path):
    assert RAW_COLUMNS==PREDICT_COLUMNS
    raw=[raw_row(1),raw_row(2)]
    path=csv_input(tmp_path,raw,header=list(reversed(RAW_COLUMNS)),bom=True)
    digest=sha256(path)
    out=tmp_path/'prepared'
    audit=prepare(path,out)
    converted=pq.read_table(out/'test_input.parquet')
    expected=pa.Table.from_pylist(raw,schema=SCHEMA)
    assert converted.equals(expected)
    assert audit['rows']==2 and not audit['reused']
    assert prepare(path,out)['reused']
    assert sha256(path)==digest
    with duckdb.connect() as db:
        a,tokens_a=encode_raw_batch(expected,db,buckets=65536)
        b,tokens_b=encode_raw_batch(converted,db,buckets=65536)
    assert a.equals(b)
    np.testing.assert_array_equal(tokens_a,tokens_b)


@pytest.mark.parametrize('bad',['','   ',' x ','duplicate'])
def test_bad_ids_rejected(tmp_path,bad):
    rows=[raw_row(1),raw_row(2)]
    rows[1]['event_id']=rows[0]['event_id'] if bad=='duplicate' else bad
    with pytest.raises(ValueError,match='Invalid raw input'):
        prepare(csv_input(tmp_path,rows),tmp_path/'out')
    assert not (tmp_path/'out/test_input.parquet').exists()


@pytest.mark.parametrize('bad',['',float('nan'),float('inf'),1722019019000])
def test_timestamp_not_silently_repaired(tmp_path,bad):
    row=raw_row()
    row['timestamp']=bad
    with pytest.raises(ValueError):prepare(csv_input(tmp_path,[row]),tmp_path/'out')


def test_unexpected_and_duplicate_headers_rejected(tmp_path):
    row=dict(raw_row(),label_binary='benign')
    with pytest.raises(ValueError,match='12 raw columns'):
        prepare(csv_input(tmp_path,[row],header=RAW_COLUMNS+['label_binary']),tmp_path/'out')
    path=tmp_path/'raw.csv'
    path.write_text(','.join(RAW_COLUMNS[:-1]+['event_id'])+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='12 raw columns'):prepare(path,tmp_path/'out')


def test_malformed_csv_not_skipped(tmp_path):
    path=csv_input(tmp_path)
    with path.open('a',encoding='utf-8') as f:f.write('bad,row,with,four,values\n')
    with pytest.raises(ValueError):prepare(path,tmp_path/'out')
    assert not (tmp_path/'out/test_input.parquet').exists()


def test_empty_and_changed_input_rejected(tmp_path):
    path=csv_input(tmp_path,[])
    with pytest.raises(ValueError):prepare(path,tmp_path/'out')
    path=csv_input(tmp_path)
    prepare(path,tmp_path/'out')
    csv_input(tmp_path,[raw_row(9)])
    with pytest.raises(ValueError,match='changed'):prepare(path,tmp_path/'out')


def audited_prediction(directory,input_path,ids,labels,model):
    directory.mkdir()
    path=directory/'res.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['event_id','pred_label'])
        writer.writerows(zip(ids,labels))
    write_json(directory/'manifest.json',{'input_sha256':sha256(input_path),'checkpoint_sha256':model})
    write_json(directory/'submission_audit.json',{'res_sha256':sha256(path),'checkpoint_sha256':model,'model':model})


def prediction_sources(tmp_path):
    input_path=tmp_path/'test.parquet'
    pq.write_table(pa.Table.from_pylist([raw_row(i) for i in range(1,4)],schema=SCHEMA),input_path)
    ids=[raw_row(i)['event_id'] for i in range(1,4)]
    old=tmp_path/'old'
    new=tmp_path/'new'
    audited_prediction(old,input_path,ids,['benign','malicious','suspicious'],'old-model')
    # New CSV deliberately reversed: align by event_id, never by physical line number.
    audited_prediction(new,input_path,list(reversed(ids)),['suspicious','malicious','malicious'],'new-model')
    return input_path,old,new


def test_comparison_exact_counts_ids_and_reuse(tmp_path):
    source,old,new=prediction_sources(tmp_path)
    old_sha,new_sha=sha256(old/'res.csv'),sha256(new/'res.csv')
    output=tmp_path/'comparison'
    result=compare(source,old,new,output)
    assert result['rows']==3 and result['changed_rows']==1
    assert result['has_ground_truth'] is False and result['competition_score'] is None
    assert compare(source,old,new,output)==result
    with (output/'disagreements.csv').open(encoding='utf-8',newline='') as f:
        diff=list(csv.DictReader(f))
    assert len(diff)==1 and diff[0]['event_id']==raw_row()['event_id']
    assert diff[0]['message_sanitized']==raw_row()['message_sanitized']
    assert sha256(old/'res.csv')==old_sha and sha256(new/'res.csv')==new_sha


@pytest.mark.parametrize('mode',['input_hash','csv_changed','missing_id','extra_id','invalid_label','duplicate_id'])
def test_comparison_rejects_stale_or_invalid_runs(tmp_path,mode):
    source,old,new=prediction_sources(tmp_path)
    if mode=='input_hash':
        m=json.loads((new/'manifest.json').read_text('utf-8'))
        m['input_sha256']='other-input'
        write_json(new/'manifest.json',m)
    else:
        path=new/'res.csv'
        text=path.read_text('utf-8')
        if mode in ['csv_changed','invalid_label']:text=text.replace('malicious','invalid')
        elif mode=='missing_id':text='\n'.join(text.splitlines()[:-1])+'\n'
        elif mode=='extra_id':text+='unknown,benign\n'
        else:text=text.replace('EVT-0000000002','EVT-0000000001')
        path.write_text(text,encoding='utf-8')
        if mode!='csv_changed':
            audit=json.loads((new/'submission_audit.json').read_text('utf-8'))
            audit['res_sha256']=sha256(path)
            write_json(new/'submission_audit.json',audit)
    with pytest.raises(ValueError):compare(source,old,new,tmp_path/'comparison')
