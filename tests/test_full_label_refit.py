from argparse import Namespace
import json
from pathlib import Path
import sys
import subprocess

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from predict_submission import (RAW_COLUMNS, encode_raw_batch, load_refit_model, predict_table,
                                audit_submission)
from prepare_full_label_refit import group_key, make_splits, prepare_source, validate_sources
from run_full_label_refit import make_plan
from soc_threat import LABELS
from soc_threat.refit_utils import bind_manifest, sha256
from train_full_label_refit import train
from train_hierarchical_content import read_data, make_loader, class_indices, predict, hierarchical_predictions
from soc_threat.hierarchical_features import transform_hierarchical_inputs


def raw_row(i, label):
    row = {c:'' for c in RAW_COLUMNS}
    row.update(event_id=f'e{i}', timestamp=1700000000+i, pipeline='syslog',
               product_name='Toy', vendor_name='Vendor', src_port='0',
               message_sanitized=json.dumps({'action':label,'object':f'object_{i}'}))
    return row


@pytest.fixture
def sources(tmp_path):
    rows = [raw_row(i,LABELS[i%3]) for i in range(60)]
    labels = [LABELS[i%3] for i in range(60)]
    raw = pa.Table.from_pylist(rows)
    pq.write_table(raw.append_column('label_binary',pa.array(labels)),tmp_path/'train.parquet')
    # Same IDs in separate splits must not cause cross-source joins or deduplication.
    pq.write_table(raw,tmp_path/'valid_input.parquet')
    pq.write_table(pa.table({'event_id':raw['event_id'],'label_binary':labels}),tmp_path/'valid_answer_private.parquet')
    return tmp_path


def prepared(sources):
    out = sources/'prepared'
    out.mkdir()
    with duckdb.connect() as db:
        db.execute("SET TimeZone='UTC'")
        assert validate_sources(db,sources)=={'train':60,'valid':60}
        paths=[]
        for split,file in [('train','train.parquet'),('valid','valid_input.parquet')]:
            paths += prepare_source(db,sources/file,split,out,'test-run',17)
        report=make_splits(db,paths,out,0.2,123)
    return out,report


def test_grouping_ignores_ids_labels_and_times_not_actions():
    a=raw_row(0,'success')
    a['message_sanitized']='hello 2026-10-15T10:00:00Z deny userA'
    b=dict(a,event_id='other',timestamp=1,label_binary='malicious',
           message_sanitized='hello   2026-11-16T20:11:22Z deny userA')
    assert group_key(a)==group_key(b)
    b['message_sanitized']=b['message_sanitized'].replace('deny','allow')
    assert group_key(a)!=group_key(b)


def test_split_no_group_leakage_source_ids_and_full_union(sources):
    out,report=prepared(sources)
    assert report['splits']['full_train.parquet']['rows']==120
    assert report['splits']['holdout.parquet']['rows']==12
    assert report['excluded_train_rows_matching_holdout']==12
    assert report['group_overlap_with_holdout']==0
    full=pq.read_table(out/'full_train.parquet').to_pandas()
    assert full.event_id.nunique()==120
    assert full.original_event_id.nunique()==60
    hold=set(pq.read_table(out/'holdout.parquet')['group_key'].to_pylist())
    for name in ['control_train.parquet','expanded_train.parquet']:
        assert hold.isdisjoint(pq.read_table(out/name)['group_key'].to_pylist())


@pytest.mark.parametrize('kind',['duplicate','wrong_id','bad_label'])
def test_invalid_answers_rejected(sources,kind):
    path=sources/'valid_answer_private.parquet'
    table=pq.read_table(path).to_pydict()
    if kind=='duplicate':table['event_id'][1]=table['event_id'][0]
    if kind=='wrong_id':table['event_id'][1]='not_in_input'
    if kind=='bad_label':table['label_binary'][1]='unknown'
    pq.write_table(pa.table(table),path)
    with duckdb.connect() as db,pytest.raises(ValueError):validate_sources(db,sources)


def test_manifest_rejects_stale_cache(tmp_path):
    path=tmp_path/'manifest.json'
    assert bind_manifest(path,{'hash':'a'})==bind_manifest(path,{'hash':'a'})
    with pytest.raises(ValueError):bind_manifest(path,{'hash':'b'})


def test_plan_does_not_select_lucky_seed():
    rows=[{'arm':arm,'seed':seed,'score':score,'best_epoch':epoch}
          for arm in ['control','expanded']
          for seed,score,epoch in [(11,.99,1),(12,1.0,3),(13,.98,2)]]
    p=make_plan(rows,[11,12,13],['--device','cpu'],'hash')
    assert p['final_seed']==11
    assert p['fixed_epochs']==2


def args_for(path,out,*,valid=None,fixed=False,epochs=1):
    return Namespace(train=path,valid=valid,fixed_epochs=fixed,output_dir=out,epochs=epochs,
                     patience=2,seed=123,device='cpu',batch_size=16,valid_batch_size=32,
                     learning_rate=.0007,num_workers=0,threads=1)


def test_training_fixed_refit_and_inference_parity(sources):
    out,_=prepared(sources)
    args=args_for(out/'full_train.parquet',sources/'model',fixed=True)
    metrics=train(args)
    assert metrics['competition_score'] is None
    assert metrics['best_epoch'] is None
    assert not (args.output_dir/'valid_predictions.parquet').exists()
    assert metrics['history'][0]['trained_rows']==120  # no dropped tail batch
    checksum=sha256(args.output_dir/'model.pt')
    assert train(args)==metrics
    assert sha256(args.output_dir/'model.pt')==checksum
    model,checkpoint=load_refit_model(args.output_dir/'model.pt',torch.device('cpu'))
    raw=pq.read_table(sources/'valid_input.parquet')
    with duckdb.connect() as db:
        result=predict_table(raw,model,checkpoint,torch.device('cpu'),32,db)
        raw_frame,tokens=encode_raw_batch(raw,db,buckets=65536)
    # Compare actual saved training features to inference's stateless feature path.
    trained_frame,trained_tokens=read_data(out/'full_train.parquet',token_column='raw_token_ids',max_rows=None,seed=1)
    pos=[trained_frame.index[trained_frame.event_id=='valid:'+eid][0] for eid in raw['event_id'].to_pylist()]
    np.testing.assert_array_equal(trained_tokens[pos],tokens)
    a=transform_hierarchical_inputs(trained_frame.iloc[pos],checkpoint['preprocessor'])
    b=transform_hierarchical_inputs(raw_frame,checkpoint['preprocessor'])
    for key in vars(a):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    loader=make_loader(a,tokens,class_indices(trained_frame.iloc[pos]),batch_size=32,shuffle=False,num_workers=0)
    p=predict(model,loader,torch.device('cpu'))
    prediction=hierarchical_predictions(p['threat'],p['subtype'],threshold=.5)[0]
    assert result['pred_label'].to_pylist()==np.asarray(LABELS)[prediction].tolist()


def test_development_resume_from_epoch_boundary(sources,monkeypatch):
    out,_=prepared(sources)
    args=args_for(out/'expanded_train.parquet',sources/'dev',valid=out/'holdout.parquet',epochs=2)
    import train_full_label_refit as module
    original=module.atomic_save
    interrupted=False
    def fail_after_committed_epoch(path,value):
        nonlocal interrupted
        original(path,value)
        if path.name=='last.pt' and not interrupted:
            interrupted=True
            raise RuntimeError('simulated power loss')
    monkeypatch.setattr(module,'atomic_save',fail_after_committed_epoch)
    with pytest.raises(RuntimeError,match='simulated power loss'):train(args)
    assert not (args.output_dir/'complete.json').exists()
    monkeypatch.setattr(module,'atomic_save',original)
    metrics=train(args)
    assert [r['epoch'] for r in metrics['history']]==[1,2]
    assert metrics['has_holdout_metrics']
    assert (args.output_dir/'valid_predictions.parquet').exists()
    reference=args_for(out/'expanded_train.parquet',sources/'uninterrupted',valid=out/'holdout.parquet',epochs=2)
    train(reference)
    resumed=torch.load(args.output_dir/'model.pt',weights_only=True)['model_state']
    uninterrupted=torch.load(reference.output_dir/'model.pt',weights_only=True)['model_state']
    for key in resumed:torch.testing.assert_close(resumed[key],uninterrupted[key],rtol=0,atol=0)


def test_refit_cannot_validate(sources):
    args=args_for(sources/'train.parquet',sources/'bad',valid=sources/'valid_input.parquet',fixed=True)
    with pytest.raises(ValueError):train(args)


def test_cli_study_refit_predict_and_reuse(sources):
    src=Path(__file__).resolve().parents[1]/'src'
    output=sources/'workflow'
    def run(script,*args):
        return subprocess.run([sys.executable,str(src/script),*map(str,args)],
                              text=True,capture_output=True,check=True,timeout=120)
    common=['--data-dir',sources,'--output-dir',output]
    run('run_full_label_refit.py','--stage','study',*common,'--seeds','123','--epochs','1',
        '--device','cpu','--threads','1','--batch-size','16','--valid-batch-size','32')
    comparison=json.loads((output/'study_comparison.json').read_text('utf-8'))
    assert len(comparison['runs'])==2
    run('run_full_label_refit.py','--stage','refit',*common,'--allow-nonimproving')
    assert (output/'v5_2_exp03_full_label_model.tar.gz.sha256').exists()
    prediction=['--input',sources/'valid_input.parquet','--refit-checkpoint',output/'final_model/model.pt',
                '--output-dir',sources/'submission','--device','cpu','--threads','1','--chunk-size','17']
    run('predict_submission.py',*prediction)
    repeated=run('predict_submission.py',*prediction)
    assert '"reused": true' in repeated.stdout
    audit=json.loads((sources/'submission/submission_audit.json').read_text('utf-8'))
    assert audit['rows']==60 and audit['invalid_rows']==0 and audit['missing_ids']==0
    assert audit['model']=='v5.2-exp03-full-label-base'
    (output/'final_model/model.pt.sha256').write_text('0'*64+'  model.pt\n')
    with pytest.raises(ValueError,match='SHA256'):load_refit_model(output/'final_model/model.pt',torch.device('cpu'))
