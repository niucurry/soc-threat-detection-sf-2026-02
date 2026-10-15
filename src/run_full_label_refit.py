"""Orchestrate paired development runs, then an explicitly requested full refit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile

from soc_threat.refit_utils import (bind_manifest, environment_info, git_revision, sha256,
                                   source_hashes, write_checksum, write_json)

ROOT = Path(__file__).resolve().parents[1]


def call(script: str, *args):
    command = [sys.executable, '-u', str(ROOT/'src'/script), *map(str, args)]
    print('RUN ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def summarize(metrics: dict, arm: str, seed: int) -> dict:
    c = metrics['confusion_matrix']['counts']
    threat = metrics['hierarchical_audit']['threat']
    return {'arm': arm, 'seed': seed, 'score': metrics['competition_score'],
            'errors': sum(map(sum,c))-sum(c[i][i] for i in range(3)),
            'fp': threat['false_positive'], 'fn': threat['false_negative'],
            'subtype_confusion': c[1][2]+c[2][1], 'macro_f1': metrics['macro_f1'],
            'malicious_recall': metrics['per_class']['malicious']['recall'],
            'suspicious_recall': metrics['per_class']['suspicious']['recall'],
            'log_loss': metrics['multiclass_log_loss'], 'best_epoch': metrics['best_epoch']}


def make_plan(rows: list[dict], seeds: list[int], common_args: list[str], data_sha: str) -> dict:
    scores = {arm: [r['score'] for r in rows if r['arm']==arm] for arm in ('control','expanded')}
    if any(len(values)!=len(seeds) for values in scores.values()):
        raise ValueError('Incomplete paired study')
    delta = statistics.mean(scores['expanded'])-statistics.mean(scores['control'])
    return {'format': 1, 'final_seed': seeds[0],
            'fixed_epochs': max(1, round(statistics.median(r['best_epoch'] for r in rows if r['arm']=='expanded'))),
            'common_training_args': common_args, 'full_train_sha256': data_sha,
            'mean_score_delta': delta,
            'selection_policy': 'Predeclared first seed; median expanded best epoch across ALL seeds, no seed cherry-picking',
            'note': 'Development comparison only. No test labels or test score used. Final full refit has no validation score.'}


def study(args):
    out = args.output_dir.resolve()
    features = out/'features'
    seeds = [int(s) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique integers')
    common = ['--device',args.device,'--batch-size',str(args.batch_size),
              '--valid-batch-size',str(args.valid_batch_size), '--num-workers',str(args.num_workers),
              '--threads',str(args.threads), '--learning-rate',str(args.learning_rate)]
    bind_manifest(out/'study_manifest.json', {'seeds': seeds, 'code': source_hashes(),
                  'common_training_args': common, 'max_epochs': args.epochs, 'patience': args.patience,
                  'holdout_fraction': args.holdout_fraction, 'split_seed': args.split_seed})
    call('prepare_full_label_refit.py', '--data-dir', args.data_dir, '--output-dir', features,
         '--holdout-fraction', args.holdout_fraction, '--split-seed', args.split_seed, '--threads', args.threads)
    rows = []
    for seed in seeds:
        for arm in ('control','expanded'):
            run_dir = out/'study'/f'{arm}_seed{seed}'
            call('train_full_label_refit.py', '--train', features/f'{arm}_train.parquet',
                 '--valid', features/'holdout.parquet', '--output-dir', run_dir,
                 '--epochs', args.epochs, '--patience', args.patience, '--seed', seed, *common)
            m = json.loads((run_dir/'metrics.json').read_text('utf-8'))
            rows.append(summarize(m, arm, seed))
            write_json(out/'study_progress.json', rows)
    aggregate = {}
    for arm in ('control','expanded'):
        selected = [r for r in rows if r['arm']==arm]
        values = [r['score'] for r in selected]
        aggregate[arm] = {'mean_score': statistics.mean(values), 'score_std': statistics.pstdev(values),
                          'min_score': min(values), 'max_score': max(values),
                          'mean_errors': statistics.mean(r['errors'] for r in selected),
                          'mean_fp': statistics.mean(r['fp'] for r in selected),
                          'mean_fn': statistics.mean(r['fn'] for r in selected)}
    comparison = {'runs': rows, 'aggregate': aggregate,
                  'paired_score_deltas': [{'seed':seed, 'expanded_minus_control':
                     next(r['score'] for r in rows if r['seed']==seed and r['arm']=='expanded')-
                     next(r['score'] for r in rows if r['seed']==seed and r['arm']=='control')} for seed in seeds],
                  'evaluation_scope': 'Shared grouped development holdout; not the official test score'}
    write_json(out/'study_comparison.json', comparison)
    plan = make_plan(rows, seeds, common, sha256(features/'full_train.parquet'))
    plan['input_manifest_sha256'] = sha256(features/'input_manifest.json')
    plan['code'] = source_hashes()
    write_json(out/'refit_plan.json', plan)
    print(json.dumps({'stage':'study_complete', **comparison, 'refit_plan':plan}, ensure_ascii=False, indent=2), flush=True)


def refit(args):
    out = args.output_dir.resolve()
    plan = json.loads((out/'refit_plan.json').read_text('utf-8'))
    features = out/'features'
    if plan['code'] != source_hashes():
        raise ValueError('Code changed after study; use the recorded commit or rerun in a new directory')
    if (sha256(features/'full_train.parquet') != plan['full_train_sha256']
            or sha256(features/'input_manifest.json') != plan['input_manifest_sha256']):
        raise ValueError('Study data changed; refusing final refit')
    if plan['mean_score_delta'] <= 0 and not args.allow_nonimproving:
        raise ValueError('Expanded data did not improve mean development score. Review study_comparison.json; '
                         'only use --allow-nonimproving after explicitly accepting this experimental risk.')
    call('train_full_label_refit.py', '--train', features/'full_train.parquet', '--fixed-epochs',
         '--output-dir',out/'final_model', '--epochs',plan['fixed_epochs'], '--seed',plan['final_seed'],
         *plan['common_training_args'])
    # A small self-contained weights bundle, not gigabytes of generated features.
    archive = out/'v5_2_exp03_full_label_model.tar.gz'
    import os
    temporary = archive.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as tf:
        for name in ['model.pt','model.pt.sha256','preprocessor.json','metrics.json','run_manifest.json','complete.json']:
            tf.add(out/'final_model'/name, arcname='final_model/'+name)
        for name in ['study_comparison.json','refit_plan.json','study_manifest.json']:
            tf.add(out/name, arcname=name)
        for name in ['input_manifest.json','split_report.json']:
            tf.add(features/name, arcname='features/'+name)
        write_json(out/'backup_provenance.json', {'git_revision':git_revision(), 'code':source_hashes(),
                                                'environment':environment_info()})
        tf.add(out/'backup_provenance.json', arcname='backup_provenance.json')
        for path in sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'scripts').glob('*.sh')):
            tf.add(path, arcname='code/'+path.relative_to(ROOT).as_posix())
        for name in ['requirements-npu.txt','docs/FULL_LABEL_REFIT.md']:
            tf.add(ROOT/name, arcname='code/'+name)
    os.replace(temporary, archive)
    write_checksum(archive)
    print(json.dumps({'stage':'full_refit_complete', 'model':str(out/'final_model'/'model.pt'),
                      'backup':str(archive), 'has_validation_score':False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['study','refit'], required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', default='20261015,20261016,20261017')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--holdout-fraction', type=float, default=0.2)
    parser.add_argument('--split-seed', type=int, default=20261015)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--valid-batch-size', type=int, default=4096)
    parser.add_argument('--learning-rate', type=float, default=0.0007)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--allow-nonimproving', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (study if args.stage=='study' else refit)(args)


if __name__ == '__main__':
    main()
