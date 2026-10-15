"""Same hierarchical base architecture, fresh preprocessing, and explicit fixed-epoch refit.

No old anchor or residual weights are loaded. Only this run's own last.pt may resume.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from soc_threat import LABELS
from soc_threat.hierarchical_features import fit_hierarchical_preprocessor, transform_hierarchical_inputs
from soc_threat.hierarchical_model import HierarchicalContentModel
from soc_threat.refit_utils import (bind_manifest, environment_info, git_revision, sha256,
                                   source_hashes, verify_artifacts, write_checksum, write_json)
from train_hierarchical_content import (binary_class_weights, class_indices, evaluate_outputs,
    make_loader, masked_cross_entropy, move_inputs, predict, read_data, seed_everything)
from train_npu_tabular import choose_device


def atomic_save(path: Path, value) -> None:
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def cpu_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}


def device_rng(device, restore=None):
    backend = getattr(torch, device.type, None)
    if device.type == 'cpu' or backend is None:
        return None
    if restore is not None:
        backend.set_rng_state_all(restore)
    return backend.get_rng_state_all()


def epoch_train(model, loader, optimizer, device, threat_weights, subtype_weights) -> dict:
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        mc, mn, sc, sn, tokens, gate, _, labels = move_inputs(batch, device)
        threat = (labels != 0).long()
        subtype = (labels-1).clamp_min(0)
        mask = labels != 0
        optimizer.zero_grad(set_to_none=True)
        output = model(mc, mn, sc, sn, tokens, gate)
        loss = (F.cross_entropy(output.threat_logits, threat, weight=threat_weights)
                + 0.75 * masked_cross_entropy(output.subtype_logits, subtype, mask, subtype_weights)
                + 0.15 * F.cross_entropy(output.metadata_threat_logits, threat, weight=threat_weights)
                + 0.25 * F.cross_entropy(output.content_threat_logits, threat, weight=threat_weights)
                + 0.35 * masked_cross_entropy(output.metadata_subtype_logits, subtype, mask, subtype_weights))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss; no final checkpoint published')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        count += len(labels)
        total += float(loss.detach().cpu()) * len(labels)
    return {'loss': total / max(1, count), 'trained_rows': count}


def train(args) -> dict:
    if args.fixed_epochs == (args.valid is not None):
        raise ValueError('Choose exactly one: --valid for development, or --fixed-epochs WITHOUT validation')
    if min(args.epochs, args.batch_size, args.valid_batch_size, args.patience, args.threads) < 1 or args.num_workers < 0:
        raise ValueError('Invalid training sizes')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Learning rate must be positive and finite')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if not (output/'run_manifest.json').exists() and any((output/n).exists() for n in ['model.pt','last.pt','metrics.json']):
        raise FileExistsError('Unowned training artifacts exist; choose a NEW directory')
    torch.set_num_threads(args.threads)
    if str(args.device).startswith('npu'):
        import torch_npu  # noqa: F401
    device = choose_device(args.device)
    identity = {'format': 1, 'train_sha256': sha256(args.train),
                'valid_sha256': sha256(args.valid) if args.valid else None,
                'arguments': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                              if k not in ('train','valid','output_dir')},
                'code': source_hashes(), 'torch': str(torch.__version__), 'resolved_device': str(device),
                'environment': environment_info()}
    run_id = bind_manifest(output / 'run_manifest.json', identity)
    completion = output / 'complete.json'
    artifact_names = ['model.pt', 'model.pt.sha256', 'metrics.json', 'preprocessor.json']
    if args.valid:
        artifact_names += ['valid_predictions.parquet']
    if completion.exists():
        completed = json.loads(completion.read_text('utf-8'))
        if verify_artifacts(output, artifact_names, completed):
            print(json.dumps({'stage': 'training_reused', 'directory': str(output)}), flush=True)
            return json.loads((output/'metrics.json').read_text('utf-8'))
    seed_everything(args.seed)
    if device.type != 'cpu':
        getattr(torch, device.type).manual_seed_all(args.seed)
    frame, tokens = read_data(args.train, token_column='raw_token_ids', max_rows=None, seed=args.seed)
    if len(tokens) == 0 or tokens.shape[1] != 96 or int(tokens.max()) >= 65536:
        raise ValueError('Expected nonempty 96-token / 65536-bucket features')
    preprocessor = fit_hierarchical_preprocessor(frame, novelty_pseudocount=32)
    arrays = transform_hierarchical_inputs(frame, preprocessor)
    y = class_indices(frame)
    train_rows = len(frame)
    train_counts = dict(zip(LABELS, np.bincount(y, minlength=3).astype(int).tolist()))
    wt = torch.tensor(binary_class_weights((y != 0).astype(np.int64), 0.25), device=device)
    ws = torch.tensor(binary_class_weights(y[y != 0]-1, 0), device=device)
    dataset = make_loader(arrays, tokens, y, batch_size=args.batch_size, shuffle=False, num_workers=0).dataset
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                        num_workers=args.num_workers, persistent_workers=args.num_workers>0)
    # Explicit generators isolate data-loader state from network dropout.
    loader.generator = torch.Generator()
    loader.sampler.generator = torch.Generator()
    del frame, arrays, tokens, y
    gc.collect()
    valid_loader, valid_ids = None, None
    if args.valid:
        vf, vt = read_data(args.valid, token_column='raw_token_ids', max_rows=None, seed=args.seed)
        valid_ids = vf['event_id'].astype(str).to_numpy(copy=True)
        va = transform_hierarchical_inputs(vf, preprocessor)
        valid_loader = make_loader(va, vt, class_indices(vf), batch_size=args.valid_batch_size,
                                   shuffle=False, num_workers=args.num_workers)
        valid_loader.generator = torch.Generator().manual_seed(args.seed)
        del vf, vt, va
        gc.collect()
    config = {'novelty_gate_mode': 'count', 'hash_buckets': 65536,
              'content_embedding_dim': 64, 'content_output_dim': 128,
              'metadata_cardinalities': list(preprocessor['metadata']['cardinalities']),
              'metadata_numeric_count': len(preprocessor['metadata']['numeric_features']),
              'semantic_cardinalities': list(preprocessor['semantic']['cardinalities']),
              'semantic_numeric_count': len(preprocessor['semantic']['numeric_features']),
              'token_dropout': 0.05, 'category_dropout': 0.02,
              'content_input_mode': 'raw', 'content_view_count': 4, 'content_tokens_per_view': 64}
    model = HierarchicalContentModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    start, best_epoch, stale = 1, 0, 0
    best_key, best_state, history = (-1.0, -1.0, -float('inf')), None, []
    last = output/'last.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=True)
        if state['run_id'] != run_id:
            raise ValueError('Resume checkpoint belongs to a different run')
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer'])
        start, best_epoch, stale = state['epoch']+1, state['best_epoch'], state['stale']
        best_key, best_state, history = tuple(state['best_key']), state['best_state'], state['history']
        torch.set_rng_state(state['cpu_rng'])
        device_rng(device, state['device_rng'])
        del state
        print(json.dumps({'stage': 'resume_epoch_boundary', 'next_epoch': start}), flush=True)
    print(json.dumps({'stage': 'refit_training_ready', 'train_rows': train_rows,
                      'valid_rows': len(valid_ids) if valid_ids is not None else 0,
                      'fixed_epochs': args.fixed_epochs, 'device': str(device),
                      'seed': args.seed, 'train_labels': train_counts}), flush=True)
    for epoch in range(start, args.epochs+1):
        if valid_loader is not None and stale >= args.patience:
            break
        started = time.monotonic()
        loader.generator.manual_seed(args.seed+epoch)
        loader.sampler.generator.manual_seed(args.seed+epoch*1009)
        result = {'epoch': epoch, **epoch_train(model, loader, optimizer, device, wt, ws)}
        if valid_loader is not None:
            values, _ = evaluate_outputs(predict(model, valid_loader, device), threat_threshold=0.5)
            audit = values['hierarchical_audit']
            key = (float(values['competition_score']), float(audit['content_threat']['f1']),
                   -float(values['multiclass_log_loss']))
            result.update({'competition_score': key[0], 'macro_f1': values['macro_f1'],
                           'threat_fp': audit['threat']['false_positive'],
                           'threat_fn': audit['threat']['false_negative']})
            better = key > best_key
        else:
            # No model selection on training scores: the last scheduled epoch wins.
            key, better = best_key, True
        if better:
            best_key, best_epoch, best_state, stale = key, epoch, cpu_state(model), 0
        else:
            stale += 1
        result['seconds'] = round(time.monotonic()-started, 2)
        history.append(result)
        atomic_save(last, {'run_id': run_id, 'epoch': epoch, 'model_state': cpu_state(model),
                          'optimizer': optimizer.state_dict(), 'best_key': best_key,
                          'best_epoch': best_epoch, 'best_state': best_state, 'stale': stale,
                          'history': history, 'cpu_rng': torch.get_rng_state(), 'device_rng': device_rng(device)})
        write_json(output/'history.json', history)
        print(json.dumps({'stage': 'epoch_complete', **result}), flush=True)
    if best_state is None:
        raise RuntimeError('No completed training epoch')
    model.load_state_dict(best_state)
    metrics = {'competition_score': None, 'has_holdout_metrics': False}
    if valid_loader is not None:
        evaluated = predict(model, valid_loader, device)
        metrics, predictions = evaluate_outputs(evaluated, threat_threshold=0.5)
        metrics['has_holdout_metrics'] = True
        names = np.asarray(LABELS)
        pd.DataFrame({'event_id': valid_ids, 'true_label': names[predictions['actual']],
                      'pred_label': names[predictions['predicted']],
                      **{'prob_'+label: predictions['probabilities'][:,i] for i,label in enumerate(LABELS)}}).to_parquet(output/'valid_predictions.parquet', index=False)
    metrics.update({'model': 'v5.2-exp03-full-label-base', 'seed': args.seed,
                    'training_mode': 'fixed_full_refit' if args.fixed_epochs else 'development',
                    'best_epoch': best_epoch if args.valid else None, 'trained_epochs': len(history),
                    'selected_epoch': best_epoch, 'train_rows': train_rows, 'train_labels': train_counts,
                    'history': history, 'device': str(device),
                    'selection': 'fixed epoch count; no validation read' if args.fixed_epochs else 'development score, content F1, log loss'})
    checkpoint = {'checkpoint_kind': 'full_label_hierarchical',
                  'model_name': 'v5.2-exp03-full-label-base', 'model_state': best_state,
                  'model_config': config, 'preprocessor': preprocessor, 'labels': list(LABELS),
                  'threat_threshold': 0.5, 'token_length': 96, 'run_id': run_id,
                  'training_mode': metrics['training_mode'], 'git_revision': git_revision()}
    atomic_save(output/'model.pt', checkpoint)
    write_checksum(output/'model.pt')
    write_json(output/'preprocessor.json', preprocessor)
    write_json(output/'metrics.json', metrics)
    write_json(completion, {'run_id': run_id, 'artifact_sha256': {n:sha256(output/n) for n in artifact_names}})
    print(json.dumps({'stage': 'training_complete', 'selected_epoch': best_epoch,
                      'competition_score': metrics['competition_score'], 'output': str(output)}), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--valid', type=Path)
    mode.add_argument('--fixed-epochs', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--valid-batch-size', type=int, default=4096)
    parser.add_argument('--learning-rate', type=float, default=0.0007)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
