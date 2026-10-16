"""Compare two audited prediction runs on the same input, without ground truth."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import duckdb

from soc_threat.refit_utils import sha256, write_json

LABELS = ['benign','malicious','suspicious']


def sqlpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def compare(input_path: Path, old_dir: Path, new_dir: Path, output_dir: Path) -> dict:
    input_path=input_path.resolve(strict=True)
    old_dir,new_dir=old_dir.resolve(strict=True),new_dir.resolve(strict=True)
    if old_dir==new_dir or output_dir.resolve() in {old_dir,new_dir,input_path.parent}:
        raise ValueError('Use separate old/new/comparison output directories')
    digest=sha256(input_path)
    provenance={}
    for name,directory in [('old',old_dir),('new',new_dir)]:
        manifest=json.loads((directory/'manifest.json').read_text('utf-8'))
        audit=json.loads((directory/'submission_audit.json').read_text('utf-8'))
        if manifest['input_sha256']!=digest:
            raise ValueError(f'{name} predictions were generated from a different input')
        csv_hash=sha256(directory/'res.csv')
        if csv_hash!=audit['res_sha256']:
            raise ValueError(f'{name} CSV changed after inference audit')
        if manifest['checkpoint_sha256']!=audit['checkpoint_sha256']:
            raise ValueError(f'{name} checkpoint provenance mismatch')
        provenance[name]={'model':audit['model'],'checkpoint_sha256':audit['checkpoint_sha256'],
                          'res_sha256':csv_hash,'directory':str(directory)}
    identity={'input_sha256':digest,'predictions':provenance,'comparison_code_sha256':sha256(Path(__file__))}
    output_dir.mkdir(parents=True,exist_ok=True)
    summary_path=output_dir/'comparison.json'
    difference_path=output_dir/'disagreements.csv'
    if summary_path.exists():
        existing=json.loads(summary_path.read_text('utf-8'))
        if existing['identity']!=identity:
            raise ValueError('Comparison inputs/code changed; use a NEW output directory')
        if not difference_path.is_file() or sha256(difference_path)!=existing['disagreements_sha256']:
            raise ValueError('Comparison output missing/corrupt; use a NEW output directory')
        return existing
    if difference_path.exists():
        raise FileExistsError('Unowned disagreements.csv exists; use a NEW output directory')
    with duckdb.connect() as db:
        db.execute('SET threads=4')
        db.execute("SET memory_limit='4GB'")
        db.read_parquet(str(input_path)).create_view('raw')
        n,unique=db.execute('SELECT count(*),count(DISTINCT event_id) FROM raw').fetchone()
        if not n or n!=unique:
            raise ValueError('Invalid input coverage/IDs')
        audits={}
        def records(query):
            c=db.execute(query)
            return [dict(zip([d[0] for d in c.description],r)) for r in c.fetchall()]
        for name,directory in [('old',old_dir),('new',new_dir)]:
            db.read_csv(str(directory/'res.csv'),header=True,all_varchar=True,strict_mode=True).create_view(name+'_pred')
            if [r[0] for r in db.execute('DESCRIBE '+name+'_pred').fetchall()]!=['event_id','pred_label']:
                raise ValueError('Submission must have exactly event_id,pred_label')
            count,distinct,invalid=db.execute(f'''SELECT count(*),count(DISTINCT event_id),
                count(*) FILTER (WHERE event_id IS NULL OR trim(event_id)='' OR event_id<>trim(event_id)
                OR pred_label IS NULL OR pred_label NOT IN ('benign','malicious','suspicious'))
                FROM {name}_pred''').fetchone()
            missing=db.execute(f'SELECT count(*) FROM raw ANTI JOIN {name}_pred USING(event_id)').fetchone()[0]
            extra=db.execute(f'SELECT count(*) FROM {name}_pred ANTI JOIN raw USING(event_id)').fetchone()[0]
            if count!=n or distinct!=n or invalid or missing or extra:
                raise ValueError(f'{name} predictions have invalid rows or incomplete IDs')
            audits[name]={'rows':count,'distinct_event_ids':distinct,'invalid_rows':invalid,
                         'missing_ids':missing,'extra_ids':extra}
        db.execute('''CREATE TEMP TABLE paired AS SELECT o.event_id,o.pred_label AS old_label,
            n.pred_label AS new_label FROM old_pred o JOIN new_pred n USING(event_id)''')
        changed=db.execute('SELECT count(*) FROM paired WHERE old_label<>new_label').fetchone()[0]
        summary={'identity':identity,'rows':n,'changed_rows':changed,'unchanged_rows':n-changed,
            'change_fraction':changed/n,'audits':audits,'has_ground_truth':False,
            'competition_score':None,'manual_overrides':False,
            'transitions':records('SELECT old_label,new_label,count(*) AS rows FROM paired GROUP BY ALL ORDER BY 1,2'),
            'label_counts':{name:dict(db.execute(f'SELECT pred_label,count(*) FROM {name}_pred GROUP BY 1').fetchall()) for name in audits},
            'changed_groups':records('''SELECT r.pipeline,r.vendor_name,r.product_name,p.old_label,p.new_label,
                count(*) AS rows,min(p.event_id) AS example_event_id FROM paired p JOIN raw r USING(event_id)
                WHERE p.old_label<>p.new_label GROUP BY ALL ORDER BY rows DESC,pipeline,product_name'''),
            'interpretation':'Disagreements are not known errors. Compare scores only on this same test/evaluation scope.'}
        temporary=output_dir/'disagreements.tmp.csv'
        db.execute(f'''COPY (SELECT p.event_id,p.old_label,p.new_label,r.* EXCLUDE(event_id)
            FROM paired p JOIN raw r USING(event_id) WHERE p.old_label<>p.new_label ORDER BY p.event_id)
            TO '{sqlpath(temporary)}' (FORMAT CSV,HEADER true)''')
        os.replace(temporary,difference_path)
        summary['disagreements_sha256']=sha256(difference_path)
        write_json(summary_path,summary)
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--old-dir',type=Path,required=True)
    parser.add_argument('--new-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps({'stage':'comparison_complete',**compare(args.input,args.old_dir,args.new_dir,args.output_dir)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
