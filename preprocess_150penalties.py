#!/usr/bin/env python3
"""Apply the previous first15 preprocessing functions to the 150penalties CSV."""
from __future__ import annotations
import argparse
import contextlib
import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import preprocess_penaltyfirst15 as prior

SELECTED_NAMES = [prior.NAMES[j] for j in prior.SELECTED]
BBOX_KEYS = ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2')

def complete_selected(row):
    return all(row[f'{name}_valid'] == '1' for name in SELECTED_NAMES)

def prepare_sources(source_rows):
    groups = defaultdict(list)
    for row in source_rows:
        groups[row['video']].append(row)
    adapted_sources, trimmed, bbox_repairs = [], [], []
    for name, sequence in groups.items():
        assert all(int(b['source_frame']) > int(a['source_frame']) for a,b in zip(sequence,sequence[1:]))
        first = next((i for i,row in enumerate(sequence) if complete_selected(row)),None)
        if first is None:
            raise ValueError(f'No complete selected-joint start: {name}')
        trimmed.extend(sequence[:first])
        sequence = sequence[first:]
        bbox_valid = [i for i,r in enumerate(sequence) if all(prior.finite_coordinate(r[k]) for k in BBOX_KEYS)]
        for i, original in enumerate(sequence):
            row = dict(original)
            observed = i in bbox_valid
            if not observed:
                before,after=[v for v in bbox_valid if v<i],[v for v in bbox_valid if v>i]
                if not before or not after:
                    raise ValueError(f'Unbracketed missing bbox: {name} frame {row["source_frame"]}')
                lo,hi=before[-1],after[0]
                f,lf,hf=[int(r['source_frame']) for r in (row,sequence[lo],sequence[hi])]
                q=(f-lf)/(hf-lf)
                for key in BBOX_KEYS:
                    row[key]=str((1-q)*float(sequence[lo][key])+q*float(sequence[hi][key]))
                bbox_repairs.append({'kick_id':name,'frame':f,'left_frame':lf,'right_frame':hf})
            adapted_sources.append((row,observed))
    return adapted_sources,trimmed,bbox_repairs

def validate_final(path, reference, canonical, normalized, original, specs, sampling):
    fields,rows=prior.read_csv(path)
    ref_fields,ref_rows=prior.read_csv(reference)
    assert set(ref_fields)<=set(fields)
    counts=Counter(r['kick_id'] for r in rows)
    assert set(counts.values())=={20}
    assert set(counts)=={r['kick_id'] for r in sampling['selected_sequences']}
    assert len(rows)==20*sampling['selected_kicks']
    features=[f'kp_{j}_{axis}n' for j in prior.SELECTED for axis in ('x','y')]+list(prior.ANGLES)
    angle_error={}
    for column,triplet in prior.ANGLES.items():
        errors=[]
        for r in ref_rows:
            pts=[np.array([float(r[f'kp_{j}_xn']),float(r[f'kp_{j}_yn'])]) for j in triplet]
            errors.append(abs(prior.angle_scaled(*pts)-float(r[column])))
        angle_error[column]=max(errors)
        assert angle_error[column]<1e-6
    source={(r['video'],r['source_frame']):r for r in original}
    _,norm=prior.read_csv(normalized)
    norm={(r['kick_id'],r['frame']):r for r in norm}
    full=defaultdict(list)
    selected=defaultdict(list)
    for r in canonical:full[r['kick_id']].append(r)
    for r in rows:selected[r['kick_id']].append(r)
    for name,sequence in selected.items():
        assert specs[name]['contact_status'] in ('visually_reviewed', 'assumed_clip_end')
        assert [r['sample_20_idx'] for r in sequence]==[str(i) for i in range(20)]
        times=[int(r['frame']) for r in sequence]
        assert all(b>a for a,b in zip(times,times[1:]))
        assert times[0]==int(full[name][0]['frame'])
        assert times[-1]==specs[name]['contact_frame'] and sequence[-1]['last_touch']=='1'
        assert sum(int(r['last_touch']) for r in sequence)==1
        assert len(full[name])>20
        torso=[]
        for r in full[name]:
            pts={j:np.array([float(r[f'kp_{j}_x']),float(r[f'kp_{j}_y'])]) for j in (5,6,11,12)}
            torso.append(np.linalg.norm((pts[5]+pts[6])/2-(pts[11]+pts[12])/2))
        scale=float(np.median(torso))
        for r in sequence:
            assert r['dataset']=='150penalties' and r['r_or_l']=='R'
            assert all(math.isfinite(float(r[c])) for c in features)
            assert math.isclose(float(r['clip_torso_scale']),scale,rel_tol=1e-12)
            assert abs((float(r['kp_11_xn'])+float(r['kp_12_xn']))/2)<1e-10
            assert abs((float(r['kp_11_yn'])+float(r['kp_12_yn']))/2)<1e-10
            raw=source[(name,r['source_frame'])]
            camera,foot,direction=prior.filename_labels(name)
            assert r['camera_direction']==camera and r['original_r_or_l']==foot
            assert r['original_kick_direction']==direction
            assert r['kick_direction']==({'L':'R','R':'L','C':'C'}[direction] if foot=='L' else direction)
            assert r['time_sec']==raw['time_sec'] and r['goal']==''
            for j in prior.SELECTED:
                for suffix in ('x','y','xc','yc','xn','yn'):
                    assert r[f'kp_{j}_{suffix}']==norm[(name,r['frame'])][f'kp_{j}_{suffix}']
            for column in prior.ANGLES:
                assert 0<=float(r[column])<=1
    return {'passed':True,'rows':len(rows),'videos':len(counts),'rows_per_video':20,
            'model_features':features,'all_model_features_finite':True,
            'existing_final_columns_present':True,'source_frame_and_time_preserved':True,
            'labels_and_right_foot_reflection_verified':True,'pelvis_origin_and_full_sequence_torso_median_verified':True,
            'angle_definition_max_error_vs_existing_csv':angle_error,'angles_in_unit_interval':True,
            'contact_endpoint_retained':True,'full_filenames_used_for_duplicate_numeric_ids':True}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('Main Data/150penalties_pose_keypoints.csv'))
    p.add_argument('--manifest',type=Path,default=Path('runs/150penalties_pose/segments.json'))
    p.add_argument('--output-dir',type=Path,default=Path('Main Data'))
    p.add_argument('--report-dir',type=Path,default=Path('runs/150penalties_preprocessing'))
    p.add_argument('--assume-clip-end-contact', action='store_true',
                   help='Include uncertain clips using their final source frame as assumed contact.')
    a=p.parse_args()
    reference=Path('Main Data/Combined_penalty_pose_keypoints_normalized_20frames_8joints_angles_scaled.csv')
    protected=[a.input,a.manifest,*Path('Main Data').glob('*.csv')]
    hashes={str(path):prior.digest(path) for path in protected}
    fields,source=prior.read_csv(a.input)
    specs=json.loads(a.manifest.read_text())
    assumed=[]
    if a.assume_clip_end_contact:
        for name, spec in specs.items():
            if spec['contact_status']=='uncertain_clip_end':
                endpoint=max(int(r['source_frame']) for r in source if r['video']==name)
                spec['contact_frame']=endpoint
                spec['contact_status']='assumed_clip_end'
                assumed.append(name)
    assert set(specs)=={r['video'] for r in source}
    prefix='150penalties_pose_keypoints'
    paths={key:a.output_dir/f'{prefix}_{suffix}.csv' for key,suffix in [
        ('canonical','canonical'),('normalized','normalized'),('sampled','normalized_20frames'),
        ('eight_joints','normalized_20frames_8joints'),('final','normalized_20frames_8joints_angles_scaled')]}
    if any(p.exists() for p in paths.values()) or (a.report_dir/'summary.json').exists():
        raise FileExistsError('Preprocessing outputs already exist; original files are protected.')
    a.output_dir.mkdir(parents=True,exist_ok=True);a.report_dir.mkdir(parents=True,exist_ok=True)
    adapted_sources,trimmed,bbox_repairs=prepare_sources(source)
    rows=[]
    for r,observed_bbox in adapted_sources:
        row=prior.adapt(r,specs[r['video']]);row['dataset']='150penalties'
        if specs[r['video']]['contact_status']=='assumed_clip_end':
            row['last_touch']=str(int(int(r['source_frame'])==specs[r['video']]['contact_frame']))
        row['source_bbox_observed_valid']=str(int(observed_bbox));row['bbox_interpolated']=str(int(not observed_bbox))
        rows.append(row)
    repairs,unfilled=prior.interpolate_internal(rows)
    canonical=[prior.canonicalize_with_quality(r) for r in rows]
    prior.write_csv(paths['canonical'],prior.FIELDS+['source_bbox_observed_valid','bbox_interpolated'],canonical)
    # The original shared routines validate their own saved outputs.
    with contextlib.redirect_stdout(io.StringIO()):
        normalization=prior.normalize_pose_csv(paths['canonical'],paths['normalized'])
        sampling=prior.extract_uniform_20frames(paths['normalized'],paths['sampled'])
        joint_selection=prior.extract_8_joints(paths['sampled'],paths['eight_joints'])
    final=prior.add_angles(paths['eight_joints'],paths['final'])
    validation=validate_final(paths['final'],reference,canonical,paths['normalized'],source,specs,sampling)
    for path,expected in hashes.items():assert prior.digest(path)==expected,path
    validation['raw_and_prior_csv_files_unchanged']=True
    prior.write_csv(a.report_dir/'trimmed_leading_frames.csv',fields,trimmed)
    excluded={r['kick_id'] for r in sampling['excluded_kicks']}
    prior.write_csv(a.report_dir/'excluded_final_raw_rows.csv',fields,[r for r in source if r['video'] in excluded])
    per_video=[]
    sampled=defaultdict(list)
    for r in final:sampled[r['kick_id']].append(int(r['source_frame']))
    for n in specs:
        seq=[r for r in canonical if r['kick_id']==n]
        per_video.append({'kick_id':n,'contact_status':specs[n]['contact_status'],
                         'first_retained_frame':int(seq[0]['frame']),'last_retained_frame':int(seq[-1]['frame']),
                         'retained_frames':len(seq),'included_in_final':n in sampled,'sampled_frames':sampled[n]})
    report={'input':str(a.input),'manifest':str(a.manifest),'reference':str(reference),'source_hashes':hashes,
            'assumed_clip_end_contact_videos':assumed,
            'input_rows':len(source),'input_videos':len(specs),'retained_full_rows':len(rows),
            'trimmed_leading_frames':len(trimmed),'trimmed_by_video':dict(Counter(r['video'] for r in trimmed)),
            'leading_trim_rule':'Start at first frame with all selected 8 joints observed, as in first15 011; no boundary extrapolation.',
            'bbox_interpolation':'Internal bracketed missing bboxes only, used for coordinate centering/reflection; source remains unchanged.',
            'bbox_repairs':bbox_repairs,'interpolated_joint_observations':len(repairs),'joint_repairs':repairs,
            'remaining_unfilled_joints':unfilled,'final_rows':len(final),'final_videos':len({r['kick_id'] for r in final}),
            'excluded_final':sampling['excluded_kicks'],'sampling':sampling['sampling'],
            'canonicalization':'Existing canonicalize_main_data.canonicalize_row and prior quality-field swaps; left kicks become right and L/R target labels swap.',
            'normalization':'Existing spatial_normalize_pose.normalize_pose_csv; pelvis midpoint / median torso length over full retained sequence.',
            'joint_interpolation':'Existing first15 internal linear interpolation in source frame time; no extrapolation.',
            'selected_joints':list(prior.SELECTED),'angles':'Same prior.add_angles: acos(dot/norms)/pi for left/right knee and hip.',
            'labels_in_final':dict(Counter(r['kick_direction'] for r in final[::20])),
            'outputs':{k:str(p) for k,p in paths.items()},'output_sha256':{k:prior.digest(p) for k,p in paths.items()},
            'per_video':per_video,'validation':validation}
    (a.report_dir/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({k:report[k] for k in ['input_rows','input_videos','retained_full_rows','trimmed_by_video','interpolated_joint_observations','bbox_repairs','final_rows','final_videos','excluded_final','labels_in_final','validation']},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
