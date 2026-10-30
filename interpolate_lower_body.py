"""Repair missing lower-body joints with bracketed source-frame interpolation."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from normalize_combined_20frames import JOINTS, SWAP, read, number, frame, select_source_group


def main():
    root = Path(__file__).resolve().parent
    rows, fields = read(root / 'penalty_kicks_20frames_combined_normalized.csv')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['kick_id'])].append(row)
    source_paths = {'CLT': 'merged_full_data_001_030_trimmed.csv',
                    'EPL': 'kicker_pose_keypoints.csv', 'Women': 'penalty_pose_keypoints.csv'}
    sources = {}
    report = []
    for (dataset, kick), seq in groups.items():
        seq.sort(key=lambda r: int(r['seq_20_idx']))
        times = np.array([float(r['resolved_source_frame']) for r in seq])
        raw = np.array([[ [number(r[f'kp_{j}_x']), number(r[f'kp_{j}_y'])]
                         for j in JOINTS] for r in seq], dtype=float)
        missing = (~np.isfinite(raw).all(axis=2)) | (raw <= 0).any(axis=2)
        repaired = raw.copy()
        if missing.any():
            if dataset not in sources:
                sources[dataset] = read(root / source_paths[dataset])[0]
            _, source, *_ = select_source_group(dataset, kick, seq, sources[dataset])
            for ji, joint in enumerate(JOINTS):
                anchors = defaultdict(list)
                for r in source:
                    fr = frame(r)
                    xy = np.array([number(r.get(f'kp_{joint}_x')), number(r.get(f'kp_{joint}_y'))], dtype=float)
                    if fr is not None and np.isfinite(xy).all() and (xy > 0).all():
                        anchors[fr].append(xy)
                # Keep valid observations in this exact combined sequence.
                for t in np.flatnonzero(~missing[:, ji]):
                    anchors[times[t]] = [raw[t, ji]]
                keys = sorted(anchors)
                for t in np.flatnonzero(missing[:, ji]):
                    target = times[t]
                    left = [f for f in keys if f < target]
                    right = [f for f in keys if f > target]
                    if not left or not right:
                        raise ValueError(f'No bracketing observations: {dataset}:{kick}, joint {joint}, frame {target}')
                    lo, hi = left[-1], right[0]
                    a = np.median(anchors[lo], axis=0); b = np.median(anchors[hi], axis=0)
                    repaired[t, ji] = a + (b-a) * (target-lo)/(hi-lo)
                    report.append({'dataset': dataset, 'kick_id': kick, 'joint_id': joint,
                                   'frame': int(target), 'left_frame': int(lo), 'right_frame': int(hi)})
        centered = repaired - ((repaired[:, 0] + repaired[:, 1])/2)[:, None, :]
        positions = centered / float(seq[0]['normalization_scale'])
        canonical_mask = missing.copy()
        if seq[0]['mirrored_to_right_foot'] == '1':
            positions[..., 0] *= -1
            positions = positions[:, SWAP]
            canonical_mask = missing[:, SWAP]
        velocity = np.zeros_like(positions)
        velocity[1:] = np.diff(positions, axis=0) / (np.diff(times)/float(seq[0]['fps']))[:, None, None]
        for t, row in enumerate(seq):
            row['lower_body_interpolated'] = int(missing[t].any())
            for ji, joint in enumerate(JOINTS):
                row[f'kp_{joint}_interpolated'] = int(missing[t, ji])
                row[f'kp_{joint}_canonical_interpolated'] = int(canonical_mask[t, ji])
                for axis, suffix in enumerate(('x', 'y')):
                    row[f'kp_{joint}_{suffix}_filled'] = repaired[t, ji, axis]
                    row[f'kp_{joint}_{suffix}c_filled'] = centered[t, ji, axis]
                    row[f'kp_{joint}_{suffix}_norm'] = positions[t, ji, axis]
                    row[f'kp_{joint}_v{suffix}_per_sec'] = velocity[t, ji, axis]
    added = ['lower_body_interpolated']
    for j in JOINTS:
        added += [f'kp_{j}_interpolated', f'kp_{j}_canonical_interpolated',
                  f'kp_{j}_x_filled', f'kp_{j}_y_filled', f'kp_{j}_xc_filled', f'kp_{j}_yc_filled']
    out = root / 'penalty_kicks_20frames_combined_normalized_interpolated.csv'
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields+added); w.writeheader(); w.writerows(rows)
    out.with_suffix('.json').write_text(json.dumps({'method': 'linear interpolation between valid source-frame observations; no extrapolation',
        'rows': len(rows), 'sequences': len(groups), 'interpolated_joint_observations': len(report),
        'interpolated_rows': sum(int(r['lower_body_interpolated']) for r in rows), 'repairs': report}, indent=2))
    print(out)
    print('repaired joints:', len(report), 'repaired rows:', sum(int(r['lower_body_interpolated']) for r in rows))


if __name__ == '__main__':
    main()
