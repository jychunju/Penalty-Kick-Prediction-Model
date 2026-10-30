"""Concatenate the processed legacy and 150penalties tables with a union schema."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'Main Data'
DEFAULT_INPUTS = [DATA / f'{prefix}_normalized_20frames_8joints_angles_scaled.csv'
                  for prefix in ['Combined_penalty_pose_keypoints', '150penalties_pose_keypoints']]
DEFAULT_OUTPUT = DATA / 'Combined_penalty_pose_keypoints_normalized_20frames_8joints_angles_scaled_with_150penalties.csv'


def merge(inputs, output):
    fields, rows = [], []
    for path in inputs:
        with Path(path).open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f'Invalid columns: {path}')
            fields.extend(name for name in reader.fieldnames if name not in fields)
            for row in reader:
                if None in row or None in row.values():
                    raise ValueError(f'Malformed CSV: {path}')
                rows.append(row)
    with Path(output).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', default=DEFAULT_INPUTS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f'Merged {merge(args.inputs, args.output)} rows into {args.output}')
