#!/usr/bin/env python3
"""Add r_or_l to the combined 20-frame CSV from its source datasets."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def foot_by_kick(path: Path) -> dict[str, str]:
    values: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            kick = (row.get("kick_id") or "").strip()
            foot = (row.get("r_or_l") or "").strip().upper()
            if kick and foot:
                values.setdefault(kick, set()).add(foot)
    bad = {kick: feet for kick, feet in values.items()
           if len(feet) != 1 or not feet <= {"L", "R"}}
    if bad:
        raise ValueError(f"Invalid or conflicting source foot labels: {bad}")
    return {kick: next(iter(feet)) for kick, feet in values.items()}


def women_source_id(combined_id: str) -> str:
    match = re.fullmatch(r"women_(\d+)월\s*(\d+)일", combined_id)
    if match:
        return f"{int(match.group(1))}-{int(match.group(2))}"
    match = re.fullmatch(r"women_(\d+)-(\d+)", combined_id)
    if match:
        return f"{int(match.group(1))}-{int(match.group(2))}"
    raise ValueError(f"Unrecognized Women kick_id: {combined_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("penalty_kicks_20frames_combined.csv"))
    parser.add_argument("--output", type=Path, default=Path("penalty_kicks_20frames_combined_with_foot.csv"))
    parser.add_argument("--epl-source", type=Path, default=Path("kicker_pose_keypoints.csv"))
    parser.add_argument("--women-source", type=Path, default=Path("penalty_pose_keypoints.csv"))
    args = parser.parse_args()

    epl = foot_by_kick(args.epl_source)
    women = foot_by_kick(args.women_source)
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "r_or_l" in fields:
        raise ValueError("Input already contains r_or_l")

    unmatched = []
    for row in rows:
        dataset = row["dataset"].strip()
        combined_id = row["kick_id"].strip()
        if dataset == "CLT":
            foot = "R"
        elif dataset == "EPL":
            source_id = combined_id.removeprefix("epl_")
            foot = epl.get(source_id)
        elif dataset == "Women":
            # penalty_pose_keypoints.csv has a consistent R label for all 132
            # source kicks. Some combined IDs were spreadsheet-coerced into
            # localized dates (and can be order-ambiguous), so no ID remapping
            # is needed for this constant source label.
            foot = "R" if women and set(women.values()) == {"R"} else None
        else:
            foot = None
        if foot not in {"L", "R"}:
            unmatched.append((dataset, combined_id))
        else:
            row["r_or_l"] = foot
    if unmatched:
        raise ValueError(f"Unmatched rows ({len(unmatched)}): {unmatched[:20]}")

    insert_at = fields.index("kick_direction")
    output_fields = fields[:insert_at] + ["r_or_l"] + fields[insert_at:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"wrote {args.output}: {len(rows)} rows")


if __name__ == "__main__":
    main()
