#!/usr/bin/env python3
"""Validate actual pose/video artifacts, including decoding every output frame."""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


def probe(path):
    return json.loads(subprocess.check_output([
        "/usr/bin/ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(path)]))


def main():
    root = Path("runs/penaltyfirst15_pose")
    manifest = json.loads((root / "segments.json").read_text())
    checks = []
    total, detected, valid_count, lower_count = 0, 0, 0, 0
    overview_frames = 0
    chapters = []
    for name, spec in manifest.items():
        out = root / Path(name).stem
        graph = np.load(out / "kicker_graph_17j.npz")
        meta = json.loads((out / "metadata.json").read_text())
        with (out / "kicker_keypoints_17j.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        n = spec["contact_frame"] - spec["runup_start_frame"] + 1
        assert len(rows) == n == meta["extracted_frames"], name
        assert graph["xy_pixels"].shape == (n, 17, 2), name
        expected = np.arange(spec["runup_start_frame"], spec["contact_frame"] + 1)
        np.testing.assert_array_equal(graph["source_frames"], expected)
        assert [int(r["source_frame"]) for r in rows] == expected.tolist(), name
        assert graph["adjacency"].shape == (17, 17), name
        np.testing.assert_array_equal(graph["adjacency"], graph["adjacency"].T)
        assert graph["edge_index"].shape == (2, 16), name
        assert np.isnan(graph["xy_pixels"][~graph["valid"]]).all(), name
        assert np.isfinite(graph["xy_pixels"][graph["valid"]]).all(), name
        assert (graph["confidence"][graph["valid"]] >= .25).all(), name
        assert np.isnan(graph["xy_normalized"][~graph["normalized_valid"]]).all(), name
        assert np.isnan(graph["velocity_normalized_per_sec"][0]).all(), name
        assert int(graph["is_contact_candidate"].sum()) == 1, name
        assert int(graph["is_contact"].sum()) == (0 if spec["contact_status"] == "uncertain_clip_end" else 1), name
        assert sum(int(r["is_contact"]) for r in rows) == int(graph["is_contact"].sum()), name
        for t, row in enumerate(rows):
            for j, joint in enumerate(graph["joint_names"]):
                assert int(row[f"{joint}_valid"]) == graph["valid"][t, j], name
                if graph["valid"][t, j]:
                    np.testing.assert_allclose([float(row[f"{joint}_x"]), float(row[f"{joint}_y"])], graph["xy_pixels"][t, j], rtol=1e-6)
                else:
                    assert row[f"{joint}_x"] == row[f"{joint}_y"] == "", name
        video_checks = []
        for filename in ("kicker_overlay.mp4", "kicker_overlay_0.5x.mp4", "kicker_review_0.5x.mp4"):
            info = probe(out / filename)
            stream = info["streams"][0]
            count = int(stream["nb_read_frames"])
            assert stream["codec_name"] == "h264", name
            assert stream["r_frame_rate"] == "25/1", name
            if filename == "kicker_overlay.mp4":
                assert count == n and (stream["width"], stream["height"]) == (1920, 1080), name
            else:
                assert abs(count - 2 * n) <= 1, name
            if filename == "kicker_review_0.5x.mp4":
                assert (stream["width"], stream["height"]) == (1920, 720), name
                chapters.append({"video": name, "overview_start_sec": overview_frames / 25,
                                 "overview_end_sec": (overview_frames + count) / 25,
                                 "runup_start_source_frame": spec["runup_start_frame"],
                                 "end_source_frame": spec["contact_frame"], "contact_status": spec["contact_status"]})
                overview_frames += count
            video_checks.append({"file": filename, "decoded_frames": count, "duration_sec": float(info["format"]["duration"])})
        total += n
        detected += meta["pose_detected_frames"]
        valid_count += int(graph["valid"].sum())
        lower_count += int(graph["valid"][:, 11:17].sum())
        checks.append({"video": name, "graph_frames": n, "passed": True, "videos": video_checks})
        print(f"PASS {name}: {n} pose frames, 3 videos fully decoded", flush=True)
    with (root / "all_kicker_keypoints_17j.csv").open(newline="") as f:
        combined = list(csv.DictReader(f))
    assert len(combined) == total
    assert set(r["video"] for r in combined) == set(manifest)
    assert (root / "all_kicker_keypoints_17j.csv").read_bytes() == Path("Main Data/penaltyfirst15_pose_keypoints.csv").read_bytes()
    assert int(probe(root / "all_kickers_review_0.5x.mp4")["streams"][0]["nb_read_frames"]) == overview_frames
    with (root / "overview_chapters.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(chapters[0]))
        writer.writeheader()
        writer.writerows(chapters)
    report = {"passed": True, "videos": len(manifest), "pose_frames": total,
              "detected_pose_frames": detected, "pose_detection_rate": detected / total,
              "joint_valid_fraction": valid_count / (total * 17),
              "lower_body_valid_fraction": lower_count / (total * 6),
              "overview_frames": overview_frames, "overview_duration_sec": overview_frames / 25,
              "visually_estimated_contacts": 15, "uncertain_clip_end": ["001-R-L-L.mp4"],
              "checks": checks}
    (root / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
