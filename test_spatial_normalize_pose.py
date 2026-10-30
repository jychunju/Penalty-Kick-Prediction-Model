"""Behavior checks for full-sequence normalization and metadata preservation."""
import contextlib
import csv
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from spatial_normalize_pose import ADDED_COLUMNS, normalize_pose_csv


def pose_row(kick_id, frame, torso, *, dataset=None):
    # Deliberately tilted torso: rotation normalization would break these tests.
    hx, hy = 100+float(frame)*0.7, 200-float(frame)*0.3
    row = {"kick_id": kick_id, "frame": str(frame), "track_id": "0007",
           "camera_direction": "R", "r_or_l": "R", "kick_direction": "L",
           "last_touch": " 0 ", "goal": "", "bbox_w": "999", "bbox_h": "888"}
    if dataset is not None:
        row = {"dataset": dataset, **row}
    for joint in range(5, 17):
        row[f"kp_{joint}_xc"] = str(hx+(joint-10)*2)
        row[f"kp_{joint}_yc"] = str(hy+(joint-10)*3)
    row.update(kp_11_xc=str(hx-1), kp_11_yc=str(hy),
               kp_12_xc=str(hx+1), kp_12_yc=str(hy),
               kp_5_xc=str(hx+0.6*torso-2), kp_5_yc=str(hy-0.8*torso),
               kp_6_xc=str(hx+0.6*torso+2), kp_6_yc=str(hy-0.8*torso))
    return row


class SpatialNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def run_normalization(self, rows):
        source, output = self.directory/"input.csv", self.directory/"output.csv"
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        original_bytes = source.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            report = normalize_pose_csv(source, output, sample_count=3)
        self.assertEqual(source.read_bytes(), original_bytes)
        with output.open(newline="") as stream:
            reader = csv.DictReader(stream)
            saved = list(reader)
            self.assertEqual(reader.fieldnames, list(rows[0])+list(ADDED_COLUMNS))
        self.assertEqual(len(saved), len(rows))
        for before, after in zip(rows, saved):
            self.assertEqual({c: after[c] for c in before}, before)
        return report, saved

    def test_full_sequence_scale_original_order_and_tilt(self):
        rows = []
        # First twenty torso lengths are 2, but the FULL 41-row median is 10.
        # Frames are intentionally descending and track/camera values vary.
        for index in range(41):
            row = pose_row("001", 100-index, 2 if index < 20 else 10)
            row["track_id"] = "0007" if index < 20 else "0009"
            row["camera_direction"] = "L" if index < 20 else "R"
            rows.append(row)
        report, saved = self.run_normalization(rows)
        self.assertEqual(report["total_rows"], 41)
        self.assertEqual(report["clip_group_count"], 1)
        self.assertTrue(all(np.isclose(float(r["clip_torso_scale"]), 10) for r in saved))
        self.assertEqual(report["random_examples"][0]["first_row_frame"], "100")
        for old, new in zip(rows, saved):
            hx = (float(old["kp_11_xc"])+float(old["kp_12_xc"]))/2
            hy = (float(old["kp_11_yc"])+float(old["kp_12_yc"]))/2
            for joint in range(5, 17):
                self.assertAlmostEqual(float(new[f"kp_{joint}_xn"]),
                                       (float(old[f"kp_{joint}_xc"])-hx)/10)
                self.assertAlmostEqual(float(new[f"kp_{joint}_yn"]),
                                       (float(old[f"kp_{joint}_yc"])-hy)/10)
            sx = (float(new["kp_5_xn"])+float(new["kp_6_xn"]))/2
            sy = (float(new["kp_5_yn"])+float(new["kp_6_yn"]))/2
            self.assertGreater(sx, 0)  # Torso was not forced vertical.
            self.assertAlmostEqual(sx/-sy, 0.75)

    def test_interleaved_dataset_ids_get_separate_fixed_scales(self):
        rows = [pose_row("1", 9, 2, dataset="EPL"),
                pose_row("1", 2, 10, dataset="CLT"),
                pose_row("1", 1, 4, dataset="EPL"),
                pose_row("1", 7, 14, dataset="CLT")]
        report, saved = self.run_normalization(rows)
        self.assertEqual(report["unique_kick_id_count"], 1)
        self.assertEqual(report["clip_group_count"], 2)
        self.assertEqual(report["group_columns"], ["dataset", "kick_id"])
        np.testing.assert_allclose([float(r["clip_torso_scale"]) for r in saved], [3, 12, 3, 12])

    def test_invalid_torso_and_partial_missingness_no_bbox_fallback(self):
        good = pose_row("good", 3, 5)
        partial = pose_row("good", 0, 0)  # Zero length is excluded from median.
        partial["kp_15_xc"] = ""
        partial["kp_15_yc"] = "NaN"
        invalid_zero = pose_row("bad", 8, 0)
        invalid_nan = pose_row("bad", 7, 5)
        invalid_nan["kp_11_xc"] = "NaN"
        invalid_inf = pose_row("bad", 5, 5)
        invalid_inf["kp_5_yc"] = "inf"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report, saved = self.run_normalization([good, invalid_zero, partial,
                                                     invalid_nan, invalid_inf])
        self.assertEqual(len(caught), 1)
        self.assertIn("No bbox fallback", str(caught[0].message))
        self.assertEqual(report["failed_kicks"], [{"kick_id": "bad"}])
        self.assertAlmostEqual(float(saved[0]["clip_torso_scale"]), 5)
        self.assertAlmostEqual(float(saved[2]["clip_torso_scale"]), 5)
        self.assertEqual(saved[2]["kp_15_xn"], "")
        self.assertEqual(saved[2]["kp_15_yn"], "")
        for index in (1, 3, 4):
            self.assertEqual(saved[index]["clip_torso_scale"], "")
            self.assertTrue(all(saved[index][f"kp_{j}_{a}"] == ""
                                for j in range(5, 17) for a in ("xn", "yn")))
        self.assertEqual(report["validation"]["normalized_nan_count"], 3*24+2)
        self.assertEqual(report["validation"]["normalized_inf_count"], 0)

    def test_input_overwrite_and_existing_normalized_columns_rejected(self):
        source = self.directory/"source.csv"
        source.write_text("unchanged input")
        with self.assertRaises(ValueError):
            normalize_pose_csv(source, source)
        self.assertEqual(source.read_text(), "unchanged input")
        row = pose_row("1", 0, 5)
        row["kp_5_xn"] = "123"
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        with self.assertRaises(ValueError):
            normalize_pose_csv(source, self.directory/"output.csv")


if __name__ == "__main__":
    unittest.main()
