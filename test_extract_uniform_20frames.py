"""Regression tests for observed-frame sampling without coordinate changes."""
import contextlib
import csv
import io
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from extract_uniform_20frames import (
    assess_candidate, candidate_sequences, extract_uniform_20frames,
    uniform_observed_indices,
)


def sequence(dataset="EPL", kick="1", camera="C", track="10", frames=range(25), touch=24):
    return [dict(dataset=dataset, kick_id=kick, camera_direction=camera,
                 track_id=track, kicker_track_id_main=track, frame=str(frame),
                 last_touch="1" if frame == touch else "0",
                 kp_5_xn="0.1234567890123456", kick_direction="L")
            for frame in frames]


class SamplingTests(unittest.TestCase):
    def test_uniform_dense_and_sparse_observations(self):
        for frames in (list(range(80)), list(range(15))+list(range(70, 90))):
            selected, targets = uniform_observed_indices(frames)
            self.assertEqual(len(selected), 20)
            self.assertEqual(len(set(selected)), 20)
            self.assertEqual(selected, sorted(selected))
            self.assertEqual((selected[0], selected[-1]), (0, len(frames)-1))
            np.testing.assert_allclose(targets, np.linspace(frames[0], frames[-1], 20))

    def test_uniform_assignment_matches_brute_force_optimum(self):
        frames = np.array([0, 1, 3, 4, 5, 20, 21, 22, 40])
        for count in (2, 3, 4, 5):
            selected, targets = uniform_observed_indices(frames, count)
            error = np.abs(frames[selected]-targets).sum()
            optimum = min(np.abs(frames[[0, *middle, len(frames)-1]]-targets).sum()
                          for middle in itertools.combinations(range(1, len(frames)-1), count-2))
            self.assertAlmostEqual(error, optimum)

    def test_twenty_or_less_excluded_and_post_touch_not_counted(self):
        for frames, touch, valid in ((range(20), 19, False), (range(21), 20, True),
                                      (range(30), 19, False), (range(30), 24, True)):
            rows = sequence(frames=frames, touch=touch)
            chosen, report = assess_candidate(candidate_sequences(rows)[0], rows)
            self.assertEqual(chosen is not None, valid)
            if valid:
                self.assertEqual(rows[chosen[0]]["frame"], "0")
                self.assertEqual(rows[chosen[-1]]["frame"], str(touch))
                self.assertEqual(rows[chosen[-1]]["last_touch"], "1")

    def test_tracker_handoff_merges_but_overlapping_timelines_do_not(self):
        rows = sequence(track="389", frames=range(14, 41), touch=48)
        rows += sequence(track="391", frames=range(41, 49), touch=48)
        candidates = candidate_sequences(rows)
        self.assertEqual(len(candidates), 1)
        chosen, report = assess_candidate(candidates[0], rows)
        self.assertEqual(report["source_track_ids"], ["389", "391"])
        self.assertEqual(report["available_unique_frames_through_touch"], 35)
        self.assertEqual((rows[chosen[0]]["frame"], rows[chosen[-1]]["frame"]), ("14", "48"))
        rows = sequence(track="1", frames=range(25)) + sequence(track="2", frames=range(30), touch=29)
        self.assertEqual(len(candidate_sequences(rows)), 2)

    def test_duplicate_endpoint_prefers_touch_marker(self):
        rows = sequence(frames=range(25), touch=None)
        rows += sequence(frames=[24], touch=24)
        chosen, report = assess_candidate(candidate_sequences(rows)[0], rows)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(chosen[-1], 25)
        self.assertEqual(report["available_unique_frames_through_touch"], 25)

    def test_missing_ambiguous_or_invalid_frame_excluded(self):
        for expected in ("missing_last_touch", "ambiguous_last_touch", "invalid_frame_number"):
            rows = sequence()
            if expected == "missing_last_touch":
                rows[-1]["last_touch"] = "0"
            elif expected == "ambiguous_last_touch":
                rows[15]["last_touch"] = "1"
            else:
                rows[10]["frame"] = ""
            chosen, report = assess_candidate(candidate_sequences(rows)[0], rows)
            self.assertIsNone(chosen)
            self.assertEqual(report["status"], expected)

    def test_integration_first_eligible_sequence_and_original_values(self):
        # EPL's first camera has no touch: use its second camera, not the third.
        rows = sequence(camera="C", touch=None)
        rows += list(reversed(sequence(camera="R", track="20")))
        rows += sequence(camera="L", track="30", frames=range(30), touch=29)
        rows += sequence(dataset="Women", frames=range(23), touch=22)
        rows += sequence(dataset="CLT", frames=range(20), touch=19)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"source.csv", Path(directory)/"sampled.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            original = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                report = extract_uniform_20frames(source, output)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(report["selected_kicks"], 2)
            self.assertEqual(report["output_rows"], 40)
            self.assertEqual(report["excluded_kicks_count"], 1)
            self.assertEqual(report["selected_sequences"][0]["camera_direction"], "R")
            with output.open(newline="", encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(saved[0]["frame"], "0")
            self.assertEqual(saved[19]["frame"], "24")
            self.assertTrue(all(row["kp_5_xn"] == "0.1234567890123456" for row in saved))
            self.assertTrue(all(row["kick_direction"] == "L" for row in saved))
            self.assertEqual(report["validation"]["duplicate_selected_timestamps"], 0)
            self.assertTrue(all(value is True for key, value in report["validation"].items()
                                if key != "duplicate_selected_timestamps"))
            with self.assertRaises(FileExistsError):
                extract_uniform_20frames(source, output)


if __name__ == "__main__":
    unittest.main()
