#!/usr/bin/env python3
"""Unit tests for balltrack trajectory logic."""

import unittest
from types import SimpleNamespace

import numpy as np

import balltrack as bt


def _args(angle="dtl", fps_hint="slowmo"):
    return SimpleNamespace(angle=angle, fps_hint=fps_hint)


class TrajectoryFitTests(unittest.TestCase):
    def test_parabolic_track_scores_better_than_straight_junk(self):
        # Ballistic arc in image coords (y concave up).
        arc = {i: (100.0 + 8 * i, 400.0 - 5 * i + 0.15 * i * i) for i in range(10)}
        straight = {i: (100.0 + 8 * i, 300.0) for i in range(10)}

        arc_score = bt.trajectory_score(arc, _args())
        straight_score = bt.trajectory_score(straight, _args())
        self.assertGreater(arc_score, straight_score)

    def test_interpolate_fills_gaps(self):
        sparse = {0: (10.0, 10.0), 2: (30.0, 14.0), 4: (50.0, 22.0), 6: (70.0, 34.0), 8: (90.0, 50.0)}
        dense = bt.interpolate_track(sparse)
        self.assertIn(1, dense)
        self.assertIn(3, dense)
        self.assertIn(5, dense)
        self.assertEqual(len(dense), 9)

    def test_gap_aware_linking_reconnects_after_one_missed_frame(self):
        per_frame = {
            0: [(0.0, 0.0, 4)],
            1: [(10.0, 2.0, 4)],
            # frame 2 intentionally missing
            3: [(30.0, 6.0, 4)],
            4: [(40.0, 8.0, 4)],
            5: [(50.0, 11.0, 4)],
            6: [(60.0, 15.0, 4)],
        }
        track = bt.build_trajectory(per_frame, _args(angle="auto"))
        self.assertGreaterEqual(len(track), 5)
        self.assertIn(3, track)


class DetectCandidatesTests(unittest.TestCase):
    def test_detects_moving_blob(self):
        args = SimpleNamespace(
            thresh=20, min_area=1, max_area=500, min_circularity=0.1,
        )
        prev = np.zeros((100, 100), dtype=np.uint8)
        gray = prev.copy()
        cv2 = __import__("cv2")
        cv2.circle(gray, (50, 50), 4, 255, -1)

        cands, _ = bt.detect_candidates(prev, gray, args)
        self.assertGreaterEqual(len(cands), 1)


class VideoWriterTests(unittest.TestCase):
    def test_open_video_writer_invalid_path_raises(self):
        with self.assertRaises(SystemExit):
            bt.open_video_writer("", 30.0, (640, 480))


if __name__ == "__main__":
    unittest.main()
