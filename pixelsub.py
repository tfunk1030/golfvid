#!/usr/bin/env python3
"""
pixelsub.py — motion isolation for swing videos via frame differencing.

Inspired by the "event sensor" idea: only show pixels that CHANGED between
frames, so the static background drops to black and the moving stuff (body,
club, ball) glows. Useful for tight crops + cleaner input to vision models.

Usage:
    python pixelsub.py swing.mp4 -o out.mp4
    python pixelsub.py swing.mp4 -o out.mp4 --mode mog2 --crop
    python pixelsub.py swing.mp4 -o out.mp4 --mode diff --thresh 18 --side-by-side

Modes:
    diff   Simple frame-to-frame difference (closest to the "pixelsub" demo).
           Fast, edge-glow look. Good for visualizing the swing.
    mog2   Background subtractor (learns the static scene). Cleaner silhouettes,
           better when the camera is locked off on a tripod. Better for crops.
    knn    Same idea as mog2, different algorithm. Try it if mog2 is noisy.
"""

import argparse
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Frame-differencing motion isolation for swing video.")
    p.add_argument("input", help="input video path")
    p.add_argument("-o", "--output", default="pixelsub_out.mp4", help="output video path")
    p.add_argument("--mode", choices=["diff", "mog2", "knn"], default="diff")
    p.add_argument("--thresh", type=int, default=18,
                   help="diff mode: brightness-change threshold (lower = more sensitive)")
    p.add_argument("--blur", type=int, default=3,
                   help="pre-blur kernel (odd number) to kill sensor noise; 0 to disable")
    p.add_argument("--side-by-side", action="store_true",
                   help="output original | motion stacked horizontally")
    p.add_argument("--crop", action="store_true",
                   help="auto-crop tight around the largest motion region each frame")
    p.add_argument("--pad", type=int, default=40, help="padding (px) around the auto-crop box")
    return p.parse_args()


def make_diff(prev_gray, gray, thresh):
    """Closest to the demo: absolute difference between consecutive frames."""
    d = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
    return mask


def largest_motion_box(mask, pad, shape):
    """Return (x, y, w, h) around the biggest blob of motion, or None."""
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:  # ignore tiny noise blobs
        return None
    x, y, w, h = cv2.boundingRect(c)
    H, W = shape[:2]
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return x0, y0, x1 - x0, y1 - y0


def main():
    a = parse_args()
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {a.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg = None
    if a.mode == "mog2":
        bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    elif a.mode == "knn":
        bg = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    out_w = W * 2 if a.side_by_side else W
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(a.output, fourcc, fps, (out_w, H))

    prev_gray = None
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if a.blur and a.blur >= 3:
            k = a.blur | 1  # force odd
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        if a.mode == "diff":
            if prev_gray is None:
                prev_gray = gray
                continue
            mask = make_diff(prev_gray, gray, a.thresh)
            prev_gray = gray
        else:
            mask = bg.apply(frame)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # the glowing-outline visual: white motion on black
        motion = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if a.crop:
            box = largest_motion_box(mask, a.pad, frame.shape)
            if box:
                x, y, w, h = box
                motion = motion[y:y + h, x:x + w]
                motion = cv2.resize(motion, (W, H))

        if a.side_by_side:
            out_frame = np.hstack([frame, motion])
        else:
            out_frame = motion

        writer.write(out_frame)

    cap.release()
    writer.release()
    print(f"Done. Wrote {a.output} ({frames} frames @ {fps:.0f} fps, mode={a.mode})")


if __name__ == "__main__":
    main()
