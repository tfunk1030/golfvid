#!/usr/bin/env python3
"""
balltrack.py — golf ball flight tracking via motion isolation + trajectory fitting.

Pipeline:
  1. Frame-difference to find everything that MOVED (ball, club, body, noise).
  2. Filter blobs by size/shape to keep ball-like candidates.
  3. Link candidates into chains with gap-aware velocity prediction.
  4. Score chains with a parabolic fit (ballistic arc) and pick the best.
  5. Interpolate missed frames from the fit; draw trace and/or emit JSON crops.

WHY THIS IS HARD (read this): the ball is small, fast, and motion-blurred.
No threshold catches it perfectly. The trajectory fit is what rescues you --
it tolerates missed frames and rejects junk that doesn't move like a ball.
You WILL need to tune --min-area / --max-area / --thresh to your footage.

Usage:
    python balltrack.py swing.mp4 -o traced.mp4
    python balltrack.py swing.mp4 -o traced.mp4 --angle dtl --fps-hint slowmo
    python balltrack.py swing.mp4 -o traced.mp4 --dump-json track.json
    python balltrack.py swing.mp4 -o debug.mp4 --debug

Dependencies: opencv-python, numpy
"""

import argparse
import json
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Golf ball flight tracking.")
    p.add_argument("input", help="input video path")
    p.add_argument("-o", "--output", default="balltrack_out.mp4", help="output video path")
    p.add_argument("--angle", choices=["faceon", "dtl", "auto"], default="dtl",
                   help="faceon = motion glow only; dtl = down-the-line tracking heuristics; "
                        "auto = tracking with no angle-specific direction checks")
    p.add_argument("--fps-hint", choices=["normal", "slowmo"], default="slowmo",
                   help="slowmo loosens the speed gate (ball moves fewer px/frame)")
    p.add_argument("--thresh", type=int, default=22,
                   help="motion threshold; raise to suppress shake, lower to catch a faint ball")
    p.add_argument("--min-area", type=int, default=1,
                   help="min blob area (px). DTL ball shrinks as it recedes; keep this very low.")
    p.add_argument("--max-area", type=int, default=400,
                   help="max blob area (px). Rejects body/club blobs that are huge.")
    p.add_argument("--min-circularity", type=float, default=0.20,
                   help="0..1; receding/blurred balls are loosely round. DTL wants this low.")
    p.add_argument("--trace-fade", action="store_true",
                   help="fade the trace tail instead of a solid line")
    p.add_argument("--dump-json", default=None, help="write per-frame ball coords + crop boxes to JSON")
    p.add_argument("--crop-pad", type=int, default=30, help="padding (px) for emitted crop boxes")
    p.add_argument("--debug", action="store_true",
                   help="render ALL candidate blobs (red) so you can tune thresholds")
    p.add_argument("--motion-view", action="store_true",
                   help="skip ball tracking; output motion glow. Auto-on for --angle faceon.")
    p.add_argument("--side-by-side", action="store_true",
                   help="stack original | motion panel horizontally (face-on or DTL)")
    p.add_argument("--original-bg", action="store_true",
                   help="DTL: draw trace on full-color video instead of motion-diff (old behavior)")
    p.add_argument("--view-mode", choices=["diff", "silhouette", "outline", "trail"], default="diff",
                   help="face-on look. diff=raw frame-difference (default); silhouette=solid body; "
                        "outline=edge glow; trail=ghosted motion path.")
    p.add_argument("--trail-decay", type=float, default=0.92,
                   help="trail mode: 0..1, higher = longer-lasting trail.")
    p.add_argument("--motion-thresh", type=int, default=16,
                   help="face-on motion sensitivity only; lower = more body/club glow (default 16)")
    p.add_argument("--out-fps", type=float, default=None,
                   help="override output playback fps. Default: match source (keeps full slo-mo). "
                        "Set e.g. 30 only if you want a faster, shorter preview.")
    return p.parse_args()


def output_fps(source_fps, override):
    """Preserve source slo-mo fps unless user explicitly overrides."""
    if override is not None and override > 0:
        return float(override)
    return float(source_fps or 30.0)


def open_video_writer(path, fps, size):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise SystemExit(f"Could not open output video: {path}")
    return writer


def motion_diff_bgr(prev_gray, gray, thresh, black_bgr):
    """Face-on style motion panel: white motion on black."""
    if prev_gray is None:
        return black_bgr.copy()
    d = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def draw_debug_boxes(out, mask, a):
    if mask is None:
        return
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if a.min_area <= area <= a.max_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)


def draw_track_overlay(out, track, ordered, idx, a):
    pts = [track[f] for f in ordered if f <= idx]
    for j in range(1, len(pts)):
        p0 = (int(pts[j - 1][0]), int(pts[j - 1][1]))
        p1 = (int(pts[j][0]), int(pts[j][1]))
        if a.trace_fade:
            alpha = j / len(pts)
            color = (0, int(180 * alpha) + 50, int(255 * alpha))
            cv2.line(out, p0, p1, color, 2, cv2.LINE_AA)
        else:
            cv2.line(out, p0, p1, (0, 200, 255), 2, cv2.LINE_AA)
    if idx in track:
        x, y = track[idx]
        cv2.circle(out, (int(x), int(y)), 6, (0, 255, 0), 2)


def render_motion_view(a, fps, W, H):
    """Face-on path: no ball tracking. Several looks via --view-mode."""
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {a.input}")

    out_fps = output_fps(fps, a.out_fps)
    out_w = W * 2 if a.side_by_side else W
    writer = open_video_writer(a.output, out_fps, (out_w, H))

    bg = None
    if a.view_mode in ("silhouette", "outline"):
        bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

    kernel = np.ones((5, 5), np.uint8)
    trail = np.zeros((H, W), np.float32) if a.view_mode == "trail" else None
    prev_gray = None
    n = 0
    black = np.zeros((H, W, 3), dtype=np.uint8)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if a.view_mode == "diff":
            if prev_gray is not None:
                vis = motion_diff_bgr(prev_gray, gray, a.motion_thresh, black)
            else:
                vis = black
            prev_gray = gray

        elif a.view_mode in ("silhouette", "outline"):
            mask = bg.apply(frame, learningRate=0.002)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            if a.view_mode == "outline":
                edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
                vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            else:
                vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        else:  # trail
            if prev_gray is not None:
                d = cv2.absdiff(prev_gray, gray)
                _, mask = cv2.threshold(d, a.motion_thresh, 255, cv2.THRESH_BINARY)
                trail *= a.trail_decay
                trail = np.maximum(trail, mask.astype(np.float32))
                vis = cv2.cvtColor(trail.clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                vis = black
            prev_gray = gray

        out_frame = np.hstack([frame, vis]) if a.side_by_side else vis
        writer.write(out_frame)
        n += 1

    cap.release()
    writer.release()
    print(f"Done (motion view, no tracking). {n} frames @ {out_fps:.1f} fps. Output: {a.output}")


def detect_candidates(prev_gray, gray, a):
    """Return list of (cx, cy, area) ball-like motion blobs in this frame."""
    d = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(d, a.thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < a.min_area or area > a.max_area:
            continue
        per = cv2.arcLength(c, True)
        if per == 0:
            continue
        circ = 4 * np.pi * area / (per * per)
        if area >= 12 and circ < a.min_circularity:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        cands.append((cx, cy, area))
    return cands, mask


def trajectory_fit_metrics(track):
    """
    Fit x linearly and y quadratically vs frame index.
    Returns (rmse, quad_a, length) or None if too few points.
    """
    if len(track) < 5:
        return None

    frames = sorted(track.keys())
    t = np.array(frames, dtype=np.float64)
    x = np.array([track[f][0] for f in frames], dtype=np.float64)
    y = np.array([track[f][1] for f in frames], dtype=np.float64)
    t0 = t - t[0]

    coef_x = np.polyfit(t0, x, 1)
    if len(t0) >= 3:
        coef_y = np.polyfit(t0, y, 2)
    else:
        coef_y = np.array([0.0, np.polyfit(t0, y, 1)[0], np.polyfit(t0, y, 1)[1]])

    pred_x = np.polyval(coef_x, t0)
    pred_y = np.polyval(coef_y, t0)
    rmse = float(np.sqrt(np.mean((x - pred_x) ** 2 + (y - pred_y) ** 2)))
    return rmse, float(coef_y[0]), len(track)


def trajectory_score(track, a):
    """Higher is better. Combines length, low fit error, and angle-specific physics."""
    metrics = trajectory_fit_metrics(track)
    if metrics is None:
        return -1e9

    rmse, quad_a, length = metrics
    if rmse > 35:
        return -1e9

    score = length * 12.0 - rmse * 3.0

    # DTL: in image coords y grows downward; ballistic flight is concave-up (quad_a > 0).
    if a.angle == "dtl" and length >= 6:
        if quad_a < 0.02:
            score -= 40
        else:
            score += min(quad_a * 200, 20)

    return score


def interpolate_track(track, max_rmse=25.0):
    """
    Fill gaps between detections using the parabolic fit.
    Detected positions override fitted ones where both exist.
    """
    metrics = trajectory_fit_metrics(track)
    if metrics is None:
        return track

    rmse, _, _ = metrics
    if rmse > max_rmse:
        return track

    frames = sorted(track.keys())
    t0_base = frames[0]
    t = np.array(frames, dtype=np.float64)
    x = np.array([track[f][0] for f in frames], dtype=np.float64)
    y = np.array([track[f][1] for f in frames], dtype=np.float64)
    t0 = t - t0_base

    coef_x = np.polyfit(t0, x, 1)
    coef_y = np.polyfit(t0, y, 2) if len(t0) >= 3 else np.array([0.0, *np.polyfit(t0, y, 1)])

    dense = {}
    for f in range(frames[0], frames[-1] + 1):
        dt = f - t0_base
        dense[f] = (float(np.polyval(coef_x, dt)), float(np.polyval(coef_y, dt)))
    dense.update(track)
    return dense


def _link_from_start(start_f, sx, sy, per_frame_cands, frames, a, max_step, max_missed):
    """Greedy chain from one launch candidate."""
    track = {start_f: (sx, sy)}
    last = (sx, sy)
    last_f = start_f
    vx = vy = None
    missed = 0

    for f in frames:
        if f <= start_f:
            continue

        gap = f - last_f
        if vx is not None:
            px = last[0] + vx * gap
            py = last[1] + vy * gap
        else:
            px, py = last

        pick = None
        pick_d = 1e9
        for (cx, cy, _ca) in per_frame_cands.get(f, []):
            d = np.hypot(cx - px, cy - py)
            step = np.hypot(cx - last[0], cy - last[1])
            if step > max_step * gap:
                continue
            if d < pick_d:
                pick_d, pick = d, (cx, cy)

        if pick is None:
            missed += 1
            if missed > max_missed:
                break
            continue

        if a.angle == "faceon" and vx is not None and abs(vx) > 1:
            if np.sign(pick[0] - last[0]) != np.sign(vx) and abs(pick[0] - last[0]) > 5:
                missed += 1
                if missed > max_missed:
                    break
                continue

        if a.angle == "dtl" and vy is not None:
            if (pick[1] - last[1]) > 12 * gap:
                missed += 1
                if missed > max_missed:
                    break
                continue

        missed = 0
        if vx is not None:
            vx = 0.6 * vx + 0.4 * (pick[0] - last[0]) / gap
            vy = 0.6 * vy + 0.4 * (pick[1] - last[1]) / gap
        else:
            vx = (pick[0] - last[0]) / gap
            vy = (pick[1] - last[1]) / gap

        track[f] = pick
        last = pick
        last_f = f

    return track


def build_trajectory(per_frame_cands, a):
    """
    Link candidates into chains, score with parabolic fit, return best track.
    """
    max_step = 250 if a.fps_hint == "normal" else 90
    max_missed = 3
    frames = sorted(per_frame_cands.keys())

    best_track = {}
    best_score = -1e9

    for start_f in frames:
        for (sx, sy, _sa) in per_frame_cands[start_f]:
            track = _link_from_start(start_f, sx, sy, per_frame_cands, frames, a, max_step, max_missed)
            if len(track) < 5:
                continue
            score = trajectory_score(track, a)
            if score > best_score:
                best_score = score
                best_track = track

    if len(best_track) < 5:
        return {}

    return interpolate_track(best_track)


def collect_candidates(a):
    """Pass 1: scan video once, return candidates and optional debug masks."""
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {a.input}")

    per_frame_cands = {}
    debug_masks = {}
    prev_gray = None
    idx = 0
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if prev_gray is not None:
            cands, mask = detect_candidates(prev_gray, gray, a)
            per_frame_cands[idx] = cands
            if a.debug:
                debug_masks[idx] = mask
        prev_gray = gray
        idx += 1

    cap.release()
    if frame_count == 0:
        raise SystemExit("No frames decoded.")
    return per_frame_cands, debug_masks, frame_count


def render_tracked_video(a, fps, W, H, track, debug_masks):
    """Pass 2: stream input video, overlay trace on motion-diff (or original)."""
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {a.input}")

    out_fps = output_fps(fps, a.out_fps)
    out_w = W * 2 if a.side_by_side else W
    writer = open_video_writer(a.output, out_fps, (out_w, H))
    ordered = sorted(track.keys())
    json_rows = []
    idx = 0
    prev_gray = None
    black = np.zeros((H, W, 3), dtype=np.uint8)
    use_motion_bg = not a.original_bg

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if use_motion_bg:
            panel = motion_diff_bgr(prev_gray, gray, a.motion_thresh, black)
        else:
            panel = frame.copy()
        prev_gray = gray

        if a.debug:
            draw_debug_boxes(panel, debug_masks.get(idx), a)

        draw_track_overlay(panel, track, ordered, idx, a)

        if a.side_by_side:
            out = np.hstack([frame, panel])
        else:
            out = panel

        if idx in track and a.dump_json:
            x, y = track[idx]
            pad = a.crop_pad
            json_rows.append({
                "frame": idx,
                "ball_x": round(x, 1),
                "ball_y": round(y, 1),
                "crop": [max(0, int(x - pad)), max(0, int(y - pad)),
                         min(W, int(x + pad)), min(H, int(y + pad))],
            })

        writer.write(out)
        idx += 1

    cap.release()
    writer.release()
    return json_rows


def main():
    a = parse_args()
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {a.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if a.angle == "faceon" or a.motion_view:
        render_motion_view(a, fps, W, H)
        return

    per_frame_cands, debug_masks, _frame_count = collect_candidates(a)
    track = build_trajectory(per_frame_cands, a)
    if not track:
        print("WARNING: no consistent ball trajectory found. "
              "Try lowering --thresh, raising --max-area, or --fps-hint slowmo. "
              "Run with --debug to see what's being detected.")

    json_rows = render_tracked_video(a, fps, W, H, track, debug_masks)
    out_fps = output_fps(fps, a.out_fps)

    if a.dump_json:
        with open(a.dump_json, "w", encoding="utf-8") as f:
            json.dump({"fps": out_fps, "w": W, "h": H, "track": json_rows}, f, indent=2)
        print(f"Wrote {a.dump_json} ({len(json_rows)} tracked frames)")

    print(f"Done. {len(track)} frames tracked @ {out_fps:.1f} fps. Output: {a.output}")
    if track:
        print("If the trace wanders onto the club/body, tighten --max-area or raise --min-circularity.")


if __name__ == "__main__":
    main()
