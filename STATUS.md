# Project status — golfvid

*Last updated: May 2026*

## Summary

`balltrack.py` is a working prototype for two workflows:

1. **Face-on motion visualization** — reliable, fast (~8–10 min for 800+ frames). Raw frame-diff glow (white on black), tunable via `--motion-thresh`.
2. **Down-the-line ball tracking** — functional for early flight (~0.7 s), slow (~25–30 min for 900 frames), needs per-camera tuning. Outputs traced video + JSON crop boxes for downstream vision models.

---

## Codebase layout

```
golfvid/
├── balltrack.py          # Main script (514 lines)
├── pixelsub.py           # Optional standalone motion isolation
├── test_balltrack.py     # 5 unit tests (trajectory + detector)
├── requirements.txt      # opencv-python, numpy
├── HOW_TO_RUN.md         # User guide
├── README.md             # Project overview
└── STATUS.md             # This file
```

---

## What's working

### Face-on (`--angle faceon`)

- [x] Motion-diff output (default `--view-mode diff`)
- [x] Side-by-side original | motion (`--side-by-side`)
- [x] Full source fps preserved by default (no accidental speed-up)
- [x] Separate sensitivity: `--motion-thresh` (default 16)
- [x] Tested on `IMG_3970.mov` (~214 fps, 837 frames) — multiple successful outputs

### DTL (`--angle dtl`)

- [x] Two-pass pipeline: detect → link → render (streaming, no full-RAM frame buffer)
- [x] Gap-aware velocity prediction + parabolic trajectory scoring
- [x] Motion-diff render background (matches face-on look) + cyan trace + green dot
- [x] `--side-by-side`, `--debug` (red candidate boxes), `--original-bg` (legacy color overlay)
- [x] JSON export: per-frame `ball_x`, `ball_y`, `crop` boxes
- [x] Unit tests for linking, interpolation, scoring

### Infrastructure

- [x] `requirements.txt`
- [x] VideoWriter open check
- [x] `python -m unittest test_balltrack.py` passes

---

## Test results (IMG_4004.mov, DTL)

| Run | Debug | Tracked frames | Frame range | Notes |
|-----|-------|----------------|-------------|-------|
| `test_dtl` | Yes | 162 | 429–594 | Red boxes visible; JSON fps stale (30) |
| `dtl_4004` | No | 155 | 435–591 | No red boxes; JSON fps correct (212) |

**Quality assessment (both runs):**

- Good for ~0.5–0.7 s after launch (frames ~430–550)
- Trace loses lock after ~frame 550 (jumps 35–75 px between frames)
- Parabolic fit RMSE ~35 px (borderline)
- Only ~17% of 900-frame timeline has ball positions
- Tracking starts late (~2 s into clip) — acceptable if ball not visible earlier

**Processing time:** ~29 minutes for 900 frames @ 1080×1920 on local Windows machine.

---

## Known limitations

| Issue | Impact | Workaround |
|-------|--------|------------|
| DTL trajectory linking is O(n²) greedy search | 25–30 min runtime on 900-frame clips | Limit search window; optimize (see next steps) |
| Ball detection uses single global `--thresh` | Misses faint ball or catches club/body noise | Tune with `--debug`; adjust `--thresh`, `--max-area` |
| Track dies mid-flight | JSON crops unreliable after ~frame 550 | Tighten `--max-area`; lower `--thresh` for distant ball |
| `--motion-thresh` (visual) ≠ `--thresh` (detector) | Motion panel can look different from red debug boxes | Documented; intentional split |
| `mp4v` codec | Some players struggle with high-fps mp4 | Re-encode with ffmpeg if needed |
| No git / CI | Manual testing only | Add when repo is initialized |

---

## Recent changes (this session)

1. DTL trajectory: gap-aware prediction, parabolic scoring, interpolation
2. Streaming render pass (memory-safe for long slo-mo)
3. Face-on: restored simple diff look; `--motion-thresh` default 16
4. Output fps defaults to source (full slo-mo preserved)
5. DTL render: motion-diff background by default (face-on aesthetic)
6. `--debug` / `--side-by-side` / `--original-bg` for DTL

---

## Next steps (priority order)

### 1. Re-run DTL with new render + debug (immediate)

The updated motion-diff DTL output has not been validated on disk yet:

```bash
python balltrack.py IMG_4004.mov -o dtl_v2.mp4 --angle dtl --side-by-side --debug --dump-json dtl_v2_track.json
```

Compare visually to `test_dtl.mp4`. Expect ~25–30 min.

### 2. Tune DTL for IMG_4004

Based on debug output:

- Ball not boxed → `--thresh 14` or `12`
- Trace jumps to body/club → `--max-area 150`, `--min-circularity 0.35`
- Re-test and save working flags per camera in a note or config file

### 3. Speed up trajectory linking (high impact)

`build_trajectory()` tries every candidate on every early frame. For 900 frames this dominates runtime.

Options:

- Limit launch search to frames with high motion density
- Cap max start frames (e.g. first 200 after motion onset)
- Early-exit when score plateaus
- Target: under 10 min for 900-frame clip

### 4. Improve track continuity

- Raise `max_missed` gap tolerance with better gap-aware velocity
- Reject chains with step jumps > N px
- Use fitted arc to extend trace a few frames past last detection

### 5. Project hygiene

- [ ] Initialize git repo
- [ ] Add `.gitignore` (outputs, `__pycache__`, large `.mov`)
- [ ] Pin opencv/numpy versions after stable run
- [ ] Optional: `tune.json` per camera profile

### 6. Future (out of scope for now)

- Kalman filter + Hungarian assignment for robust tracking
- Auto-detect impact frame to narrow search window
- H.264 output via ffmpeg pipe
- Vision model handoff script (crop patches from JSON)

---

## Commands cheat sheet

```bash
# Tests
python -m unittest test_balltrack.py -v

# Face-on (full slo-mo)
python balltrack.py IMG_3970.mov -o motion.mp4 --angle faceon --side-by-side

# Face-on (more sensitive)
python balltrack.py IMG_3970.mov -o motion.mp4 --angle faceon --side-by-side --motion-thresh 12

# DTL tune (always start here)
python balltrack.py IMG_4004.mov -o debug.mp4 --angle dtl --debug --side-by-side

# DTL final
python balltrack.py IMG_4004.mov -o traced.mp4 --angle dtl --side-by-side --dump-json track.json
```

---

## Output artifacts (local, not in repo)

Generated during testing — safe to delete or `.gitignore`:

- `faceon_*.mp4`, `dtl_*.mp4`, `traced*.mp4`, `test_*.mp4`, `debug.mp4`
- `*_track.json`, `track.json`
