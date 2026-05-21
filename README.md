# golfvid

Motion isolation and golf ball flight tracking for phone slo-mo swing clips.

Two modes, one main script:

| Mode | Command | Output |
|------|---------|--------|
| **Face-on** (side view) | `--angle faceon` | Motion-diff glow (white on black). No ball tracking. |
| **Down-the-line** (behind ball) | `--angle dtl` (default) | Motion-diff + ball trace + optional JSON crop boxes. |

## Quick start

```bash
pip install -r requirements.txt

# Face-on motion glow
python balltrack.py IMG_3970.mov -o motion.mp4 --angle faceon --side-by-side

# DTL ball tracking (tune with --debug first)
python balltrack.py IMG_4004.mov -o traced.mp4 --angle dtl --side-by-side --debug --dump-json track.json
```

## Files

| File | Purpose |
|------|---------|
| `balltrack.py` | Main pipeline — face-on motion + DTL ball tracking |
| `pixelsub.py` | Standalone motion-isolation utility (optional) |
| `HOW_TO_RUN.md` | Setup, commands, tuning workflow |
| `STATUS.md` | Current state, test results, roadmap |
| `test_balltrack.py` | Unit tests for trajectory logic |
| `requirements.txt` | Python dependencies |

## Test clips (local)

| Clip | Angle | ~FPS | Frames | Resolution |
|------|-------|------|--------|------------|
| `IMG_3970.mov` | Face-on | 214 | 837 | 1080×1920 |
| `IMG_4004.mov` | DTL | 212 | 900 | 1080×1920 |

## Docs

- [How to run](HOW_TO_RUN.md) — install, commands, flag reference, tuning
- [Project status](STATUS.md) — what's working, known limits, next steps
