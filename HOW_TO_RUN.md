# How to run — swing motion + ball tracking

**`balltrack.py`** — ball flight tracking for down-the-line clips, plus face-on motion glow (white-on-black frame diff).

---

## 1. One-time setup

You need Python 3 and two libraries.

**On a computer (Mac/Windows/Linux):**

```
pip install -r requirements.txt
```

**On Android (Termux app):**

```
pkg install python
pip install opencv-python numpy
```

**On your DigitalOcean VPS** (recommended if phone install is fussy):

```
pip install -r requirements.txt
```
then upload a clip with `scp` or run it over SSH.

**No install at all — Google Colab** (works in a phone browser): OpenCV is preinstalled. Upload the script + clip, run it in a cell.

To confirm it worked:
```
python -c "import cv2, numpy; print(cv2.__version__)"
```
If that prints a version number, you're set.

---

## 2. Put the files together

Drop `balltrack.py` and your swing video in the same folder, then `cd` into it:

```
cd /path/to/that/folder
```

Your clip can be `.mp4`, `.mov`, etc. — whatever your phone records.

---

## 3. Run it

### Down-the-line clip (behind you, ball flies away) — full ball tracking

```
python balltrack.py myswing.mp4 -o traced.mp4 --dump-json track.json
```

- Default output is **motion-diff (white on black) + ball trace** — same look as face-on, with cyan trace line and green dot.
- Add `--side-by-side` for original | motion+trace split.
- Add `--debug` to show red boxes on every ball candidate (for tuning).
- Add `--original-bg` only if you want the old full-color overlay instead.

Example with debug + side-by-side:
```
python balltrack.py myswing.mp4 -o traced.mp4 --angle dtl --side-by-side --debug --dump-json track.json
```

### Face-on clip (side view) — motion glow, no ball tracking

```
python balltrack.py myswing.mp4 -o motion.mp4 --angle faceon --side-by-side
```

- `motion.mp4` — original on the left, raw motion diff on the right (white glow on black).
- Output keeps the **source frame rate** (~120/240fps) — every slo-mo frame, full slow motion. Skip `--out-fps 30` unless you want a sped-up preview.

Tune sensitivity with `--motion-thresh` (default 16; lower = more glow). Example:
```
python balltrack.py myswing.mp4 -o motion.mp4 --angle faceon --side-by-side --motion-thresh 12
```

---

## 4. If the ball isn't being tracked (DTL)

This is expected on the first try — thresholds depend on your camera and lighting. Work through these in order:

**Step 1 — see what it's detecting:**
```
python balltrack.py myswing.mp4 -o debug.mp4 --debug
```
This draws red boxes on everything it considers a candidate. Watch `debug.mp4`:

- **Ball has no red box on it** → detector isn't catching it. Lower the threshold:
  `--thresh 14` (try 14, then 10).
- **Ball is caught but so is everything else (club, body, shake)** → too sensitive. Raise it:
  `--thresh 30`.

**Step 2 — ball caught but trace won't follow it:**

- Ball blob looks big/elongated (motion blur) → raise the size ceiling: `--max-area 800`.
- Trace jumps onto your club or body → lower the ceiling: `--max-area 150`, and/or raise roundness: `--min-circularity 0.4`.

**Step 3 — combine what worked**, e.g.:
```
python balltrack.py myswing.mp4 -o traced.mp4 --thresh 12 --max-area 600 --dump-json track.json
```

---

## 5. Flags reference

| Flag | What it does | When to change |
|---|---|---|
| `-o NAME.mp4` | output filename | always set it |
| `--angle dtl` / `faceon` | tracking vs. motion glow | `faceon` for side-on clips |
| `--fps-hint slowmo` / `normal` | speed gate | `normal` only if NOT slo-mo |
| `--motion-thresh` | face-on glow sensitivity (default 16) | faint body/club → lower; noisy → raise |
| `--thresh N` | DTL ball detection sensitivity | ball missed → lower; too noisy → raise |
| `--max-area N` | biggest allowed blob | trace grabs body/club → lower |
| `--min-area N` | smallest allowed blob | ball missed at distance → lower |
| `--min-circularity 0..1` | how round a blob must be | false hits → raise |
| `--dump-json FILE` | save ball coords + crops | when feeding a vision model |
| `--debug` | show all candidates (red) | tuning — use on DTL before final run |
| `--original-bg` | DTL trace on full-color video | only if you don't want motion-diff look |
| `--motion-view` | force glow, skip tracking | glow on a DTL clip |
| `--view-mode diff` | raw frame-difference glow (default) | face-on output |
| `--side-by-side` | original \| motion split | face-on and DTL |
| `--out-fps N` | override output fps | only for a faster preview; default matches source slo-mo |
| `--trace-fade` | fading trace tail | cosmetic |

---

## 6. What to realistically expect

- **Face-on**: works out of the box — raw motion diff on black at full source fps (every slo-mo frame).
- **Down-the-line**: best results come from slo-mo (120/240fps). Default render is motion-diff + trace (same look as face-on). Use `--debug --side-by-side` while tuning. Expect ~25–30 min processing for a 900-frame clip. Trace usually dies once the ball is too small/far — normal, not a bug.
- Thresholds are **never** universal — one good tuning pass per camera/lighting setup and you can reuse those numbers.
