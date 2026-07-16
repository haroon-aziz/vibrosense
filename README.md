# VibroSense

Contactless material classification from a single video — not by appearance, but by how a surface **vibrates**. Tap, brush, or lightly excite an object on camera; VibroSense amplifies sub-pixel motion (Eulerian video magnification), extracts a per-region frequency signature, and clusters regions by vibration behavior. Wood, metal, plastic, and rubber all ring differently — this reads that difference straight out of RGB video.

## Pipeline

1. **Motion magnification** (`motion_mag.py`) — Gaussian pyramid downsampling + temporal bandpass filtering + amplification, reconstructed back into visible motion (linear Eulerian Video Magnification).
2. **Patch-wise vibration extraction** (`video_io.py`) — the frame is divided into a grid; each patch's mean intensity over time is a 1D vibration signal.
3. **Frequency-signature features** (`vibration_features.py`) — dominant frequency, spectral centroid, spectral flatness, bandwidth, and energy per patch.
4. **Material clustering** (`material_cluster.py`) — unsupervised K-Means groups patches with similar vibration signatures and renders a color-coded overlay.

## Structure

```
vibrosense/
  video_io.py           frame loading, grid patches, patch time-series
  motion_mag.py         Eulerian motion magnification
  vibration_features.py per-patch frequency-domain features
  material_cluster.py   K-Means clustering + overlay rendering
demo/
  magnify.py            CLI: produce a motion-magnified video
  analyze.py            CLI: cluster regions by vibration signature
```

## Usage

```bash
pip install -r requirements.txt

python demo/magnify.py --input clip.mp4 --output magnified.mp4 --low 0.5 --high 6.0 --amplification 15
python demo/analyze.py --input clip.mp4 --output analyzed.png --clusters 4
```

## Notes

- Clustering is unsupervised — it groups patches by vibration similarity, it does not itself know material names. Attaching labels (wood/metal/plastic) requires a small labeled calibration set per clustered ID.
- Works best with a static camera and a brief excitation (tap, light knock, ambient hum) during capture; camera motion will dominate the signal otherwise.
- Frequency band (`--low`/`--high`) should be tuned to the excitation type — light taps produce broadband high-frequency content, low-frequency hums are closer to 1-10 Hz.

## License

MIT
