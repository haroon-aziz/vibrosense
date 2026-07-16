import numpy as np
import cv2


def read_video_frames(path, max_frames=None, resize=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return np.array(frames), fps


def make_grid_patches(frame_shape, patch_size=16):
    h, w = frame_shape[:2]
    coords = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            coords.append((y, x))
    return coords


def extract_patch_series(frames_gray, coords, patch_size=16):
    n = frames_gray.shape[0]
    series = np.zeros((len(coords), n), dtype=np.float64)
    for i, (y, x) in enumerate(coords):
        patch = frames_gray[:, y:y + patch_size, x:x + patch_size]
        series[i] = patch.mean(axis=(1, 2))
    return series
