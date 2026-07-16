import numpy as np
import cv2
from scipy.signal import butter, filtfilt


def build_gaussian_pyramid(frame, levels):
    pyramid = [frame.astype(np.float32)]
    for _ in range(levels - 1):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame.astype(np.float32))
    return pyramid[-1]


def gaussian_downsample_stack(frames, levels=3):
    stack = []
    for f in frames:
        stack.append(build_gaussian_pyramid(f, levels))
    return np.array(stack)


def temporal_bandpass(stack, fps, low, high, order=3):
    nyq = 0.5 * fps
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    filtered = filtfilt(b, a, stack, axis=0)
    return filtered


def magnify(frames, fps, low=0.5, high=6.0, amplification=15.0, levels=3):
    small_stack = gaussian_downsample_stack(frames, levels=levels)
    filtered = temporal_bandpass(small_stack, fps, low, high)
    amplified = filtered * amplification

    out_frames = []
    h, w = frames.shape[1:3]
    for i in range(frames.shape[0]):
        upsampled = cv2.resize(amplified[i], (w, h), interpolation=cv2.INTER_LINEAR)
        combined = frames[i].astype(np.float32) + upsampled
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        out_frames.append(combined)
    return np.array(out_frames)
