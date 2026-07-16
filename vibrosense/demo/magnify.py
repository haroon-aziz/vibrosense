import argparse
import sys
import cv2
import numpy as np

sys.path.insert(0, "..")
from vibrosense.video_io import read_video_frames
from vibrosense.motion_mag import magnify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="magnified.mp4")
    parser.add_argument("--low", type=float, default=0.5)
    parser.add_argument("--high", type=float, default=6.0)
    parser.add_argument("--amplification", type=float, default=15.0)
    args = parser.parse_args()

    frames, fps = read_video_frames(args.input)
    magnified = magnify(frames, fps, low=args.low, high=args.high, amplification=args.amplification)

    h, w = magnified.shape[1:3]
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in magnified:
        writer.write(frame)
    writer.release()

    print(f"Wrote magnified video to {args.output}")


if __name__ == "__main__":
    main()
