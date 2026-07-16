import argparse
import sys
import cv2
import numpy as np

sys.path.insert(0, "..")
from vibrosense.video_io import read_video_frames, make_grid_patches, extract_patch_series
from vibrosense.vibration_features import extract_feature_matrix
from vibrosense.material_cluster import MaterialClusterer, draw_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="analyzed.png")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--clusters", type=int, default=4)
    args = parser.parse_args()

    frames, fps = read_video_frames(args.input)
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float64)

    coords = make_grid_patches(gray.shape[1:], patch_size=args.patch_size)
    series = extract_patch_series(gray, coords, patch_size=args.patch_size)
    features = extract_feature_matrix(series, fps)

    clusterer = MaterialClusterer(n_clusters=args.clusters)
    labels = clusterer.fit_predict(features)

    overlay = draw_overlay(frames[0], coords, labels, patch_size=args.patch_size)
    cv2.imwrite(args.output, overlay)

    print(f"Wrote clustered overlay to {args.output}")
    print(f"Patches analyzed: {len(coords)}, clusters: {args.clusters}")


if __name__ == "__main__":
    main()
