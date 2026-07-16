import numpy as np


def _spectrum(series, fs):
    n = len(series)
    windowed = (series - series.mean()) * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spectrum


def extract_features(series, fs):
    freqs, spectrum = _spectrum(series, fs)
    total_power = spectrum.sum() + 1e-8

    dominant_idx = np.argmax(spectrum)
    dominant_freq = freqs[dominant_idx]

    centroid = np.sum(freqs * spectrum) / total_power

    geo_mean = np.exp(np.mean(np.log(spectrum + 1e-8)))
    arith_mean = spectrum.mean() + 1e-8
    flatness = geo_mean / arith_mean

    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total_power)

    energy = np.sum(series ** 2) / len(series)

    return np.array([dominant_freq, centroid, flatness, bandwidth, energy])


def extract_feature_matrix(patch_series, fs):
    feats = [extract_features(s, fs) for s in patch_series]
    return np.array(feats)
