import os
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.stats import kurtosis, skew

def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def load_vibration_file(path: str) -> np.ndarray:
    df = pd.read_csv(path, header=None, sep=r"[\s,]+", engine="python")
    x = df.values.astype(np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

def time_features(x: np.ndarray) -> np.ndarray:
    feats = []
    for c in range(x.shape[1]):
        s = x[:, c]
        rms = np.sqrt(np.mean(s**2))
        mean = np.mean(s)
        std = np.std(s) + 1e-12
        k = kurtosis(s, fisher=False)
        sk = skew(s)
        peak = np.max(np.abs(s))
        crest = peak / (rms + 1e-12)
        feats.extend([mean, std, rms, k, sk, peak, crest])
    return np.array(feats, dtype=np.float32)

def list_experiment_dirs(extract_dir: str, min_files: int = 50) -> List[str]:
    exp_dirs = []
    for root, _, files in os.walk(extract_dir):
        if len(files) >= min_files:
            exp_dirs.append(root)
    return sorted(set(exp_dirs))

def build_runs(extract_dir: str, max_runs: int = None) -> List[Dict]:
    runs = []
    exp_dirs = list_experiment_dirs(extract_dir)

    if max_runs:
        exp_dirs = exp_dirs[:max_runs]

    for exp_dir in exp_dirs:
        files = sorted([f for f in os.listdir(exp_dir) if os.path.isfile(os.path.join(exp_dir, f))],
                       key=natural_key)
        # filter obvious non-data
        files = [f for f in files if not f.lower().endswith((".pdf", ".doc", ".docx"))]
        paths = [os.path.join(exp_dir, f) for f in files]
        T = len(paths)
        if T < 50:
            continue

        X = []
        rul = []
        for i, p in enumerate(paths):
            vib = load_vibration_file(p)
            feat = time_features(vib)
            X.append(feat)
            rul.append((T - 1 - i))  # step-based RUL

        runs.append({
            "exp_dir": exp_dir,
            "X": np.stack(X, axis=0).astype(np.float32),
            "rul": np.array(rul, dtype=np.float32),
        })

    return runs
