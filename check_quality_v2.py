import sys, glob, os
from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from lib.configs import LABELS, GESTURES_6CL

MIN_RATIO = 1.5        # avg per-take gesture/rest ratio below this = dead (no signal)
MIN_GESTURE_ABS = 50   # median per-take gesture std-sum below this = weak coupling (calibrate below)

def labcol(df):
    for c in df.columns:
        vals = {str(v).strip().lower() for v in df[c].dropna().unique()[:50]}
        if vals & set(GESTURES_6CL):
            return c
    return None

def take_stats(p):
    df = pd.read_csv(p); lc = labcol(df)
    if lc is None: return None
    ch = df[LABELS].values.astype(float)
    lab = df[lc].astype(str).str.strip().str.lower().values
    def absamp(g):
        m = lab == g
        return float(ch[m].std(0).sum()) if m.any() else None
    rest = absamp("rest")
    if not rest: return None
    g_abs = {g: absamp(g) for g in GESTURES_6CL if g != "rest"}
    g_ratio = {g: (a/rest if a else None) for g, a in g_abs.items()}
    return rest, g_abs, g_ratio

def assess(folder):
    paths = sorted(glob.glob(str(ROOT/folder/"*.csv")))
    rest_l, gabs_l = [], []
    per_g_ratio = {g: [] for g in GESTURES_6CL if g != "rest"}
    for p in paths:
        s = take_stats(p)
        if s is None: continue
        rest, g_abs, g_ratio = s
        rest_l.append(rest)
        vals = [v for v in g_abs.values() if v is not None]
        gabs_l.append(np.mean(vals))
        for g, r in g_ratio.items():
            if r is not None: per_g_ratio[g].append(r)
    if not rest_l:
        print(f"\n=== {folder} ===  (no data)"); return
    med_rest = float(np.median(rest_l))
    med_gabs = float(np.median(gabs_l))
    avg_ratio = {g: (float(np.mean(v)) if v else None) for g, v in per_g_ratio.items()}
    dead = [g for g, r in avg_ratio.items() if r is not None and r < MIN_RATIO]
    weak = med_gabs < MIN_GESTURE_ABS
    if dead:
        verdict = "DEAD (no signal) -> " + ",".join(dead)
    elif weak:
        verdict = f"WEAK (low amplitude: median gesture abs {med_gabs:.0f} < {MIN_GESTURE_ABS})"
    else:
        verdict = "OK"
    print(f"\n=== {folder} ===")
    print(f"  median rest std-sum   : {med_rest:6.1f}")
    print(f"  median gesture std-sum: {med_gabs:6.1f}")
    print("  avg per-take ratio    : " + "  ".join(
        f"{g[:4]}={avg_ratio[g]:.2f}" for g in avg_ratio if avg_ratio[g] is not None))
    print(f"  VERDICT: {verdict}")

for folder in ("data/S006_BATCH1", "data/S006_BATCH2",
               "data/S003_BATCH1_PRONATION", "data/S003_BATCH2_PRONATION"):
    assess(folder)
