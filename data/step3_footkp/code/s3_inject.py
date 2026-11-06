#!/usr/bin/env python3
"""Đo độ nhạy Viterbi 3.1 bằng swap tiêm vào chuỗi thật (không có ground truth nhãn L/R).

Lấy chuỗi đã sửa làm nền, hoán đổi L/R trên các đoạn biết trước rồi chạy lại. Hai kiểu tiêm:
  mid      đoạn bắt đầu/kết thúc ở vị trí bất kỳ: nhãn nhảy khi 2 chân đang xa nhau
  crossing đoạn đúng 1 nửa chu kỳ giữa 2 lần chéo chân: kiểu khó nhất, không có bước nhảy vị trí

  python step3_footkp/code/s3_inject.py --switch 0.3 --phase 1.0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import s3_common as C
import s3_swap as W


def run_len_runs(flag, cont):
    ev = np.flatnonzero((np.diff(flag.astype(int)) != 0) & (np.diff(cont) == 0))
    return len(ev)


def eval_runs(fl, truth, runs):
    """Tỉ lệ đoạn tiêm được sửa đúng: >= 80% frame của đoạn khớp sự thật (không tính lật toàn cục)."""
    agree = fl == truth
    if agree.mean() < 0.5:
        agree = ~agree
    return [agree[a:b].mean() >= 0.8 for a, b in runs], agree.mean()


def trial(c, run_sign, base_flag, rng, mode, kw, n_runs=3):
    kp0 = W.apply_swap(c.kp, base_flag)
    d = kp0[:, 15, 0] - kp0[:, 16, 0]
    truth = np.zeros(c.n, bool)
    runs = []
    if mode == "crossing":
        z = np.flatnonzero(np.diff(np.sign(np.nan_to_num(d))) != 0)
        z = z[np.r_[True, np.diff(z) > 2]] if len(z) else z
        cand = [(z[i] + 1, z[i + 1] + 1) for i in range(len(z) - 1)
                if 2 <= z[i + 1] - z[i] and c.cont[z[i]] == c.cont[z[i + 1]]]
        if not cand:
            return None
        for j in rng.permutation(len(cand))[:n_runs]:
            a, b = cand[j]
            if truth[max(0, a - 3):b + 3].any():
                continue
            truth[a:b] = True
            runs.append((a, b))
    else:
        for _ in range(n_runs):
            ln = int(rng.choice([2, 4, 8, 16]))
            if c.n <= ln + 6:
                continue
            a = int(rng.integers(3, c.n - ln - 3))
            if c.cont[a] != c.cont[a + ln] or truth[max(0, a - 3):a + ln + 3].any():
                continue
            truth[a:a + ln] = True
            runs.append((a, a + ln))
    if not runs:
        return None
    c2 = C.Clip(**{**c.__dict__, "kp": W.apply_swap(kp0, truth), "score": _swap_score(c.score, base_flag, truth)})
    fl, _, _ = W.fix_swaps(c2, run_sign, **kw)
    ok, acc = eval_runs(fl, truth, runs)
    # theo độ dài đoạn
    return [(mode, b - a, o) for (a, b), o in zip(runs, ok)], acc


def _swap_score(s, f1, f2):
    out = s.copy()
    for f in (f1, f2):
        tmp = out.copy()
        tmp[np.ix_(f, C.L_LEG)], tmp[np.ix_(f, C.R_LEG)] = out[np.ix_(f, C.R_LEG)], out[np.ix_(f, C.L_LEG)]
        out = tmp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", type=float, default=C.SWITCH_COST)
    ap.add_argument("--phase", type=float, default=C.W_PHASE)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = C.load_meta()
    kw = dict(switch=a.switch, w_phase=a.phase)
    rng = np.random.default_rng(a.seed)
    rows, ev, sw = [], 0, []
    for vid in meta.video_id:
        c = C.load_clip(vid, meta)
        s = int(c.meta.s_final)
        base, _, _ = W.fix_swaps(c, s, **kw)
        ev += run_len_runs(base, c.cont)
        sw.append(base.mean())
        for mode in ("mid", "crossing"):
            for _ in range(a.trials):
                r = trial(c, s, base, rng, mode, kw)
                if r:
                    rows += [dict(vid=vid, mode=m, ln=l, ok=o) for m, l, o in r[0]]
    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df.ln, [0, 2, 4, 8, 16, 1000], labels=["<=2", "3-4", "5-8", "9-16", ">16"])
    t = df.groupby(["mode", "bin"], observed=True).ok.agg(["mean", "size"]).round(3)
    print(f"switch={a.switch} phase={a.phase}: sự kiện swap trên dữ liệu thật={ev}, video có swap={sum(x > 0 for x in sw)}")
    print(t.to_string())
    print("tổng:", df.groupby("mode").ok.mean().round(3).to_dict())
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
