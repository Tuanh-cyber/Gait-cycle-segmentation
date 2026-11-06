# -*- coding: utf-8 -*-
"""
Per-frame biomechanical signals from COCO-17 keypoints, plus gait-cycle normalisation.

This is the shared foundation: the descriptive comparison and the eventual feature
computation both read their numbers from here, so a change to a definition lands in
both at once.

Conventions
-----------
* image y grows downward, so "higher up" means a smaller y
* x is flipped where needed so that +x is always the running direction, letting
  clips shot in either direction be compared without sign cancellation
* every distance is divided by L_torso, making it invariant to camera distance
  and body size
* bilateral joints resolve to the camera-side limb: seen from the side the far
  half of the body is occluded and its keypoints are guesses, so averaging the
  two sides drags a good number down with a bad one
"""
import numpy as np
from scipy.signal import savgol_filter, find_peaks

NOSE = 0
EAR_L, EAR_R = 3, 4
SHO_L, SHO_R = 5, 6
ELB_L, ELB_R = 7, 8
WRI_L, WRI_R = 9, 10
HIP_L, HIP_R = 11, 12
KNE_L, KNE_R = 13, 14
ANK_L, ANK_R = 15, 16

CONF_MIN = 0.2


# ── preprocessing ─────────────────────────────────────────────────────────────
def _interp_nan(seq):
    seq = np.asarray(seq, np.float32).copy()
    seq[~np.isfinite(seq)] = np.nan
    T, J, C = seq.shape
    idx = np.arange(T, dtype=np.float32)
    for j in range(J):
        for c in range(C):
            x = seq[:, j, c]
            ok = np.isfinite(x)
            if ok.sum() == 0:
                seq[:, j, c] = 0.0
                continue
            if ok.sum() == 1:
                seq[:, j, c] = x[ok][0]
                continue
            first, last = int(np.argmax(ok)), T - 1 - int(np.argmax(ok[::-1]))
            x[:first] = x[first]
            x[last + 1:] = x[last]
            ok = np.isfinite(x)
            x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
            seq[:, j, c] = x
    return np.nan_to_num(seq)


def preprocess(xy, conf, smooth_win=9):
    """Mask low-confidence points, fill gaps, smooth, and report the running direction.

    Direction comes from the nose sitting anterior to the ear. Posture-based cues
    (shoulder ahead of hip) would be circular here, because leaning and hips-back
    are among the very faults being measured.
    """
    seq = xy.copy().astype(np.float32)
    seq[conf < CONF_MIN] = np.nan
    seq = _interp_nan(seq)
    if len(seq) >= smooth_win:
        seq = savgol_filter(seq, smooth_win, 3, axis=0).astype(np.float32)

    ear = np.where((conf[:, EAR_L].mean() >= conf[:, EAR_R].mean()), EAR_L, EAR_R)
    face = float(np.median(seq[:, NOSE, 0] - seq[:, ear, 0]))
    direction = 1.0 if face >= 0 else -1.0

    hip_mid = (seq[:, HIP_L] + seq[:, HIP_R]) / 2.0
    sho_mid = (seq[:, SHO_L] + seq[:, SHO_R]) / 2.0
    L = float(np.median(np.linalg.norm(sho_mid - hip_mid, axis=1)))
    if not np.isfinite(L) or L < 1e-6:
        L = 1.0

    cen = seq - hip_mid[:, None, :]
    cen[:, :, 0] *= direction                    # +x is now always forward
    return cen / L, L, direction, hip_mid


def near_side(conf, left, right):
    """Which of a keypoint pair the camera actually sees."""
    return left if conf[:, left].mean() >= conf[:, right].mean() else right


# ── geometry ──────────────────────────────────────────────────────────────────
def _ang_vert(v):
    """Angle of an upward-pointing segment from vertical; positive means tilted forward.

    For the trunk only. A downward-pointing limb measured this way sits near +-180
    and wraps between the two, which inflates any range-of-motion computed from it:
    the upper arm produced an arm_flexion_ROM of 352 degrees that way. Downward
    segments use _ang_down instead.
    """
    return np.degrees(np.arctan2(v[:, 0], -v[:, 1]))


def _ang_down(v):
    """Angle of a downward-pointing segment from vertical; positive means forward.

    Zero is straight down, so thigh, shank and upper arm all stay near zero and
    never wrap.
    """
    return np.degrees(np.arctan2(v[:, 0], v[:, 1]))


def _ang_horiz(v):
    """Angle from horizontal, in degrees; positive means the tip points upward."""
    return np.degrees(np.arctan2(-v[:, 1], v[:, 0]))


def _ang3(a, b, c):
    """Interior angle at b, in degrees. 180 means the segment is straight."""
    ba, bc = a - b, c - b
    nba = np.linalg.norm(ba, axis=1, keepdims=True) + 1e-8
    nbc = np.linalg.norm(bc, axis=1, keepdims=True) + 1e-8
    cos = np.sum(ba / nba * bc / nbc, axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def signals(xy, conf):
    """Every per-frame signal used by the descriptive comparison.

    Returns (dict of name -> (T,) array, meta dict).
    """
    s, L, direction, hip_raw = preprocess(xy, conf)

    ear = near_side(conf, EAR_L, EAR_R)
    sho = near_side(conf, SHO_L, SHO_R)
    elb = near_side(conf, ELB_L, ELB_R)
    wri = near_side(conf, WRI_L, WRI_R)
    hip = near_side(conf, HIP_L, HIP_R)
    kne = near_side(conf, KNE_L, KNE_R)
    ank = near_side(conf, ANK_L, ANK_R)

    sho_mid = (s[:, SHO_L] + s[:, SHO_R]) / 2.0
    hip_mid = (s[:, HIP_L] + s[:, HIP_R]) / 2.0     # ~0 after centring

    # the trunk points up from hip to shoulder; thigh, shank and upper arm point
    # down, so they are referenced to the downward vertical to avoid wrapping
    trunk = _ang_vert(sho_mid - hip_mid)
    thigh = _ang_down(s[:, kne] - s[:, hip])
    shank = _ang_down(s[:, ank] - s[:, kne])
    arm   = _ang_down(s[:, elb] - s[:, sho])

    out = {
        # trunk and pelvis
        'trunk_lean':      trunk,
        'hip_shoulder_dx': sho_mid[:, 0] - hip_mid[:, 0],
        'hip_drop':        s[:, HIP_L, 1] - s[:, HIP_R, 1],
        'sho_tilt':        s[:, SHO_L, 1] - s[:, SHO_R, 1],
        'hip_height':      -(hip_raw[:, 1] - np.median(hip_raw[:, 1])) / L,

        # lower limb
        'thigh_angle':     thigh,
        'hip_flexion':     thigh - trunk,            # thigh relative to the trunk
        'knee_angle':      _ang3(s[:, hip], s[:, kne], s[:, ank]),
        'shank_angle':     shank,
        'ankle_x':         s[:, ank, 0] - hip_mid[:, 0],
        'ankle_height':    -s[:, ank, 1],
        'knee_height':     -s[:, kne, 1],
        'ankle_hip_dist':  np.linalg.norm(s[:, ank] - s[:, hip], axis=1),

        # upper limb
        'elbow_angle':     _ang3(s[:, sho], s[:, elb], s[:, wri]),
        'arm_angle':       arm - trunk,              # upper arm relative to the trunk
        'wrist_x':         s[:, wri, 0] - sho_mid[:, 0],
        'upper_arm_len':   np.linalg.norm(s[:, elb] - s[:, sho], axis=1),

        # head and shoulder girdle
        'head_pitch':      _ang_horiz(s[:, NOSE] - s[:, ear]),
        'sho_ear_gap':     s[:, sho, 1] - s[:, ear, 1],
    }
    meta = dict(L_torso=L, direction=direction,
                sides=dict(ear=ear, sho=sho, elb=elb, wri=wri,
                           hip=hip, kne=kne, ank=ank))
    return out, meta


# ── gait cycle ────────────────────────────────────────────────────────────────
def static_mask(sig, thresh=0.004):
    """Frames where the body is not moving - freeze frames and held shots.

    These clips are edited social-media instructional videos, not continuous
    running: they cut in slow-motion replays, frozen stills and pieces to camera.
    C11 holds a single still for nearly four seconds, during which every keypoint
    is literally constant. A frozen span has no stride in it and must not be fed
    to a cycle detector.
    """
    a = sig['ankle_x']
    k = sig['knee_height']
    mv = np.abs(np.diff(a, prepend=a[:1])) + np.abs(np.diff(k, prepend=k[:1]))
    # a short still inside normal motion still counts as still, so smooth lightly
    mv = np.convolve(mv, np.ones(3) / 3, mode='same')
    return mv < thresh


def detect_ic(sig, fps=None, min_len=8):
    """Initial-contact frames, taken as peak forward reach of the ankle.

    Deliberately free of any cadence assumption. An earlier version gated peaks by
    a distance derived from expected steps-per-minute, which fell apart here: these
    clips mix normal-speed and slow-motion footage, so one clip carries two very
    different stride durations and no single distance suits both. Because cycles are
    resampled onto a 0-100 % axis anyway, a slow-motion stride normalises to the
    same curve as a full-speed one - playback rate simply does not matter. What does
    matter is excluding frozen frames, which is handled separately.

    Returns peak indices; filtering of implausible strides happens in `cycles`.
    """
    x = np.asarray(sig['ankle_x'], float)
    if len(x) < min_len * 2:
        return np.array([], int)
    spread = float(np.percentile(x, 90) - np.percentile(x, 10))
    if spread < 1e-6:
        return np.array([], int)
    pk, _ = find_peaks(x, prominence=0.35 * spread, distance=4)
    return pk


def valid_strides(sig, ic, min_len=8, amp_frac=0.5):
    """Stride boundaries worth keeping, as (start, end) frame pairs.

    A stride survives when the foot actually swept through a full arc during it and
    no part of it sits in a frozen span. Amplitude is judged against the clip's own
    median stride, so the test works the same for a slow-motion and a full-speed
    stride and does not assume a cadence.
    """
    if len(ic) < 2:
        return []
    stat = static_mask(sig)
    x = sig['ankle_x']
    cand = [(a, b) for a, b in zip(ic[:-1], ic[1:]) if b - a >= min_len]
    if not cand:
        return []
    amps = np.array([x[a:b].max() - x[a:b].min() for a, b in cand])
    med = float(np.median(amps))
    out = []
    for (a, b), amp in zip(cand, amps):
        if amp < amp_frac * med:
            continue
        if stat[a:b].mean() > 0.25:          # mostly frozen: not a real stride
            continue
        out.append((a, b))
    return out


def cycles(sig_arr, strides, n=101):
    """Resample the given strides onto a common 0-100 % axis. Returns (n_strides, n)."""
    out = []
    for a, b in strides:
        seg = np.asarray(sig_arr[a:b], float)
        if len(seg) < 5 or not np.isfinite(seg).all():
            continue
        out.append(np.interp(np.linspace(0, 1, n),
                             np.linspace(0, 1, len(seg)), seg))
    return np.array(out) if out else np.empty((0, n))


def cadence(strides, fps):
    """Steps per minute, from the clip's median stride duration.

    Only meaningful for clips shot at a constant playback rate; a clip that mixes
    slow motion with normal speed has no single cadence, which the wide spread of
    stride durations will show.
    """
    if len(strides) < 1:
        return np.nan, np.nan
    d = np.array([b - a for a, b in strides], float)
    spm = 120.0 * fps / np.median(d)
    spread = float(np.percentile(d, 90) / max(np.percentile(d, 10), 1e-9))
    return spm, spread
