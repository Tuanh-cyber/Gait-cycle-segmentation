"""Four-state gait HSMM with a directed cycle and regional flow evidence."""
import numpy as np

from event_data import runs
from hsmm import Posterior, segment
from pixel_emission import PixelEmission, with_elapsed_time

BRANCH = "pixel_set"
PHASES = ("IC", "Flat", "TO", "Swing")
EVENT_PAIRS = ((3, 0), (2, 3))
PHASE_QUANTILES = (.05, .95)


def phase_interval(marginal, start, stop, state, frames):
    anchor = (start+stop-1)//2
    start_weight, end_weight = marginal.phase_edge_weights(anchor, state)
    mass = float(start_weight.sum())
    if mass <= 0:
        raise ValueError("MAP phase midpoint has no posterior mass")
    lo = np.searchsorted(np.cumsum(start_weight), PHASE_QUANTILES[0]*mass)
    hi = np.searchsorted(np.cumsum(end_weight), PHASE_QUANTILES[1]*mass)
    return int(frames[lo]), int(frames[hi]), min(1., mass)


def flow_evidence(raw, valid):
    """Soft phase cues from Step 5, using image vertical direction (down is positive)."""
    slow = (raw[:, [9, 23, 37]] + raw[:, [10, 24, 38]]) / 2
    vertical = raw[:, [6, 20, 34]]
    scale = max(.35, float(np.quantile(np.abs(vertical[valid]), .8)))
    heel, mid, toe = slow.T
    heel_y, _, toe_y = np.tanh(vertical / scale).T
    downward_heel = np.maximum(heel_y, 0)
    upward_toe = np.maximum(-toe_y, 0)
    return np.column_stack([
        1.4 * heel + .8 * (heel-toe) + .65 * downward_heel,
        2.35 * mid + .6 * (heel+toe)/2 - .3 * (np.abs(heel_y)+np.abs(toe_y))/2,
        1.0 * toe + .85 * (toe-heel) + 1.0 * upward_toe,
        2.65 * (1-slow.mean(axis=1)),
    ])


class CyclicModel:
    def __init__(self):
        self.k = len(PHASES)
        self.emission = PixelEmission()
        self.log_trans = np.full((self.k, self.k), -np.inf)
        phase = np.arange(self.k)
        self.log_trans[phase, (phase+1) % self.k] = 0.
        self.log_bos = np.full(self.k, -np.log(self.k))

    def fit_durations(self, counts):
        """Smooth each phase's counts with one pooled pseudo-observation."""
        self.duration_counts = counts.copy()
        shared = counts.sum(1) + 1. / len(counts)
        shared /= shared.sum()
        self.duration_probability = (counts + shared[:, None]) / (counts.sum(0) + 1.)

    def duration_scores(self, n):
        probability = self.duration_probability
        survival = np.cumsum(probability[::-1], axis=0)[::-1]
        residual = survival / survival.sum(0)
        residual_survival = np.cumsum(residual[::-1], axis=0)[::-1]
        limit = min(n, len(probability))
        scores = np.full((n, limit+1, self.k), -np.inf)
        duration = np.arange(1, limit+1)
        for a in range(n):
            length = min(n-a, limit)
            scores[a, 1:length+1] = np.log(probability[:length])
        # The first phase is sampled part-way through. The last may continue
        # beyond the observation, so only its observed minimum is known.
        scores[0, 1:] = np.log(residual[:limit])
        scores[n-duration, duration] = np.log(survival[:limit])
        if n == limit:
            scores[0, n] = np.log(residual_survival[n-1])
        return scores

    def emissions_with_cues(self, run, initialize=False):
        n = len(run["moments"])
        score = np.zeros((n, n+1, self.k)) if initialize else self.emission.scores(run, self.duration_scores(n))
        observed_cues = np.where(run["observed"][:, None], run["cue"], 0.)
        cumulative = np.vstack([np.zeros(self.k), np.cumsum(observed_cues, axis=0)])
        for length in range(1, score.shape[1]):
            a = np.arange(n-length+1)
            score[a, length] += cumulative[a+length] - cumulative[a]
        return score


def make_runs(base, moments, cues, *, include_missing=False):
    valid = base["valid"]
    times = base["f"]["t_real_s_v2"]
    if include_missing:
        blocks = runs(np.ones_like(valid), base["groups"])
        intervals = []
        for ids in blocks:
            observed = np.flatnonzero(valid[ids])
            if observed.size:
                intervals.append(ids[observed[0]:observed[-1]+1])
    else:
        intervals = [ids for ids in base["runs"] if len(ids) >= 2]
        if not intervals:
            intervals = base["runs"]
    # The scorer uses only the time at each observation's first frame.
    return [dict(ids=ids, moments=moments[ids], cue=cues[ids], edges=times[ids], observed=valid[ids])
            for ids in intervals]


def fit_and_decode(model, runs, iterations=4, progress=print, *, inference_runs=None):
    if not runs:
        raise ValueError("HSMM fitting requires at least one usable flow observation")
    paths = [segment(model.emissions_with_cues(run, initialize=True), model)[0] for run in runs]
    counts = np.zeros((max(len(run["moments"]) for run in runs), model.k))
    for path in paths:
        for a, b, state in path[1:-1]:
            counts[b-a-1, state] += 1
    for iteration in range(iterations):
        model.fit_durations(counts)
        total = np.zeros((model.k, *runs[0]["moments"].shape[1:2], 6, 6))
        for run, path in zip(runs, paths):
            for start, stop, phase in path:
                elapsed = run["edges"][start:stop] - run["edges"][start]
                total[phase] += with_elapsed_time(run["moments"][start:stop], elapsed).sum(axis=0)
        model.emission.fit(total)
        if iteration + 1 == iterations:
            break
        updated = []
        counts.fill(0)
        for run in runs:
            scores = model.emissions_with_cues(run)
            updated.append(segment(scores, model)[0])
            Posterior(scores, model).count_complete_durations(counts)
        changed = sum(a != b for a, b in zip(paths, updated))
        progress(f"Iteration {iteration+1}: {changed}/{len(runs)} run paths changed", flush=True)
        paths = updated
    for run in (runs if inference_runs is None else inference_runs):
        score = model.emissions_with_cues(run)
        path, _ = segment(score, model)
        yield run, path, Posterior(score, model)


def boundary_interval(weights, frames, level=.9):
    """Equal-tailed interval conditional on the first event in the supplied window."""
    mass = float(weights.sum())
    if mass <= 0:
        raise ValueError("MAP event has no posterior mass")
    cdf = np.cumsum(weights) / mass
    tail = (1-level)/2
    a, b = np.searchsorted(cdf, (tail, 1-tail))
    return int(frames[a]), int(frames[b]), min(1., mass)
