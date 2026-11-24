"""Four-state gait HSMM with a directed cycle and regional flow evidence."""
import numpy as np

from rff_hsmm import Model, posterior, segment

PHASES = ("IC", "Flat", "TO", "Swing")
EVENT_PAIRS = ((3, 0), (2, 3))
# Seconds: lower bound, lognormal median, upper bound for each complete phase.
WALK_DURATIONS = {
    "IC": (.06, .11, .24),
    "Flat": (.22, .43, .90),
    "TO": (.06, .13, .26),
    "Swing": (.25, .42, .75),
}
SLOW_DURATIONS = {
    "IC": (.06, .13, .27),
    "Flat": (.28, .70, 1.12),
    "TO": (.06, .16, .30),
    "Swing": (.36, .57, .92),
}
RUN_DURATIONS = {
    "IC": (.04, .07, .15),
    "Flat": (.06, .12, .33),
    "TO": (.04, .07, .17),
    "Swing": (.22, .39, .72),
}
FAST_RUN_DURATIONS = {
    "IC": (.04, .06, .13),
    "Flat": (.04, .09, .27),
    "TO": (.04, .06, .15),
    "Swing": (.20, .37, .64),
}
OVERGROUND_DURATIONS = {
    "Walk_Slow": SLOW_DURATIONS,
    "Run_Comfortable": RUN_DURATIONS,
    "Run_Fast": FAST_RUN_DURATIONS,
}


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


class CyclicModel(Model):
    def __init__(self, dimensions, durations=WALK_DURATIONS, observation_weight=1., seed=20261117):
        super().__init__(classes=len(PHASES), dimensions=dimensions, rff=12, noise=.5, seed=seed)
        bounds = np.array([durations[p] for p in PHASES], dtype=float)
        if bounds.shape != (4, 3) or not np.isfinite(bounds).all() or not (
            (bounds[:, 0] > 0) & (bounds[:, 0] < bounds[:, 1]) & (bounds[:, 1] < bounds[:, 2])
        ).all():
            raise ValueError("Each phase requires finite bounds with 0 < min < median < max")
        self.minimum, self.median, self.maximum = bounds.T
        self.log_trans = np.full((self.k, self.k), -np.inf)
        phase = np.arange(self.k)
        self.log_trans[phase, (phase+1) % self.k] = 0.
        self.log_bos = np.full(self.k, -np.log(self.k))
        self.phase_labels = phase
        self.duration_mean = np.log(self.median)
        self.duration_sd = np.full(self.k, .42)
        # Temper the coordinate product so higher RFF dimension alone does not
        # drown out the motion cues or sharpen the boundary time.
        self.observation_weight = observation_weight
        self.likelihood_weight = 4. * observation_weight / dimensions

    def emissions_with_cues(self, run, use_gp=True):
        n = len(run["x"])
        score = self.emissions(run) if use_gp else self.duration_scores(run["edges"], min(n, self.max_samples))
        cumulative = np.vstack([np.zeros(self.k), self.observation_weight * np.cumsum(run["cue"], axis=0)])
        # A run shorter than the shortest complete phase supplies no observed
        # transition, even if two censored fragments would fit algebraically.
        if run["edges"][-1] - run["edges"][0] < self.minimum.min() - 1e-9:
            score[:, :n] = -np.inf
        for length in range(1, score.shape[1]):
            a = np.arange(n-length+1)
            score[a, length] += cumulative[a+length] - cumulative[a]
        return score


def make_runs(base, model_input, cues):
    times = base["f"]["t_real_s_v2"]
    return [dict(ids=ids, x=model_input[ids], cue=cues[ids],
                 edges=np.r_[times[ids], times[ids[-1]+1]])
            for ids in base["runs"] if len(ids) >= 2]


def fit_and_decode(model, runs, iterations=4, progress=print):
    paths = [segment(model.emissions_with_cues(run, use_gp=False), model)[0] for run in runs]
    for iteration in range(iterations):
        total = model.empty_stats()
        for run, path in zip(runs, paths):
            current = model.stats(run, path)
            for key in total:
                total[key] += current[key]
        model.fit_gp(total)
        updated = [segment(model.emissions_with_cues(run), model)[0] for run in runs]
        changed = sum(a != b for a, b in zip(paths, updated))
        progress(f"Iteration {iteration+1}: {changed}/{len(runs)} run paths changed", flush=True)
        paths = updated
        if changed == 0:
            break
    for run, path in zip(runs, paths):
        if not path:
            continue
        score = model.emissions_with_cues(run)
        occupancy, _, ic, to = posterior(score, model, event_pairs=EVENT_PAIRS)
        yield run, path, occupancy, (ic, to)


def boundary_interval(weights, frames, index, fps, min_cycle_s, level=.9):
    """Equal-tailed event window conditional on one boundary near its MAP instance."""
    # Width strictly below one minimum complete cycle so this bracket cannot
    # contain two transitions of the same type along any admissible path.
    radius = max(0, (int(np.ceil(min_cycle_s*fps - 1e-9)) - 1) // 2)
    left, right = max(1, index-radius), min(len(frames), index+radius+1)
    local_frames = frames[left:right]
    if not len(local_frames) or (local_frames[-1]-local_frames[0])/fps >= min_cycle_s - 1e-9:
        raise ValueError("Boundary interval expects contiguous source frames")
    local = weights[left:right]
    mass = float(local.sum())
    if mass <= 0:
        raise ValueError("MAP event has no posterior mass")
    cdf = np.cumsum(local) / mass
    tail = (1-level)/2
    a, b = np.searchsorted(cdf, (tail, 1-tail))
    return int(local_frames[a]), int(local_frames[b]), min(1., mass)
