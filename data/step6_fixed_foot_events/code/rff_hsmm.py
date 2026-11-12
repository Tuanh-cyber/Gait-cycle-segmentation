"""RFF Bayesian GP segment emissions + explicit-duration HSMM, NumPy/SciPy.

Blocked FFBS updates remove the entire current run before fitting predictive
parameters. Missing regional coordinates are omitted, never zero observations.
"""
import numpy as np
from scipy.special import logsumexp, log_ndtr, ndtr

from event_data import semantic_phase, two_means


def probabilities(log_weights):
    p = np.exp(log_weights - logsumexp(log_weights))
    return p / p.sum()


class Model:
    def __init__(self, classes=3, rff=12, seed=20261112, noise=.5, min_duration=.08, max_duration=.75, max_samples=256, phase_penalty=4.):
        self.k, self.m, self.d = classes, rff + 1, 16
        self.rng = np.random.default_rng(seed)
        self.omega = self.rng.normal(0, 1 / .15, rff)
        self.offset = self.rng.uniform(0, 2 * np.pi, rff)
        self.noise = noise
        self.minimum, self.maximum, self.max_samples = min_duration, max_duration, max_samples
        self.phase_penalty = phase_penalty

    def phi(self, t):
        t = np.asarray(t)
        return np.concatenate([np.ones(t.shape + (1,)), np.sqrt(2 / (self.m - 1)) * np.cos(t[..., None] * self.omega + self.offset)], axis=-1)

    def empty_stats(self):
        return dict(xx=np.zeros((self.k, self.d, self.m, self.m)), xy=np.zeros((self.k, self.d, self.m)),
                    durations=np.zeros((self.k, 3)), slow=np.zeros((self.k, 2)), trans=np.zeros((self.k, self.k)), bos=np.zeros(self.k))

    def stats(self, run, segments):
        result = self.empty_stats()
        for j, (a, b, state) in enumerate(segments):
            phi = self.phi(run["edges"][a:b] - run["edges"][a])
            y = run["x"][a:b]
            for d in range(self.d):
                observed = np.isfinite(y[:, d])
                design = phi[observed]
                result["xx"][state, d] += design.T @ design
                result["xy"][state, d] += design.T @ y[observed, d]
            duration = max(run["edges"][b] - run["edges"][a], .001)
            ld = np.log(duration)
            result["durations"][state] += [1, ld, ld * ld]
            slow = run["contact"][a:b]
            result["slow"][state] += [np.nansum(slow), np.isfinite(slow).sum()]
            if j:
                result["trans"][segments[j - 1][2], state] += 1
            else:
                result["bos"][state] += 1
        return result

    def fit(self, statistics, semantic_centers):
        precision = np.eye(self.m)[None, None] + statistics["xx"] / self.noise
        self.cov = np.linalg.inv(precision)
        self.weight = np.einsum("kdij,kdj->kdi", self.cov, statistics["xy"] / self.noise)
        n, total, square = statistics["durations"].T
        # Weak duration prior, updated from inferred segments, never from IC/TO labels.
        prior_log, prior_strength = np.log(.18), 2.
        self.duration_mean = (total + prior_strength * prior_log) / (n + prior_strength)
        second = (square + prior_strength * (prior_log ** 2 + .7 ** 2)) / (n + prior_strength)
        self.duration_sd = np.maximum(np.sqrt(np.maximum(second - self.duration_mean ** 2, 0)), .35)
        count = statistics["slow"][:, 1]
        levels = np.divide(statistics["slow"][:, 0], count, out=np.full(self.k, np.nan), where=count > 0)
        self.phase_labels = np.array([semantic_phase(v, semantic_centers) for v in levels], np.int8)
        self.slow_means = levels
        trans = statistics["trans"] + 1.
        for i in range(self.k):
            for j in range(self.k):
                if self.phase_labels[i] >= 0 and self.phase_labels[j] >= 0 and self.phase_labels[i] != self.phase_labels[j]:
                    trans[i, j] *= np.exp(-self.phase_penalty)
        trans /= trans.sum(1, keepdims=True)
        self.log_trans = np.log(trans)
        self.log_bos = np.log((statistics["bos"] + 1) / (statistics["bos"].sum() + self.k))

    def emissions(self, run):
        x, edges = run["x"], run["edges"]
        n, limit = len(x), min(len(x), self.max_samples)
        scores = np.full((n, limit + 1, self.k), -np.inf)
        accumulated = np.zeros((n, self.k))
        step = max(float(np.median(np.diff(edges))), .001)
        normalizer = np.maximum(ndtr((np.log(self.maximum) - self.duration_mean) / self.duration_sd) - ndtr((np.log(self.minimum) - self.duration_mean) / self.duration_sd), 1e-12)
        for offset in range(limit):
            a = np.arange(n - offset)
            ends = a + offset + 1
            lag = edges[a + offset] - edges[a]
            phi = self.phi(lag)
            y = x[a + offset]
            observed = np.isfinite(y)
            for state in range(self.k):
                mu = phi @ self.weight[state].T
                variance = self.noise + np.einsum("ni,dij,nj->nd", phi, self.cov[state], phi, optimize=True)
                residual = np.where(observed, y - mu, 0)
                point_log = -.5 * np.where(observed, np.log(2 * np.pi * variance) + residual ** 2 / variance, 0).sum(1)
                accumulated[a, state] += point_log
            duration = edges[ends] - edges[a]
            boundary = (a == 0) | (ends == n)
            allowed = (duration <= self.maximum) & ((duration >= self.minimum) | boundary)
            # At least two observations, except a censored endpoint singleton.
            allowed &= (offset >= 1) | boundary
            log_duration = np.log(np.maximum(duration, .001))
            z = (log_duration[:, None] - self.duration_mean) / self.duration_sd
            prior = -.5 * z ** 2 - np.log(self.duration_sd) - .5 * np.log(2 * np.pi) - log_duration[:, None] + np.log(step) - np.log(normalizer)
            prior[boundary] = log_ndtr(-z[boundary])
            prior[(a == 0) & (ends == n)] = 0
            scores[a[allowed], offset + 1] = accumulated[a[allowed]] + prior[allowed]
        return scores


def forward(scores, log_trans, log_bos, viterbi=False):
    n, lengths, classes = scores.shape
    alpha = np.full((n + 1, classes), -np.inf)
    incoming = np.full_like(alpha, -np.inf)
    incoming[0] = log_bos
    chosen_length = np.zeros((n + 1, classes), int)
    chosen_previous = np.full((n + 1, classes), -1, int)
    incoming_previous = np.full_like(chosen_previous, -1)
    for end in range(1, n + 1):
        duration = np.arange(1, min(lengths - 1, end) + 1)
        start = end - duration
        values = incoming[start] + scores[start, duration]
        if viterbi:
            best = np.argmax(values, axis=0)
            alpha[end] = values[best, np.arange(classes)]
            chosen_length[end] = duration[best]
            chosen_previous[end] = incoming_previous[start[best], np.arange(classes)]
            options = alpha[end, :, None] + log_trans
            incoming_previous[end] = np.argmax(options, axis=0)
            incoming[end] = options[incoming_previous[end], np.arange(classes)]
        else:
            alpha[end] = logsumexp(values, axis=0)
            incoming[end] = logsumexp(alpha[end, :, None] + log_trans, axis=0)
    return alpha, incoming, chosen_length, chosen_previous


def segment(scores, model, sample=False):
    alpha, incoming, lengths, previous = forward(scores, model.log_trans, model.log_bos, not sample)
    n = len(scores)
    if not np.isfinite(alpha[n]).any():
        return [], -np.inf
    state = int(model.rng.choice(model.k, p=probabilities(alpha[n]))) if sample else int(np.argmax(alpha[n]))
    end, result = n, []
    while end:
        if sample:
            ds = np.arange(1, min(scores.shape[1] - 1, end) + 1)
            start = end - ds
            weights = incoming[start, state] + scores[start, ds, state]
            length = int(model.rng.choice(ds, p=probabilities(weights)))
            a = end - length
            preceding = int(model.rng.choice(model.k, p=probabilities(alpha[a] + model.log_trans[:, state]))) if a else -1
        else:
            length = int(lengths[end, state])
            a, preceding = end - length, int(previous[end, state])
        if length < 1:
            raise ValueError("Invalid HSMM backtracking duration")
        result.append((a, end, state))
        end, state = a, preceding
    return result[::-1], float(logsumexp(alpha[n])) if sample else float(np.max(alpha[n]))


def posterior(scores, model):
    n, lengths, classes = scores.shape
    alpha, incoming, _, _ = forward(scores, model.log_trans, model.log_bos)
    tail = np.full((n + 1, classes), -np.inf)
    tail[n] = 0
    first_state = np.full((n, classes), -np.inf)
    for a in range(n - 1, -1, -1):
        duration = np.arange(1, min(lengths - 1, n - a) + 1)
        first_state[a] = logsumexp(scores[a, duration] + tail[a + duration], axis=0)
        tail[a] = logsumexp(model.log_trans + first_state[a][None], axis=1)
    z = logsumexp(model.log_bos + first_state[0])
    diff = np.zeros((n + 1, classes))
    boundary = np.zeros(n)
    p_ic, p_to = np.zeros(n), np.zeros(n)
    for a in range(n):
        duration = np.arange(1, min(lengths - 1, n - a) + 1)
        weight = np.exp(incoming[a] + scores[a, duration] + tail[a + duration] - z)
        diff[a] += weight.sum(0)
        np.add.at(diff, a + duration, -weight)
        if a:
            pair = np.exp(alpha[a, :, None] + model.log_trans + first_state[a][None] - z)
            boundary[a] = pair.sum()
            for before in range(classes):
                for after in range(classes):
                    label_pair = (model.phase_labels[before], model.phase_labels[after])
                    if label_pair == (0, 1):
                        p_ic[a] += pair[before, after]
                    elif label_pair == (1, 0):
                        p_to[a] += pair[before, after]
    occupancy = np.maximum(np.cumsum(diff[:-1], axis=0), 0)
    if not np.allclose(occupancy.sum(1), 1, atol=1e-6):
        raise ValueError("HSMM posterior occupancy does not sum to one")
    return occupancy, boundary, p_ic, p_to


def initial_segments(run, centers, model):
    labels = ((run["x"][:, 12:16, None] - centers.T[None]) ** 2).sum(1).argmin(1)
    for _ in range(2):
        for i in range(1, len(labels) - 1):
            if labels[i - 1] == labels[i + 1]:
                labels[i] = labels[i - 1]
    edges = np.r_[0, np.flatnonzero(np.diff(labels) != 0) + 1, len(labels)]
    result = []
    for a, b in zip(edges[:-1], edges[1:]):
        while a < b:
            end = min(int(b), int(a + model.max_samples))
            while end > a + 1 and run["edges"][end] - run["edges"][a] > model.maximum:
                end -= 1
            result.append((int(a), int(end), int(labels[a])))
            a = end
    return result


def train(videos, model, sweeps=3, progress=print):
    training_runs = []
    for v in videos:
        for ids in v["runs"]:
            if len(ids) < 2:
                continue
            training_runs.append(dict(video_id=v["video_id"], ids=ids, x=v["x"][ids], contact=v["contact"][ids],
                                      edges=np.r_[v["f"]["t_real_s_v2"][ids], v["f"]["t_real_s_v2"][ids[-1] + 1]]))
    pool = np.concatenate([r["x"][:, 12:16] for r in training_runs])
    centers = pool[model.rng.choice(len(pool), model.k, replace=False)].copy()
    for _ in range(40):
        labels = ((pool[:, :, None] - centers.T[None]) ** 2).sum(1).argmin(1)
        newer = np.array([pool[labels == k].mean(0) if (labels == k).any() else centers[k] for k in range(model.k)])
        if np.allclose(newer, centers, atol=1e-6):
            break
        centers = newer
    semantic_centers = two_means(np.concatenate([r["contact"] for r in training_runs]))
    segments = [initial_segments(r, centers, model) for r in training_runs]
    per_run = [model.stats(r, s) for r, s in zip(training_runs, segments)]
    total = model.empty_stats()
    for statistics in per_run:
        for key in total:
            total[key] += statistics[key]
    trace = []
    for sweep in range(sweeps):
        changed = 0
        for position, index in enumerate(model.rng.permutation(len(training_runs))):
            for key in total:
                total[key] -= per_run[index][key]
            model.fit(total, semantic_centers)
            scores = model.emissions(training_runs[index])
            proposed, _ = segment(scores, model, sample=True)
            if not proposed:
                proposed = segments[index]
            changed += int(proposed != segments[index])
            segments[index] = proposed
            per_run[index] = model.stats(training_runs[index], proposed)
            for key in total:
                total[key] += per_run[index][key]
            if (position + 1) % 40 == 0:
                progress(f"HSMM sweep {sweep+1}/{sweeps}: {position+1}/{len(training_runs)} runs", flush=True)
        trace.append(dict(sweep=sweep + 1, changed_runs=changed, segments=sum(map(len, segments))))
        progress(f"HSMM sweep {sweep+1}: {changed} changed runs; {trace[-1]['segments']} latent segments", flush=True)
    model.fit(total, semantic_centers)
    return training_runs, trace, total, semantic_centers


def infer_video(v, model):
    n = len(v["valid"])
    phase, latent = np.full(n, -1, np.int8), np.full(n, -1, np.int16)
    occupancy = np.full((n, model.k), np.nan)
    boundary, p_ic, p_to = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    cuts, all_segments = [], []
    for ids in v["runs"]:
        if len(ids) < 2:
            continue
        run = dict(x=v["x"][ids], edges=np.r_[v["f"]["t_real_s_v2"][ids], v["f"]["t_real_s_v2"][ids[-1] + 1]])
        scores = model.emissions(run)
        segments, _ = segment(scores, model)
        if not segments:
            continue
        occ, b, ic, to = posterior(scores, model)
        occupancy[ids], boundary[ids], p_ic[ids], p_to[ids] = occ, b, ic, to
        for a, end, state in segments:
            latent[ids[a:end]] = state
            phase[ids[a:end]] = model.phase_labels[state]
            if a:
                cuts.append(int(ids[a]))
            all_segments.append(dict(first_index=int(ids[a]), stop_index=int(ids[end - 1] + 1), latent_class=state,
                                     phase=int(model.phase_labels[state])))
    return dict(phase=phase, latent=latent, state_probability=occupancy, boundary_probability=boundary,
                IC_probability=p_ic, TO_probability=p_to, cuts=cuts, segments=all_segments)
