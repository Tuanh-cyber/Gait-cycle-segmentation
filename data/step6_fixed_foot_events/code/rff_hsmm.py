"""RFF Bayesian GP segment emissions and explicit-duration HSMM inference."""
import numpy as np
from scipy.special import logsumexp, log_ndtr, ndtr

DURATION_TOLERANCE_S = 1e-9


class Model:
    def __init__(self, classes=4, rff=12, seed=20261112, noise=.5, min_duration=.08, max_duration=.75,
                 max_samples=256, dimensions=16):
        self.k, self.m, self.d = classes, rff + 1, dimensions
        rng = np.random.default_rng(seed)
        self.omega = rng.normal(0, 1 / .15, rff)
        self.offset = rng.uniform(0, 2 * np.pi, rff)
        self.noise = noise
        self.minimum, self.maximum, self.max_samples = min_duration, max_duration, max_samples

    def phi(self, t):
        t = np.asarray(t)
        return np.concatenate([np.ones(t.shape + (1,)), np.sqrt(2 / (self.m - 1)) * np.cos(t[..., None] * self.omega + self.offset)], axis=-1)

    def empty_stats(self):
        return dict(xx=np.zeros((self.k, self.d, self.m, self.m)), xy=np.zeros((self.k, self.d, self.m)))

    def stats(self, run, segments):
        result = self.empty_stats()
        for a, b, state in segments:
            phi = self.phi(run["edges"][a:b] - run["edges"][a])
            y = run["x"][a:b]
            if np.isfinite(y).all():
                result["xx"][state] += phi.T @ phi
                result["xy"][state] += (phi.T @ y).T
            else:
                for d in range(self.d):
                    observed = np.isfinite(y[:, d])
                    design = phi[observed]
                    result["xx"][state, d] += design.T @ design
                    result["xy"][state, d] += design.T @ y[observed, d]
        return result

    def fit_gp(self, statistics):
        design_products = statistics["xx"]
        shared_design = np.all(design_products == design_products[:, :1])
        precision = np.eye(self.m) + (design_products[:, :1] if shared_design else design_products) / self.noise
        self.cov = np.broadcast_to(np.linalg.inv(precision), design_products.shape)
        self.weight = np.einsum("kdij,kdj->kdi", self.cov, statistics["xy"] / self.noise)

    def emissions(self, run):
        x, edges = run["x"], run["edges"]
        n, limit = len(x), min(len(x), self.max_samples)
        scores = self.duration_scores(edges, limit)
        accumulated = np.zeros((n, self.k))
        likelihood_weight = getattr(self, "likelihood_weight", 1.)
        shared_covariance = np.all(self.cov == self.cov[:, :1])
        for offset in range(limit):
            a = np.arange(n - offset)
            lag = edges[a + offset] - edges[a]
            phi = self.phi(lag)
            y = x[a + offset]
            observed = np.isfinite(y)
            for state in range(self.k):
                mu = phi @ self.weight[state].T
                if shared_covariance:
                    variance = self.noise + ((phi @ self.cov[state, 0]) * phi).sum(1, keepdims=True)
                else:
                    variance = self.noise + np.einsum("ni,dij,nj->nd", phi, self.cov[state], phi, optimize=True)
                residual = np.where(observed, y - mu, 0)
                point_log = -.5 * np.where(observed, np.log(2 * np.pi * variance) + residual ** 2 / variance, 0).sum(1)
                accumulated[a, state] += point_log
            scores[a, offset+1] += likelihood_weight * accumulated[a]
        return scores

    def duration_scores(self, edges, limit):
        n = len(edges) - 1
        scores = np.full((n, limit+1, self.k), -np.inf)
        step = max(float(np.median(np.diff(edges))), .001)
        normalizer = np.maximum(ndtr((np.log(self.maximum) - self.duration_mean) / self.duration_sd)
                                - ndtr((np.log(self.minimum) - self.duration_mean) / self.duration_sd), 1e-12)
        for offset in range(limit):
            a = np.arange(n - offset)
            ends = a + offset + 1
            duration = (edges[ends] - edges[a])[:, None]
            boundary = (a == 0) | (ends == n)
            allowed = (duration <= self.maximum + DURATION_TOLERANCE_S) & (
                (duration >= self.minimum - DURATION_TOLERANCE_S) | boundary[:, None]
            )
            # At least two observations, except a censored endpoint singleton.
            allowed &= (offset >= 1) | boundary[:, None]
            log_duration = np.log(np.maximum(duration, .001))
            z = (log_duration - self.duration_mean) / self.duration_sd
            prior = -.5 * z ** 2 - np.log(self.duration_sd) - .5 * np.log(2 * np.pi) - log_duration + np.log(step) - np.log(normalizer)
            prior[boundary] = log_ndtr(-z[boundary])
            prior[(a == 0) & (ends == n)] = 0
            scores[a, offset+1] = np.where(allowed, prior, -np.inf)
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


def segment(scores, model):
    alpha, _, lengths, previous = forward(scores, model.log_trans, model.log_bos, viterbi=True)
    n = len(scores)
    if not np.isfinite(alpha[n]).any():
        return [], -np.inf
    state = int(np.argmax(alpha[n]))
    end, result = n, []
    while end:
        length = int(lengths[end, state])
        a, preceding = end - length, int(previous[end, state])
        if length < 1:
            raise ValueError("Invalid HSMM backtracking duration")
        result.append((a, end, state))
        end, state = a, preceding
    return result[::-1], float(np.max(alpha[n]))


def posterior(scores, model, phase_labels=None, event_pairs=((0, 1), (1, 0))):
    n, lengths, classes = scores.shape
    labels = model.phase_labels if phase_labels is None else phase_labels
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
                    label_pair = (labels[before], labels[after])
                    if label_pair == event_pairs[0]:
                        p_ic[a] += pair[before, after]
                    elif label_pair == event_pairs[1]:
                        p_to[a] += pair[before, after]
    occupancy = np.maximum(np.cumsum(diff[:-1], axis=0), 0)
    if not np.allclose(occupancy.sum(1), 1, atol=1e-6):
        raise ValueError("HSMM posterior occupancy does not sum to one")
    return occupancy, boundary, p_ic, p_to
