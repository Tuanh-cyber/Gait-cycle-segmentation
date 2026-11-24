"""Viterbi and forward–backward for an explicit-duration HSMM."""
import numpy as np
from scipy.special import logsumexp


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


class Posterior:
    def __init__(self, scores, model):
        self.scores = scores
        self.n, lengths, classes = scores.shape
        self.limit = lengths - 1
        self.transitions = model.log_trans
        self.alpha, self.incoming, _, _ = forward(scores, self.transitions, model.log_bos)
        self.tail = np.full((self.n + 1, classes), -np.inf)
        self.tail[self.n] = 0
        self.first_state = np.full_like(self.tail, -np.inf)
        for a in range(self.n - 1, -1, -1):
            duration = np.arange(1, min(self.limit, self.n - a) + 1)
            self.first_state[a] = logsumexp(scores[a, duration] + self.tail[a + duration], axis=0)
            self.tail[a] = logsumexp(self.transitions + self.first_state[a][None], axis=1)
        self.log_evidence = logsumexp(model.log_bos + self.first_state[0])
        if not np.isfinite(self.log_evidence):
            raise ValueError("No admissible path for HSMM posterior")

    def segment_weights(self):
        for a in range(self.n):
            duration = np.arange(1, min(self.limit, self.n - a) + 1)
            log_weight = self.incoming[a] + self.scores[a, duration] + self.tail[a + duration]
            yield a, duration, np.exp(log_weight - self.log_evidence)

    def count_complete_durations(self, counts):
        for a, duration, weight in self.segment_weights():
            if a:
                complete = a + duration < self.n
                counts[duration[complete] - 1] += weight[complete]

    def occupancy(self):
        diff = np.zeros_like(self.tail)
        for a, duration, weight in self.segment_weights():
            diff[a] += weight.sum(0)
            diff[a + duration] -= weight
        result = np.maximum(np.cumsum(diff[:-1], axis=0), 0)
        if not np.allclose(result.sum(1), 1, atol=1e-6):
            raise ValueError("HSMM posterior occupancy does not sum to one")
        return result

    def phase_edge_weights(self, anchor, state):
        """Marginal start and inclusive-end weights for segments of state covering anchor."""
        start = np.zeros(self.n)
        end = np.zeros(self.n)
        for a in range(max(0, anchor+1-self.limit), anchor+1):
            duration = np.arange(anchor+1-a, min(self.limit, self.n-a)+1)
            stop = a+duration
            log_weight = self.incoming[a, state] + self.scores[a, duration, state] + self.tail[stop, state]
            weight = np.exp(log_weight-self.log_evidence)
            start[a] = weight.sum()
            end[stop-1] += weight
        return start, end

    def transitions_by_frame(self):
        pair = np.zeros((self.n, *self.transitions.shape))
        pair[1:] = np.exp(self.alpha[1:self.n, :, None] + self.transitions
                          + self.first_state[1:self.n, None, :] - self.log_evidence)
        return pair

    def first_event_weights(self, pair, start, stop):
        """Unconditional mass of the first matching transition at each frame in the window."""
        if not 1 <= start < stop <= self.n:
            raise ValueError("Boundary window must lie inside a continuous run")
        before, after = pair
        blocked = self.transitions.copy()
        blocked[before, after] = -np.inf
        incoming = self.incoming[:stop].copy()
        weights = np.empty(stop-start)
        for t in range(start, stop):
            duration = np.arange(1, min(self.limit, t) + 1)
            begin = t-duration
            alpha = logsumexp(incoming[begin] + self.scores[begin, duration], axis=0)
            weights[t-start] = np.exp(alpha[before] + self.transitions[before, after]
                                      + self.first_state[t, after] - self.log_evidence)
            incoming[t] = logsumexp(alpha[:, None] + blocked, axis=0)
        return weights


def posterior(scores, model, phase_labels=None, event_pairs=((0, 1), (1, 0))):
    marginal = Posterior(scores, model)
    labels = model.phase_labels if phase_labels is None else phase_labels
    pair = marginal.transitions_by_frame()
    matches = [(labels[:, None] == before) & (labels[None, :] == after) for before, after in event_pairs]
    return marginal.occupancy(), pair.sum(axis=(1, 2)), *(pair[:, use].sum(1) for use in matches)
