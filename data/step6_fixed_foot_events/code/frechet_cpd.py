"""Local Fréchet mean/variance scans on regional Euclidean flow features."""
import numpy as np

from event_data import runs, semantic_phase


def scan(x, min_count=2):
    """Dubey–Mueller statistic: Euclidean cross-variance term is 4||muL-muR||^4."""
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x[None]
    batch, n, dimensions = x.shape
    k = np.arange(min_count, n - min_count + 1)
    if not len(k):
        return k, np.empty((batch, 0))
    prefix = np.cumsum(x, axis=1)
    square = np.cumsum((x * x).sum(2), axis=1)
    ml = prefix[:, k - 1] / k[None, :, None]
    mr = (prefix[:, -1, None] - prefix[:, k - 1]) / (n - k)[None, :, None]
    vl = np.maximum(square[:, k - 1] / k - (ml * ml).sum(2), 0)
    vr = np.maximum((square[:, -1, None] - square[:, k - 1]) / (n - k) - (mr * mr).sum(2), 0)
    dist_mean = ((ml - mr) ** 2).sum(2)
    centered_sq = ((x - x.mean(1, keepdims=True)) ** 2).sum(2)
    sigma2 = np.maximum(centered_sq.var(1), 1e-8)
    u = k / n
    statistic = n * u * (1 - u) * ((vl - vr) ** 2 + 4 * dist_mean ** 2) / sigma2[:, None]
    return k, statistic


def detect(v, seed=20261112, bootstrap=99, alpha=.05, window_s=.60, min_duration=.08, block_length=1):
    rng = np.random.default_rng(seed)
    t, x = v["f"]["t_real_s_v2"], v["x"]
    candidates = {}
    for region in range(4):
        region_valid = v["valid"] & np.isfinite(x[:, region * 4:region * 4 + 4]).all(1)
        for ids in runs(region_valid, v["groups"]):
            if len(ids) < 6:
                continue
            seen = set()
            for center in range(2, len(ids) - 2, 2):
                a = max(0, np.searchsorted(t[ids], t[ids[center]] - window_s / 2))
                b = min(len(ids), np.searchsorted(t[ids], t[ids[center]] + window_s / 2, side="right"))
                if b - a < 6:
                    a, b = max(0, center - 4), min(len(ids), center + 5)
                if (a, b) in seen or b - a < 6:
                    continue
                seen.add((a, b))
                values = x[ids[a:b], region * 4:region * 4 + 4]
                k, statistic = scan(values)
                allowed = (t[ids[a + k]] - t[ids[a]] >= min_duration) & (t[ids[b - 1]] - t[ids[a + k]] >= min_duration * .5)
                if not allowed.any():
                    continue
                scores = np.where(allowed, statistic[0], -np.inf)
                best = int(np.argmax(scores))
                observed = float(scores[best])
                n = len(values)
                if block_length == 1:
                    sampled = rng.integers(0, n, (bootstrap, n))
                else:
                    starts = rng.integers(0, n, (bootstrap, int(np.ceil(n / block_length))))
                    sampled = ((starts[..., None] + np.arange(block_length)) % n).reshape(bootstrap, -1)[:, :n]
                _, null = scan(values[sampled])
                p = (1 + (null[:, allowed].max(1) >= observed).sum()) / (bootstrap + 1)
                if p > alpha:
                    continue
                index = int(ids[a + k[best]])
                entry = dict(p=float(p), score=observed, region=region, window_first=int(ids[a]), window_last=int(ids[b - 1]))
                if index not in candidates or (p, -observed) < (candidates[index]["p"], -candidates[index]["score"]):
                    candidates[index] = entry
    kept = {}
    for ids in v["runs"]:
        proposed = [i for i in candidates if ids[0] < i <= ids[-1]]
        chosen = []
        for i in sorted(proposed, key=lambda i: (candidates[i]["p"], -candidates[i]["score"])):
            if any(abs(i - j) < 2 or abs(t[i] - t[j]) < min_duration for j in chosen):
                continue
            if t[i] - t[ids[0]] < min_duration or t[ids[-1] + 1] - t[i] < min_duration:
                continue
            chosen.append(i)
            kept[i] = candidates[i]
    phase = np.full(len(t), -1, np.int8)
    for ids in v["runs"]:
        edges = [int(ids[0])] + sorted(i for i in kept if ids[0] < i <= ids[-1]) + [int(ids[-1] + 1)]
        for a, b in zip(edges[:-1], edges[1:]):
            level = float(np.nanmedian(v["contact"][a:b]))
            phase[a:b] = semantic_phase(level, v["centers"])
    return phase, kept
