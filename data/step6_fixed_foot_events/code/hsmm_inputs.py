"""Regional flow statistics, pixel clouds and the embedding for one fixed foot.

No feature is reconstructed from saved medians. Missing masks stay missing.
Empty moving/slow subsets have explicitly defined neutral summaries (see README).
"""
import sys

import cv2
import numpy as np

from event_data import DATA, runs, save_npz, sha
from mask_overlay import decode_crop

REGIONS = ("heel", "mid", "toe")
LOCAL_NAMES = ("speed_p10", "speed_p25", "speed_p50", "speed_p75", "speed_p90",
               "vx_median", "vy_median", "vx_iqr", "vy_iqr", "slow_1", "slow_2",
               "direction_cos", "direction_sin", "slow_centroid_xi")
FEATURE_NAMES = [f"{r}.{f}" for r in REGIONS for f in LOCAL_NAMES]


def regional_statistics(cloud):
    """cloud columns: xi, eta, ground-relative vx, vy (L_leg/s)."""
    xi, _, vx, vy = np.asarray(cloud, float).T
    speed = np.hypot(vx, vy)
    moving, slow = speed > .2, speed < 2.
    direction = np.array([np.mean(vx[moving] / speed[moving]), np.mean(vy[moving] / speed[moving])]) if moving.any() else np.zeros(2)
    # No slow pixels => no slow cluster offset. slow_2=0 disambiguates this
    # convention from an actual cluster centered on the region centroid.
    centroid = xi[slow].mean() - xi.mean() if slow.any() else 0.
    return np.r_[np.percentile(speed, [10, 25, 50, 75, 90]), np.median(vx), np.median(vy),
                 np.ptp(np.percentile(vx, [25, 75])), np.ptp(np.percentile(vy, [25, 75])),
                 np.mean(speed < 1.), np.mean(slow), direction, centroid]


def build_clouds(v, destination, data=DATA):
    """Reapply existing Step5 compensation to regenerated/cached GMA pixels."""
    sys.path.insert(0, str(data / "step5_flow/code"))
    from s5_store import Store
    store_path = data / "step5_flow/flow_store" / (v["video_id"] + ".npz")
    if not store_path.exists():
        raise FileNotFoundError(f"Missing pixel flow cache: {store_path}; run Step 5 with --store-only first")
    store = Store(v["video_id"])
    with np.load(data / "step5_flow/features" / (v["video_id"] + ".npz")) as z:
        f = {k: z[k] for k in ("comp_tx", "comp_ty", "comp_slow", "dt_used", "L_leg", "s_final", "crop", "scale")}
    with np.load(data / "step4_footmask/masks_post" / (v["video_id"] + ".npz")) as z:
        mask = {k: z[k] for k in ("W", "H", "rle_ptr", "rle_counts", "origin", "axis", "l_foot")}
    n, slot = len(v["valid"]), v["slot"]
    raw = np.full((n, 42), np.nan)
    pointers = np.zeros((n, 3, 2), np.int64)
    counts = np.zeros((n, 3), int)
    fb_fraction = np.full((n, 3), np.nan)
    all_clouds, offset = [], 0
    eligible = v["valid"].copy()
    for row in np.flatnonzero(eligible):
        if int(row) not in store.pos:
            eligible[row] = False
            continue
        flow, fb, sb, box, scale, shape = store.get(int(row))
        if tuple(f["crop"][row]) != box or not np.isclose(f["scale"][row], scale, rtol=1e-5):
            raise ValueError("Regenerated flow crop/scale differs from saved Step5 compensation")
        axis, length = mask["axis"][row, slot].astype(float), float(mask["l_foot"][row, slot])
        dt, leg = float(f["dt_used"][row]), float(f["L_leg"][row])
        if not np.isfinite(np.r_[axis, length, dt, leg]).all() or min(length, dt, leg, np.linalg.norm(axis)) <= 0:
            eligible[row] = False
            continue
        unit = axis / np.linalg.norm(axis)
        sy, sx = np.indices(flow.shape[:2])
        px, py = (sx + sb[0]) / scale + box[0], (sy + sb[1]) / scale + box[1]
        dx, dy = px - mask["origin"][row, slot, 0], py - mask["origin"][row, slot, 1]
        xi = (dx * unit[0] + dy * unit[1]) / length
        eta = (-dx * unit[1] + dy * unit[0]) / length
        vx = (flow[..., 0] - f["comp_tx"][row]) * float(f["s_final"]) / (scale * leg * dt) - f["comp_slow"][row]
        vy = (flow[..., 1] - f["comp_ty"][row]) / (scale * leg * dt)
        for region in range(3):
            a, b = mask["rle_ptr"][row, slot, region + 1]
            cropped = decode_crop(mask["rle_counts"][int(a):int(b)], int(mask["W"]), int(mask["H"]), box)
            resized = cv2.resize(cropped.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            current = resized[sb[1]:sb[3], sb[0]:sb[2]]
            fb_fraction[row, region] = fb[current].mean() if current.any() else np.nan
            selected = current & fb & np.isfinite(vx) & np.isfinite(vy)
            cloud = np.column_stack([xi[selected], eta[selected], vx[selected], vy[selected]]).astype(np.float32)
            counts[row, region] = len(cloud)
            pointers[row, region] = [offset, offset + len(cloud)]
            all_clouds.append(cloud)
            offset += len(cloud)
            if len(cloud) >= 10:
                raw[row, region * 14:(region + 1) * 14] = regional_statistics(cloud)
    eligible &= np.isfinite(raw).all(1)
    points = np.concatenate(all_clouds) if all_clouds else np.empty((0, 4), np.float32)
    payload = dict(points=points, ptr=pointers, raw42=raw, valid=eligible, counts=counts,
                   fb_fraction=fb_fraction, frame=v["f"]["frame"], feature_names=np.array(FEATURE_NAMES))
    save_npz(destination, **payload)
    return payload, dict(path=str(store_path), sha256=sha(store_path), selection="FB-valid pixels in each foot region",
                         base_valid=int(v["valid"].sum()), common_valid=int(eligible.sum()))


def common_view(v, payload):
    valid = payload["valid"].astype(bool)
    return dict(v, valid=valid, runs=runs(valid, v["groups"]))


class InputTransform:
    """Train-only feature map and invertible shrunk-covariance whitening.

Whitening is a full-rank internal parameterization, not dimensionality reduction.
It induces correlated Gaussian emissions in original feature coordinates. This
uses total training-feature covariance, NOT fitted latent residual covariance.
"""
    def __init__(self, embedding_per_region=32, seed=20261117, shrinkage=.5):
        self.kind = "embedding"
        self.size, self.seed, self.shrinkage = embedding_per_region, seed, shrinkage

    def features(self, payload):
        n = len(payload["ptr"])
        result = np.full((n, self.size * 3), np.nan)
        for row in np.flatnonzero(payload["valid"]):
            for r, (a, b) in enumerate(payload["ptr"][row]):
                cloud = payload["points"][a:b].astype(float).copy()
                cloud[:, 2:] = np.sign(cloud[:, 2:]) * np.log1p(abs(cloud[:, 2:]))
                cloud = (cloud - self.pixel_center) / self.pixel_scale
                result[row, r*self.size:(r+1)*self.size] = (np.sqrt(2/self.size) * np.cos(cloud @ self.omega + self.offset)).mean(0)
        return result

    def fit(self, payload, training_rows):
        rows = np.asarray(training_rows, int)
        pieces = [payload["points"][a:b] for row in rows for a, b in payload["ptr"][row]]
        pixels = np.concatenate(pieces).astype(float)
        pixels[:, 2:] = np.sign(pixels[:, 2:]) * np.log1p(abs(pixels[:, 2:]))
        self.pixel_center = np.r_[0., 0., np.median(pixels[:, 2:], axis=0)]
        self.pixel_scale = np.r_[.33, .3, np.maximum(np.ptp(np.quantile(pixels[:, 2:], [.25, .75], axis=0), axis=0), .25)]
        rng = np.random.default_rng(self.seed)
        self.omega, self.offset = rng.normal(size=(4, self.size)), rng.uniform(0, 2*np.pi, self.size)
        features = self.features(payload)
        self.center = np.median(features[rows], axis=0)
        self.scale = np.maximum(np.ptp(np.quantile(features[rows], [.25, .75], axis=0), axis=0), .05)
        standardized = (features[rows] - self.center) / (2 * self.scale)
        covariance = np.atleast_2d(np.cov(standardized, rowvar=False, ddof=0))
        target = np.diag(np.maximum(np.diag(covariance), .05))
        covariance = (1-self.shrinkage) * covariance + self.shrinkage * target
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        self.covariance = covariance
        self.whitener = (eigenvectors * (1 / np.sqrt(np.maximum(eigenvalues, 1e-4)))) @ eigenvectors.T
        return self

    def transform(self, payload):
        return ((self.features(payload) - self.center) / (2*self.scale)) @ self.whitener

    def save(self, path):
        values = {name: value for name, value in self.__dict__.items() if isinstance(value, (np.ndarray, str, int, float))}
        save_npz(path, **values)
