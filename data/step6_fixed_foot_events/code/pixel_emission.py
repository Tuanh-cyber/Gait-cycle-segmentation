"""Gaussian flow conditioned on pixel position and elapsed phase time."""
import numpy as np

POSITION = (0, 1, 2)
VELOCITY = (3, 4)
DESIGN = (*POSITION, 5)


def pixel_moments(payload):
    """Exact per-region averages of outer products over all valid pixels."""
    result = np.full((*payload["ptr"].shape[:2], 5, 5), np.nan)
    for frame in np.flatnonzero(payload["valid"]):
        for region, (start, stop) in enumerate(payload["ptr"][frame]):
            pixels = payload["points"][start:stop]
            augmented = np.column_stack((np.ones(len(pixels)), pixels))
            result[frame, region] = augmented.T @ augmented / len(pixels)
    return result


def with_elapsed_time(moments, elapsed):
    """Append elapsed time to [1, xi, eta, vx, vy] for each pixel."""
    augmented = np.empty((*moments.shape[:2], 6, 6))
    augmented[..., :5, :5] = moments
    time_products = elapsed[:, None, None] * moments[..., 0, :]
    augmented[..., 5, :5] = time_products
    augmented[..., :5, 5] = time_products
    augmented[..., 5, 5] = elapsed[:, None] ** 2
    return augmented


def regression(moments):
    xx = moments[..., DESIGN, :][..., DESIGN]
    xy = moments[..., DESIGN, :][..., VELOCITY]
    yy = moments[..., VELOCITY, :][..., VELOCITY]
    coefficients = np.linalg.pinv(xx, hermitian=True) @ xy
    residual = yy - xy.swapaxes(-1, -2) @ coefficients
    return coefficients, (residual + residual.swapaxes(-1, -2)) / 2


class PixelEmission:
    def fit(self, moments):
        """Fit phase/region conditional Gaussians with per-region pooled shrinkage."""
        weight = moments[..., 0, 0]
        coefficients, residual = regression(moments)
        pooled_coefficients, pooled_residual = regression(moments.sum(axis=0))
        pooled_covariance = pooled_residual / weight.sum(axis=0)[:, None, None]
        self.coefficients = np.where(weight[..., None, None] > 0, coefficients, pooled_coefficients)
        covariance = (residual + pooled_covariance) / (weight[..., None, None] + 1)
        eigenvalues, directions = np.linalg.eigh(covariance)
        speed_squared = moments[..., VELOCITY, :][..., VELOCITY].sum(axis=0).trace(axis1=-2, axis2=-1)
        scale = np.maximum(1., speed_squared / weight.sum(axis=0))
        eigenvalues = np.maximum(eigenvalues, np.sqrt(np.finfo(float).eps) * scale[:, None])
        self.covariance = (directions * eigenvalues[..., None, :]) @ directions.swapaxes(-1, -2)

    def scores(self, run, scores):
        """Add observed log densities while keeping missing frames in elapsed time."""
        observed = run["observed"]
        moments, edges = run["moments"][observed], run["edges"]
        n, lengths, classes = scores.shape
        base = np.zeros((n, classes))
        linear = np.zeros_like(base)
        quadratic = np.zeros(classes)
        precision = np.linalg.inv(self.covariance)
        log_norm = 2 * np.log(2 * np.pi) + np.linalg.slogdet(self.covariance)[1]
        for phase in range(classes):
            for region, data in enumerate(moments.transpose(1, 0, 2, 3)):
                coefficients = self.coefficients[phase, region]
                residual = np.vstack((-coefficients[:3], np.eye(2)))
                metric = precision[phase, region]
                time_slope = coefficients[3]
                base[observed, phase] += np.einsum("tij,ij->t", data, residual @ metric @ residual.T) + log_norm[phase, region]
                linear[observed, phase] -= 2 * (data[:, 0] @ residual @ metric @ time_slope)
                quadratic[phase] += time_slope @ metric @ time_slope
        accumulated = np.zeros_like(base)
        for offset in range(lengths - 1):
            stop = n - offset
            elapsed = (edges[offset:n] - edges[:stop])[:, None]
            penalty = base[offset:n] + elapsed * linear[offset:n] + elapsed ** 2 * quadratic
            accumulated[:stop] -= .5 * np.where(observed[offset:n, None], penalty, 0.)
            scores[:stop, offset+1] += accumulated[:stop]
        return scores
