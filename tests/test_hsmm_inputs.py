from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.stats import multivariate_normal, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"data/step6_fixed_foot_events/code"))
from cyclic_hsmm import CyclicModel, fit_and_decode, flow_evidence, make_runs
from hsmm_inputs import regional_statistics, InputTransform


def payload_fixture():
    rng = np.random.default_rng(31)
    ptr = np.zeros((18, 3, 2), int)
    points, offset, raw = [], 0, []
    for t in range(18):
        features = []
        for r in range(3):
            cloud = rng.normal(size=(20, 4))
            cloud[:, 0] = r*.33+cloud[:, 0]*.1
            cloud[:, 1] *= .1
            cloud[:, 2:] += 4*(t//3 % 2)
            ptr[t, r] = [offset, offset+len(cloud)]
            offset += len(cloud)
            points.append(cloud)
            features.extend(regional_statistics(cloud))
        raw.append(features)
    return dict(points=np.concatenate(points), ptr=ptr, raw42=np.array(raw), valid=np.ones(18, bool))


class InputTests(unittest.TestCase):
    def test_statistics_known_motion_and_empty_subsets(self):
        cloud = np.column_stack([np.arange(10)/10, np.zeros(10), np.arange(10), np.zeros(10)])
        result = regional_statistics(cloud)
        np.testing.assert_allclose(result[:5], np.percentile(np.arange(10), [10,25,50,75,90]))
        self.assertAlmostEqual(result[9], .1)
        self.assertAlmostEqual(result[10], .2)
        self.assertAlmostEqual(result[11], 1.)
        self.assertAlmostEqual(result[12], 0.)
        self.assertAlmostEqual(result[13], -.4)
        static = cloud.copy()
        static[:, 2:] = 0
        s = regional_statistics(static)
        np.testing.assert_array_equal(s[9:13], [1,1,0,0])
        self.assertEqual(s[13], 0)
        fast = cloud.copy()
        fast[:, 2] = 10
        self.assertEqual(regional_statistics(fast)[13], 0)

    def test_transforms_never_fit_heldout_pixels_or_features(self):
        first = payload_fixture()
        second = {k: v.copy() for k,v in first.items()}
        second["raw42"][12:] += 999
        begin = second["ptr"][12,0,0]
        second["points"][begin:] += 999
        a = InputTransform().fit(first, np.arange(12))
        b = InputTransform().fit(second, np.arange(12))
        for field in ("center", "scale", "covariance", "whitener"):
            np.testing.assert_allclose(getattr(a,field), getattr(b,field))
        np.testing.assert_allclose(a.transform(first)[:12], b.transform(second)[:12])

    def test_joint_density_matches_whitened_product_with_jacobian(self):
        transform = InputTransform().fit(payload_fixture(), np.arange(12))
        w = transform.whitener
        self.assertEqual(np.linalg.matrix_rank(w), 96)
        inverse = np.linalg.inv(w)
        x = np.linspace(-.5,.5,96)
        product = norm.logpdf(x@w).sum() + np.linalg.slogdet(w)[1]
        joint = multivariate_normal.logpdf(x, mean=np.zeros(96), cov=inverse.T@inverse)
        self.assertAlmostEqual(product, joint, places=8)

    def test_cyclic_training_and_posterior(self):
        payload = payload_fixture()
        base = dict(runs=[np.arange(18)],
                    f={"t_real_s_v2":np.arange(19)*.04})
        transform = InputTransform().fit(payload, np.arange(18))
        cues = flow_evidence(payload["raw42"], payload["valid"])
        model = CyclicModel(dimensions=96, seed=31)
        decoded = list(fit_and_decode(model, make_runs(base, transform.transform(payload), cues),
                                      iterations=2, progress=lambda *a, **k: None))
        self.assertEqual(len(decoded), 1)
        self.assertEqual(model.weight.shape[1], 96)
        _, path, occupancy, (ic, to) = decoded[0]
        for before, after in zip(path, path[1:]):
            self.assertEqual(after[2], (before[2]+1) % 4)
        np.testing.assert_allclose(occupancy.sum(1), 1, atol=1e-8)
        self.assertTrue(np.all((ic+to) <= 1+1e-8))


if __name__ == "__main__":
    unittest.main()
