from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"data/step6_fixed_foot_events/code"))
from cyclic_hsmm import CyclicModel, EVENT_PAIRS, fit_and_decode, flow_evidence, make_runs
from hsmm_inputs import regional_statistics
from pixel_emission import PixelEmission, pixel_moments, with_elapsed_time


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

    def test_pixel_scores_equal_explicit_mean_of_all_log_densities(self):
        payload = payload_fixture()
        moments = pixel_moments(payload)
        length = 4
        edges = np.arange(length+1)*.02
        run = dict(moments=moments[:length], edges=edges, observed=np.ones(length, bool))
        training = [with_elapsed_time(moments[a:a+length], edges[:-1]).sum(0) for a in (0, 4)]
        emission = PixelEmission()
        emission.fit(np.stack(training))
        score = emission.scores(run, np.zeros((length, length+1, 2)))
        missing = dict(run, observed=np.array([True, True, False, True]))
        missing_score = emission.scores(missing, np.zeros((length, length+1, 2)))
        np.testing.assert_array_equal(missing_score[2, 1], 0.)
        for phase in range(2):
            expected = 0.
            observed_total = 0.
            for t in range(1, length):
                for r, (first, stop) in enumerate(payload["ptr"][t]):
                    pixels = payload["points"][first:stop]
                    design = np.column_stack(
                        (np.ones(len(pixels)), pixels[:, :2], np.full(len(pixels), edges[t]-edges[1]))
                    )
                    residual = pixels[:, 2:] - design @ emission.coefficients[phase, r]
                    log_density = multivariate_normal.logpdf(residual, cov=emission.covariance[phase, r]).mean()
                    expected += log_density
                    if missing["observed"][t]:
                        observed_total += log_density
            self.assertAlmostEqual(score[1, length-1, phase], expected, places=8)
            self.assertAlmostEqual(missing_score[1, length-1, phase], observed_total, places=8)

    def test_cyclic_training_and_posterior(self):
        payload = payload_fixture()
        base = dict(runs=[np.arange(18)], valid=payload['valid'], groups=[],
                    f={"t_real_s_v2":np.arange(19)*.04})
        cues = flow_evidence(payload["raw42"], payload["valid"])
        model = CyclicModel()
        decoded = list(fit_and_decode(model, make_runs(base, pixel_moments(payload), cues),
                                      iterations=2, progress=lambda *a, **k: None))
        self.assertEqual(len(decoded), 1)
        self.assertEqual(model.emission.coefficients.shape, (4, 3, 4, 2))
        self.assertTrue((np.linalg.eigvalsh(model.emission.covariance) > 0).all())
        _, path, marginal = decoded[0]
        occupancy = marginal.occupancy()
        joint = marginal.transitions_by_frame()
        ic, to = (joint[:, a, b] for a, b in EVENT_PAIRS)
        for before, after in zip(path, path[1:]):
            self.assertEqual(after[2], (before[2]+1) % 4)
        np.testing.assert_allclose(occupancy.sum(1), 1, atol=1e-8)
        self.assertTrue(np.all((ic+to) <= 1+1e-8))

    def test_hsmm_infers_hidden_transition_inside_missing_flow(self):
        model = CyclicModel()
        model.fit_durations(np.ones((2, model.k)))
        # Upstream cues can constrain the endpoints while the missing frames have neutral cues.
        cues = np.zeros((5, model.k))
        cues[[0, 4], [2, 0]] = 50
        observed = np.array([True, False, False, False, True])
        run = dict(moments=np.empty(5), cue=cues, observed=observed)
        duration = model.duration_scores(5)
        scores = duration + model.emissions_with_cues(run, initialize=True)[:, :duration.shape[1]]
        from hsmm import Posterior
        marginal = Posterior(scores, model)
        np.testing.assert_allclose(marginal.occupancy().sum(1), 1.)
        transition = marginal.transitions_by_frame()
        self.assertGreater(transition[1:4, 2, 3].sum(), .99)
        self.assertEqual(model.duration_scores(5).shape, (5, 3, model.k))

    def test_duration_fit_adapts_to_phase_lengths_in_each_video(self):
        for flat in (7, 16):
            states = np.tile(np.repeat(np.arange(4), [3, flat, 3, 8]), 4)
            n = len(states)
            template = pixel_moments(payload_fixture())
            moments = template[np.arange(n) % len(template)]
            run = dict(moments=moments, cue=30*np.eye(4)[states], edges=np.arange(n+1)*.02,
                       observed=np.ones(n, bool))
            model = CyclicModel()
            _, path, _ = next(fit_and_decode(model, [run], iterations=2, progress=lambda *a, **k: None))
            np.testing.assert_allclose(model.duration_probability.sum(0), 1.)
            self.assertEqual(np.argmax(model.duration_probability[:, 1])+1, flat)
            self.assertTrue(np.isfinite(model.duration_scores(n)[0, n]).all())
            self.assertTrue(any(state == 1 and b-a == flat for a, b, state in path))


if __name__ == "__main__":
    unittest.main()
