from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"data/step6_fixed_foot_events/code"))
sys.path.insert(0, str(ROOT/"scripts"))
from cyclic_hsmm import boundary_interval, make_runs, phase_interval
from evaluate_cyclic_overground import evaluate_ranges
from event_data import runs
from hsmm import Posterior, posterior, segment
from mask_overlay import decode_crop, foreground_bounds


class EventTests(unittest.TestCase):
    def test_runs_split_at_invalid_observations_and_passage_edges(self):
        valid = np.array([True, True, False, True, True, True, True])
        passage = np.array([1, 1, 1, 1, 2, 2, 2])
        self.assertEqual([ids.tolist() for ids in runs(valid, [passage])], [[0, 1], [3], [4, 5, 6]])

    def test_rle_crop_and_bounds_match_full_mask(self):
        for seed in range(5):
            mask=np.random.default_rng(seed).random((9,12))>.6
            mask[0,0]=True
            flat=mask.ravel()
            changes=np.flatnonzero(flat[1:]!=flat[:-1])+1
            lengths=np.diff(np.r_[0,changes,len(flat)])
            lengths=np.r_[0,lengths] if flat[0] else lengths
            np.testing.assert_array_equal(decode_crop(lengths,12,9,(2,3,10,8)),mask[3:8,2:10])
            x0,y0,x1,y1=foreground_bounds(lengths,12,9)
            ys,xs=np.nonzero(mask)
            self.assertTrue(x0<=xs.min() and y0<=ys.min() and x1>xs.max() and y1>ys.max())
        with self.assertRaises(ValueError):
            foreground_bounds(np.array([107]),12,9)

    def test_viterbi_and_posterior_against_all_segmentations(self):
        rng=np.random.default_rng(42)
        n,classes,limit=5,2,3
        scores=np.full((n,limit+1,classes),-np.inf)
        for a in range(n):
            for d in range(1,min(limit,n-a)+1):
                scores[a,d]=rng.normal(size=classes)
        model=SimpleNamespace()
        model.log_trans=np.log([[.7,.3],[.4,.6]])
        model.log_bos=np.log([.6,.4])
        model.phase_labels=np.array([0,1])
        paths=[]
        def walk(a,path,score):
            if a==n:
                paths.append((score,path))
                return
            for d in range(1,min(limit,n-a)+1):
                for state in range(classes):
                    transition=model.log_trans[path[-1][2],state] if path else model.log_bos[state]
                    walk(a+d,path+[(a,a+d,state)],score+transition+scores[a,d,state])
        walk(0,[],0.)
        expected_score,expected=max(paths,key=lambda item:item[0])
        actual,score=segment(scores,model)
        self.assertEqual(actual,expected)
        self.assertAlmostEqual(score,expected_score)
        z=logsumexp([s for s,_ in paths])
        actual=posterior(scores,model)
        occ,boundary,ic,to=np.zeros((n,classes)),np.zeros(n),np.zeros(n),np.zeros(n)
        first_ic = np.zeros(n-1)
        complete_counts = np.zeros((limit, classes))
        starts, ends = np.zeros(n), np.zeros(n)
        anchor, target_state = 2, 1
        for score,path in paths:
            probability=np.exp(score-z)
            seen_ic = False
            for j,(a,b,state) in enumerate(path):
                occ[a:b,state]+=probability
                if a <= anchor < b and state == target_state:
                    starts[a] += probability
                    ends[b-1] += probability
                if a > 0 and b < n:
                    complete_counts[b-a-1, state] += probability
                if a:
                    boundary[a]+=probability
                    if (path[j-1][2],state)==(0,1):
                        ic[a]+=probability
                        if not seen_ic:
                            first_ic[a-1] += probability
                            seen_ic = True
                    if (path[j-1][2],state)==(1,0): to[a]+=probability
        for a,b in zip(actual,(occ,boundary,ic,to)):
            np.testing.assert_allclose(a,b,atol=1e-10)
        marginal = Posterior(scores, model)
        np.testing.assert_allclose(marginal.first_event_weights((0, 1), 1, n), first_ic, atol=1e-10)
        for actual, expected in zip(marginal.phase_edge_weights(anchor, target_state), (starts, ends)):
            np.testing.assert_allclose(actual, expected, atol=1e-10)
        lo, hi, mass = phase_interval(marginal, 2, 3, target_state, np.arange(100, 100+n))
        self.assertLessEqual(lo, 100+anchor)
        self.assertGreaterEqual(hi, 100+anchor)
        self.assertAlmostEqual(mass, occ[anchor, target_state])
        counts = np.zeros_like(complete_counts)
        marginal.count_complete_durations(counts)
        np.testing.assert_allclose(counts, complete_counts, atol=1e-10)

    def test_event_interval_normalizes_its_local_mass(self):
        weights = np.array([.4, .1, .3, .1, 0.])
        start, end, mass = boundary_interval(weights, np.arange(101, 106))
        self.assertEqual((start, end), (101, 104))
        self.assertAlmostEqual(mass, .9)

    def test_missing_flow_timeline_respects_hard_boundaries(self):
        valid = np.array([False, True, False, True, False, True, False, True, False, False])
        groups = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                  np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])]
        base = dict(valid=valid, groups=groups, runs=runs(valid, groups),
                    f={"t_real_s_v2": np.arange(10)*.02})
        timeline = make_runs(base, np.empty(10), np.zeros((10, 4)), include_missing=True)
        self.assertEqual([run["ids"].tolist() for run in timeline], [[1, 2, 3], [5, 6, 7]])
        self.assertEqual([run["observed"].tolist() for run in timeline], [[True, False, True]]*2)

    def test_scoring_with_no_predicted_intervals(self):
        events = pd.DataFrame(columns=["branch", "trial", "event", "interval_frames"])
        truth = pd.DataFrame([dict(trial=1, event="IC", time_s=.2, nearest_frame=10)])
        metrics, annotated = evaluate_ranges(events, truth, ("pixel_set",))
        self.assertTrue(annotated.empty)
        self.assertEqual(metrics["contained"].tolist(), [0, 0])
        self.assertEqual(metrics["references"].tolist(), [1, 0])


if __name__=="__main__":
    unittest.main()
