"""Prespecified engineering thresholds, including incomplete-data refusal."""
import importlib
from pathlib import Path
import unittest


class SelectionRuleChecks(unittest.TestCase):
    def test_thresholds_and_stability_cannot_be_relaxed_for_a_favorable_aggregate(self):
        self.assertTrue(Path(__file__).with_name('summarize_hg_light.py').is_file())
        choose=importlib.import_module('summarize_hg_light').select_implementation
        cases=[
            (1.20,[1.2,1.2,1.2,1.2,.95],[1.15,1.20,1.25],True,'recommend_G'),
            (1.20,[1.2,1.2,1.2,1.2,.89],[1.15,1.20,1.25],True,'retain_H_insufficient_or_mixed'),
            (1.12,[1.2,1.2,1.2,1.2,.95],[1.09,1.12,1.15],True,'retain_H_insufficient_or_mixed'),
            (.89,[.9,.9,.9,.9,1.05],[.88,.89,.90],True,'recommend_H_engineering_advantage'),
            (.89,[.9,.9,.9,.9,1.05],[.88,.89,.92],True,'retain_H_insufficient_or_mixed'),
            (1.30,[1.3]*5,[1.3]*3,False,'blocked'),
            (1.30,[1.3]*4,[1.3]*3,True,'blocked')]
        for R,medians,repeats,gate,want in cases:
            with self.subTest(R=R,medians=medians,repeats=repeats,gate=gate):
                self.assertEqual(choose(R,medians,repeats,gate)['decision'],want)


if __name__=='__main__':unittest.main(verbosity=2)
