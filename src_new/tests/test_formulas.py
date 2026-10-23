import unittest

import numpy as np

from vla_sbfl.formulas import (
    ALL_FORMULA_IDS,
    FORMULA_REGISTRY,
    LEGACY14_FORMULA_IDS,
    TABLE4_FORMULA_IDS,
    Spectrum,
    evaluate_formula_set,
)


class FormulaRegistryTests(unittest.TestCase):
    def test_catalogue_sizes_and_names(self):
        self.assertEqual(len(TABLE4_FORMULA_IDS), 31)
        self.assertEqual(len(LEGACY14_FORMULA_IDS), 14)
        self.assertEqual(len(ALL_FORMULA_IDS), 42)
        self.assertEqual(len(set(TABLE4_FORMULA_IDS)), 31)
        expected_names = {
            "Braun-Banquet", "Dennis", "Mountford", "Fossum", "Pearson",
            "Gower", "Michael", "Pierce", "Baroni-Urbani & Buser", "Tarwid",
            "Ample", "Phi (Geometric Mean)", "Arithmetic Mean", "Cohen",
            "Fleiss", "Zoltar", "Harmonic Mean", "Rogot2", "Simple Matching",
            "Rogers & Tanimoto", "Hamming", "Hamann", "Sokal", "Scott",
            "Rogot1", "Kulczynski", "Anderberg", "Dice", "Goodman", "Jaccard",
            "Sorensen-Dice",
        }
        self.assertEqual(
            {FORMULA_REGISTRY[item].name for item in TABLE4_FORMULA_IDS},
            expected_names,
        )

    def test_selected_table4_values(self):
        # N_CF=3, N_CS=2, N_UF=1, N_US=4, n=10.
        spectrum = Spectrum(
            ef=np.asarray([3]), ep=np.asarray([2]),
            nf=np.asarray([1]), npass=np.asarray([4]),
        )
        values = evaluate_formula_set(TABLE4_FORMULA_IDS, spectrum)
        self.assertAlmostEqual(values["jaccard"].scores[0], 3/6)
        self.assertAlmostEqual(values["dice"].scores[0], 6/6)
        self.assertAlmostEqual(values["sorensen_dice"].scores[0], 6/9)
        self.assertAlmostEqual(values["simple_matching"].scores[0], 7/10)
        self.assertAlmostEqual(values["hamming"].scores[0], 7)
        self.assertAlmostEqual(values["ample"].scores[0], abs(3/4-2/6))
        self.assertAlmostEqual(
            values["zoltar"].scores[0],
            3/(3+1+2+(10000*1*2/3)),
        )

    def test_every_formula_produces_finite_output_for_edge_spectra(self):
        spectrum = Spectrum(
            ef=np.asarray([2, 0, 2, 1]),
            ep=np.asarray([0, 3, 3, 1]),
            nf=np.asarray([0, 2, 0, 1]),
            npass=np.asarray([3, 0, 0, 2]),
        )
        values = evaluate_formula_set(ALL_FORMULA_IDS, spectrum)
        self.assertEqual(set(values), set(ALL_FORMULA_IDS))
        for formula_id, result in values.items():
            self.assertTrue(np.isfinite(result.scores).all(), formula_id)
            self.assertEqual(result.scores.shape, (4,))

    def test_undefined_and_infinite_cases_are_distinguished(self):
        universal = Spectrum(
            ef=np.asarray([2]), ep=np.asarray([3]),
            nf=np.asarray([0]), npass=np.asarray([0]),
        )
        pearson = evaluate_formula_set(("pearson",), universal)["pearson"]
        self.assertEqual(pearson.undefined_count, 1)
        self.assertEqual(pearson.statuses[0], "undefined_neutral")
        self.assertEqual(pearson.scores[0], 0.0)

        ideal = Spectrum(
            ef=np.asarray([2]), ep=np.asarray([0]),
            nf=np.asarray([0]), npass=np.asarray([3]),
        )
        dstar = evaluate_formula_set(("dstar2",), ideal)["dstar2"]
        self.assertEqual(dstar.positive_infinity_count, 1)
        self.assertEqual(dstar.statuses[0], "positive_infinity")
        self.assertTrue(np.isfinite(dstar.scores[0]))


if __name__ == "__main__":
    unittest.main()

