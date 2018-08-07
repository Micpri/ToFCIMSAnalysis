import unittest
from ToFCIMSAnalysis.mixins.chemical_formulas import chemical_formulas as cf
import collections
import numpy as np
import pandas as pd

class TestClass(unittest.TestCase, cf):

    def setUp(self):

        cf.__init__(self)
        self.formulas = ["CHO","C6H6O6","C6H6O6","C5H4O3","C3H2O8","C7H9O6","C4H4O4","C2H5O5","C4H4NO4","C2H5N2O5"]
        self.list_of_lists = [[1],[2],[3]]
        self.C6H6O6 = {"C":6, "H":6, "O":6}
        self.C6H6O6N = {"C":6, "H":6, "O":6, "N":1}
        self.C6H6O6N0 = {"C":6, "H":6, "O":6, "N":0}
        self.IdotC6H6O6 = {"C":6, "H":6, "O":6, "I":1}
        self.IdotC6H6O6N = {"C":6, "H":6, "O":6, "N":1, "I":1}
        self.IdotC6H6O6N0 = {"C":6, "H":6, "O":6, "N":0, "I":1}
        self.frequency_dist = pd.DataFrame({
                                "C" : [0,1,2,1,2,1,2,1,0,0],
                                "H" : [0,1,1,0,3,2,2,0,0,1],
                                "O" : [0,1,0,1,2,2,3,0,1,0],
                                "N" : [8,1,1,0,0,0,0,0,0,0]
                            })
    def tearDown(self):
        pass

    def test_Flatten_F(self):
       
        self.assertIsInstance(self.list_of_lists[0], list)

    def test_Flatten_T(self):

        self.assertIsInstance(self._Flatten(self.list_of_lists)[0], int)

    def test__Remove_substring(self):

        self.assertEqual(self._Remove_substring("I.C6H6O6-",["-"]), "I.C6H6O6")

    def test_Count_elements_C6H6O6(self):

        C6H6O6 = self.Count_elements("C6H6O6")
        self.assertDictEqual(C6H6O6, self.C6H6O6)

    def test_Count_elements_C6H6O6N(self):

        C6H6O6N = self.Count_elements("C6H6O6N")
        self.assertDictEqual(C6H6O6N, self.C6H6O6N)

    def test_Count_elements_C6H6O6N0(self):

        C6H6O6N0 = self.Count_elements("C6H6O6N0")
        self.assertDictEqual(C6H6O6N0, self.C6H6O6)

    def test_Count_elements_IdotC6H6O6(self):

        IdotC6H6O6 = self.Count_elements("I.C6H6O6")
        self.assertDictEqual(IdotC6H6O6, self.IdotC6H6O6)

    def test_Count_elements_IdotC6H6O6N(self):

        IdotC6H6O6N = self.Count_elements("I.C6H6O6N")
        self.assertDictEqual(IdotC6H6O6N, self.IdotC6H6O6N)

    def test_Count_elements_IdotC6H6O6N0(self):

        IdotC6H6O6N0 = self.Count_elements("I.C6H6O6N0")
        self.assertDictEqual(IdotC6H6O6N0, self.IdotC6H6O6)

    def test_Counted_elements_to_formula_C6H6O6(self):

        ordered_dict = collections.OrderedDict(self.C6H6O6)
        answer = self.Counted_elements_to_formula(ordered_dict)
        self.assertEqual(answer, "C6H6O6")
 
    def test_Counted_elements_to_formula_C6H6O6N(self):

        ordered_dict = collections.OrderedDict(self.C6H6O6N)
        answer = self.Counted_elements_to_formula(ordered_dict)
        self.assertEqual(answer, "C6H6NO6")

    def test_Counted_elements_to_formula_C6H6O6N0(self):

        ordered_dict = collections.OrderedDict(self.C6H6O6N0)
        answer = self.Counted_elements_to_formula(ordered_dict)
        self.assertEqual(answer, "C6H6O6")

    def test_Counted_elements_to_formula_IdotC6H6O6(self):

        ordered_dict = collections.OrderedDict(self.IdotC6H6O6)
        answer = self.Counted_elements_to_formula(ordered_dict)
        self.assertEqual(answer, "C6H6IO6")

    def test_Counted_elements_to_formula_IdotC6H6O6N(self):

        ordered_dict = collections.OrderedDict(self.IdotC6H6O6N)
        answer = self.Counted_elements_to_formula(ordered_dict)
        self.assertEqual(answer, "C6H6INO6")

    def test_Counted_elements_to_formula_IdotC6H6O6N0(self):

        ordered_dict = collections.OrderedDict(self.IdotC6H6O6N0)
        answer = self.Counted_elements_to_formula(ordered_dict)
        self.assertEqual(answer, "C6H6IO6")

    def test_Remove_reagent_ion_I(self):

        self.assertEqual(self.Remove_reagent_ion("I.C6H6O6"), "C6H6O6")

    def test_Remove_reagent_ion_NO3(self):

        self.assertEqual(self.Remove_reagent_ion("NO3.C6H6O6"), "C6H6O6")

    def test_Add_reagent_ion_I(self):

        self.assertEqual(self.Add_reagent_ion("C6H6O6", "I"), "I.C6H6O6")
 
    def test_Add_reagent_ion_NO3(self):

        self.assertEqual(self.Add_reagent_ion("C6H6O6", "NO3"), "NO3.C6H6O6")

    def test_Mass_I(self):

        self.assertEqual(round(self.Mass("I"), 6), 126.904477)

    def test_Mass_CH2O2(self):

        self.assertEqual(round(self.Mass("CH2O2"), 6), 46.005480)

    def test_Mass_defect_12C(self):

        assert self.Mass_defect(12.0000) == 0.0
        self.assertIsInstance(self.Mass_defect(12.0000), float) 

    def test_Mass_defect_16O2(self):

        assert self.Mass_defect(2*15.994915) == 2*0.005085
        self.assertIsInstance(self.Mass_defect(2*15.994915), float) 

    def test_Kendrick_mass_defect_CH2(self):

        answer = self.Kendrick_mass_defect(2*14.015650, 14.015650, 20.0)
        self.assertEqual(answer[0], 0.0)

    def test_Kendrick_mass_defect_CH2_e(self):

        answer = self.Kendrick_mass_defect(14.015650, 2*14.015650, 20.0)
        self.assertEqual(answer[1], 0.00028)

    def test_Element_ratios_C12H6O6N6(self):

        OtoC, HtoC, NtoC = self.Element_ratios("C12H6O6N6")
        self.assertEqual(OtoC, 0.5)
        self.assertEqual(HtoC, 0.5)
        self.assertEqual(NtoC, 0.5)

    def test_Element_ratios_missing_N(self):

        OtoC, HtoC, NtoC = self.Element_ratios("C6H6O6")
        self.assertTrue(np.isnan(NtoC))

    def test_Is_hydrocarbon_C6H6O6(self):

        self.assertFalse(self.Is_hydrocarbon("C6H6O6"))

    def test_Is_hydrocarbon_C6H6O6(self):

        self.assertTrue(self.Is_hydrocarbon("C6H6"))

    def test_Osc(self):

        Osc = self.Osc(0.5, 0.5, 0.5)
        self.assertEqual(Osc, -2.0)

    def test_Osc_noN(self):

        Osc = self.Osc(0.5, 0.5)
        self.assertEqual(Osc, 0.5)

    def test_element_frequency(self):

        answer = self.element_frequency(self.formulas, ["C","H","O","N"])
        self.assertDictEqual(answer.to_dict(), self.frequency_dist.to_dict())