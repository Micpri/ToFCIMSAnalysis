import unittest
from ToFCIMSAnalysis.peaklist.kmd import KMD
import collections
import numpy as np
import pandas as pd
import os
import time
import json

class TestClass(unittest.TestCase, KMD):

	def setUp(self):

		KMD.__init__(self, path=".\\tests\\test_data\\",fname="test_peaklist.txt")
		
		self.MD = [
			-0.041865,
			-0.057515,
			-0.073165,
			-0.044939,
			-0.060589,
			-0.076239,
			-0.048013,
			-0.063663,
			-0.079313,
			-0.03678,
			-0.05243,
			-0.06808,
			-0.039854,
			-0.055504,
			-0.071154,
			-0.042928,
			-0.058578,
			-0.074228,
			-0.031695,
			-0.047345,
			-0.062995,
			-0.034769,
			-0.050419,
			-0.066069,
			-0.037843,
			-0.053493,
			-0.069143
		]

		self.KMD_CH2 = [
			0.00954576,
			0.00954576,
			0.00954576,
			0.02210772,
			0.02210772,
			0.02210772,
			0.03466967,
			0.03466967,
			0.03466967,
			0.03249082,
			0.03249082,
			0.03249082,
			0.04505278,
			0.04505278,
			0.04505278,
			0.05761474,
			0.05761474,
			0.05761474,
			0.05543589,
			0.05543589,
			0.05543589,
			0.06799785,
			0.06799785,
			0.06799785,
			0.0805598,
			0.0805598,
			0.0805598,
		]

		self.e_KMD_CH2 = [
			0.00091981,
			0.00119981,
			0.00147981,
			0.00119956,
			0.00147956,
			0.00175956,
			0.00147931,
			0.00175931,
			0.00203931,
			0.00123935,
			0.00151935,
			0.00179935,
			0.0015191,
			0.0017991,
			0.0020791,
			0.00179885,
			0.00207885,
			0.00235885,
			0.00155889,
			0.00183889,
			0.00211889,
			0.00183864,
			0.00211864,
			0.00239864,
			0.00211839,
			0.00239839,
			0.00267839
		]

		self.KMD_CH2_matches = [
			'C2H6O, C3H8O, m131',
			'C2H6O, C3H8O, m131',
			'C2H6O, C3H8O, m131',
			'C2H6ON, m122, m132',
			'C2H6ON, m122, m132',
			'C2H6ON, m122, m132',
			'm113, m123, m133',
			'm113, m123, m133',
			'm113, m123, m133',
			'm211, m221, m231',
			'm211, m221, m231',
			'm211, m221, m231',
			'C2H6O2N, m222, m232',
			'C2H6O2N, m222, m232',
			'C2H6O2N, m222, m232',
			'm213, m223, m233',
			'm213, m223, m233',
			'm213, m223, m233',
			'm311, m321, m331',
			'm311, m321, m331',
			'm311, m321, m331',
			'm312, m322, m332',
			'm312, m322, m332',
			'm312, m322, m332',
			'm313, m323, m333',
			'm313, m323, m333',
			'm313, m323, m333'
		]

		self.KMD_O_matches = [
			'C2H6O, m211, m311',
			'C3H8O, m221, m321',
			'm131, m231, m331',
			'C2H6ON, C2H6O2N, m312',
			'm122, m222, m322',
			'm132, m232, m332',
			'm113, m213, m313',
			'm123, m223, m323',
			'm133, m233, m333',
			'C2H6O, m211, m311',
			'C3H8O, m221, m321',
			'm131, m231, m331',
			'C2H6ON, C2H6O2N, m312',
			'm122, m222, m322',
			'm132, m232, m332',
			'm113, m213, m313',
			'm123, m223, m323',
			'm133, m233, m333',
			'C2H6O, m211, m311',
			'C3H8O, m221, m321',
			'm131, m231, m331',
			'C2H6ON, C2H6O2N, m312',
			'm122, m222, m322',
			'm132, m232, m332',
			'm113, m213, m313',
			'm123, m223, m323',
			'm133, m233, m333'
   		]

		with open('.\\tests\\test_data\\test_matched_id_data.json') as f:
			self.matched_id_data = json.load(f)


	def tearDown(self):
		pass

	# def test_ChemSpiderInstance(self):
        
	# 	assert isinstance(self.peaklist, pd.DataFrame)

	def test_selfdotpeaklist_type(self):
        
		self.assertIsInstance(self.peaklist, pd.DataFrame)

	def test_selfdotpeaklist_cols(self):

		self.assertListEqual(list(self.peaklist.columns), ["ion","x0","integer_mass"])

	def test_Kendrick_bases_apart(self):

		answer = self.Kendrick_bases_apart(14.0,28.0,14.0)
		self.assertEqual(answer, -1)

	def test_F_amu_apart(self):

		answer = self.F_amu_apart(14.0,28.0,14.0)
		self.assertTrue(answer)

	def test_F_within_error_type1(self):
		
		answer = self.F_within_error(10, 1, 9, 2)
		self.assertTrue(answer)

	def test_F_within_error_type2(self):
		
		answer = self.F_within_error(10, 1, 10, 1)
		self.assertTrue(answer)

	def test_F_within_error_type3(self):
		
		answer = self.F_within_error(10, 1, 10, 2)
		self.assertTrue(answer)

	def test_F_within_error_type4(self):
		
		answer = self.F_within_error(10, 2, 9, 1)
		self.assertTrue(answer)

	def test_F_within_error_False(self):
		
		answer = self.F_within_error(20, 1, 10, 1)
		self.assertFalse(answer)

	def test_Calculate_mass_defect(self):
		
		self.Calculate_mass_defect()
		self.assertListEqual(list(self.peaklist["MD"]), self.MD)

	def test_Calculate_kendrick_mass_defect_CH2(self):
		
		self.Calculate_kendrick_mass_defect("CH2")
		self.assertListEqual(list(self.peaklist["KMD_CH2"]), self.KMD_CH2)

	def test_Calculate_kendrick_mass_defect_CH2e(self):
		
		self.Calculate_kendrick_mass_defect("CH2")
		self.assertListEqual(list(self.peaklist["e_KMD_CH2"]), self.e_KMD_CH2)

	def test_Match_peaks_on_kmd_CH2(self):
		
		self.Match_peaks_on_kmd("CH2")
		self.assertListEqual(list(self.peaklist["KMD_CH2_matches"].values), self.KMD_CH2_matches)

	def test_Match_peaks_on_kmd_O(self):
		
		self.Match_peaks_on_kmd("O")
		self.assertListEqual(list(self.peaklist["KMD_O_matches"].values), self.KMD_O_matches)

	def test_Find_unknown_formula_2CH2_positive(self):

		answer = self.Find_unknown_formula("C2H4O", self.Mass("C4H8O"), "CH2")
		self.assertEqual(answer, "C4H8O")

	def test_Find_unknown_formula_2O_negative(self):

		answer = self.Find_unknown_formula("C2H4O5", self.Mass("C2H4O3"), "O")
		self.assertEqual(answer, "C2H4O3")

	def test_Output_matched_identities(self):

		# initialise columns in peaklist
		[self.Match_peaks_on_kmd(kb) for kb in ["O","CH2"]]
		# run method to test
		self.Output_matched_identities("m")
		# load list to check against
		with open('.\\matched_id_data.json') as f:
			matched_id_data = json.load(f)
		self.assertDictEqual(self.matched_id_data, matched_id_data)
		os.remove("matched_id_data.json")









	# def test_Is_formula_realistic_True(self):

	# 	assert not self.Is_formula_realistic("CH2O2")

	# def test_Is_formula_realistic_False(self):

	# 	assert not self.Is_formula_realistic("C20H1")
