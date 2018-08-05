# from chemspipy import ChemSpider
# from pyteomics import mass
# from itertools import cycle
# import re
import collections
# import os
import itertools
# import time
# import array
# import pyteomics
# import sys
# import csv
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cims_analysis.mixins.chemical_formulas import chemical_formulas as cf


class KMD(cf):

    def __init__(self, path):

        cf.__init__(self)

        self.path = path
        self.peaklist = pd.read_csv(self.path, sep="\t")[["ion","x0"]]
        self.peaklist["integer_mass"] = [int(np.round(mass)) for mass in self.peaklist["x0"]]
        
        # """
        # self.cs: str, ChemSpider instance
        # self.reagent_ion: str, reagent ion e.g. I
        # self.kendrick_bases: str list, base moeities
        # self.kendrick_base_mass: float list, exact masses of kendrick bases
        # self.ion: str list, names of peaks
        # self.sumFormula: str list, formulas of peaks
        # self.x0: float list, exact masses of formulas
        # self.exact_mass_minus_reagent_ion: copy of x0 minus exact mass of reagent ion
        # self.integer_mass: int list, rounded copy of x0
        # self.integer_mass_minus_reagent_ion: copy of integer_mass minus exact mass of reagent ion
        # self.mass_defect: float list, mass defect of peaks
        # self.mass_defect_minus_reagent_ion : float list, mass defect of peaks (with reagent ion removed)
        # self._KEM: dict, keys=kendrick base, vals=float list, normalised mass to new kendrick base
        # self._KMD: dict, keys=kendrick base, vals=float list, mass defect normalised to new kendrick base
        # self._KMD_error: dict, keys=kendrick base, vals=float list, mass defect error to new kendrick base
        # self.KMD_matches: dict, keys=kendrick base, vals=str list,
        #                   for ith self.ion, ith KMD_matches[kendrick_base] are peaks that meet matching c
        #                   criteria.
        # self.suggested_formulas: list of dicts,
        #                   for ith self.ion, ith suggested_formulas is the suggested formulas
        #                   with a weighting based on how many times it was picked.
        # self.suggested_names: list of dicts,
        #                   for ith self.ion, ith suggested_compounds is the suggested common names
        # self.suggested_errors: list of dicts,
        #                   for ith self.ion, ith suggested_compounds is the suggested error of the assignment
        # """

        # self.cs = ChemSpider(os.environ["CHEM_SECURITY_TOKEN"])
        # self.directory = ""
        # self.fname = ""
        # self.reagent_ion = reagent_ion
        # self.kendrick_bases = kendrick_bases
        # self.kendrick_base_mass = [self._Mass_calculator(mass) for mass in self.kendrick_bases]
        # self.ion = None
        # self.sumFormula = None
        # self.x0 = None
        # self.exact_mass_minus_reagent_ion = None
        # self.integer_mass = None
        # self.integer_mass_minus_reagent_ion = None
        # self.mass_defect = None
        # self.mass_defect_minus_reagent_ion = None
        # self._KEM = {}
        # self._KMD = {}
        # self._KMD_error = {}
        # self.KMD_matches = {}
        # self.suggested_formulas = {}
        # self.suggested_names = {}
        # self.suggested_errors = {}


    def Kendrick_bases_apart(self, amu1, amu2, kendrick_base_amu):

        """
        How many kendrick_base amu1 and amu2 are different from each other.
        + amu1. float.
        + amu2. float.
        + kendrick_base_amu. float.
        """

        ans = (int(np.round(amu1)) - int(np.round(amu2))) / np.round(kendrick_base_amu)
        ans = ans if ans % 1 == 0.0 else None

        return int(ans)

    def F_amu_apart(self, amu1, amu2, kendrick_base_mass):

        """
        Returns True if difference between amu1 and
        amu2 when divided by moiety mass is equal to 0.
        + amu1. numeric.
        + amu2. numeric.
        + kendrick_base_amu. numeric.
        """

        return abs(int(amu1) - int(amu2)) % int(kendrick_base_mass) == 0

    def F_within_error(self, KMD1, KMD1e, KMD2, KMD2e):

        """
        Returns True if KMD1 +/- KMD1e is within KMD2 +/- KMD2e.
        + KMD1. float.
        + KMD1e. float.
        + KMD2. float.
        + KMD2e. float.
        """

        # Get upper and lower limits
        KMD1_ulim = KMD1 + KMD1e
        KMD2_ulim = KMD2 + KMD2e
        KMD1_llim = KMD1 - KMD1e
        KMD2_llim = KMD2 - KMD2e

        # The 4 different cases
        type1 = ((KMD1_llim >= KMD2_llim) and (KMD1_llim <= KMD2_ulim))
        type2 = ((KMD1_ulim >= KMD2_llim) and (KMD1_ulim <= KMD2_ulim))
        type3 = ((KMD1_ulim <= KMD2_ulim) and (KMD1_llim >= KMD2_llim))
        type4 = ((KMD1_llim <= KMD2_llim) and (KMD1_ulim >= KMD2_ulim))

        return type1 or type2 or type3 or type4

    def Calculate_mass_defect(self):

        """
        Calcuates mass defects for all masses in self.peaklist
        """
        self.peaklist["MD"] = [self.Mass_defect(mass) for mass in self.peaklist.x0]

    def Calculate_kendrick_mass_defect(self, kendrick_base):

        """
        Calcuates Kendrick mass defects for all masses in self.peaklist.
        + kendrick_base. str.
        """
        kendrick_base_exact_mass = self.Mass(kendrick_base)

        kmds = []
        ekmds = []
        for exact_mass in self.peaklist["x0"]:
            kmd, ekmd = self.Kendrick_mass_defect(exact_mass, kendrick_base_exact_mass)
            kmds.append(kmd)
            ekmds.append(ekmd)

        self.peaklist["KMD_"+kendrick_base] = kmds
        self.peaklist["e_KMD_"+kendrick_base] = ekmds


    def Match_peaks_on_kmd(self, kendrick_base):

        """
        For each peak in peaklist.ion, the conditions for peaks that match for
        the passed kendrick base are evaluated. Where the conditions are true,
        extract the names of those peaks. Must run self._Calc_kendrick_mass_defect first to ensure
        the relevent columns are there:
        + kendrick_base. str.
        """

        # if kendrick mass defect column isnt present then make it
        if "KMD_"+kendrick_base not in self.peaklist.columns:
            self.Calculate_kendrick_mass_defect(kendrick_base)

        # set some local variables
        match_col_name = "KMD_"+kendrick_base+"_matches"
        df_length = len(self.peaklist['ion'])

        # Get Kendrick mass information
        kendrick_base_exact_mass = self.Mass(kendrick_base)
        kendrick_base_integer_mass = int(np.round(kendrick_base_exact_mass))

        # initalise empty column in peaklist
        self.peaklist[match_col_name] = [None] * len(self.peaklist['ion'])
        
        # make bool matrices where criteria for matching is met
        # each row in the matrix returns a 1 or 0 where the mass
        # in that row agrees with the condition with the mass
        # of the index in the row.
        amuApart = np.zeros(shape=(df_length, df_length))
        for i, amu1 in enumerate(self.peaklist['integer_mass']):
            amuApart[i] = [self.F_amu_apart(amu1, amu2, kendrick_base_integer_mass) for amu2 in self.peaklist['integer_mass']]

        i=0
        withinError = np.zeros(shape=(df_length, df_length))
        for KMD1, e_KMD1 in zip(self.peaklist['KMD_'+kendrick_base], self.peaklist['e_KMD_'+kendrick_base]):
            withinError[i] = [self.F_within_error(KMD1, e_KMD1, KMD2, e_KMD2) for KMD2, e_KMD2 in zip(self.peaklist['KMD_'+kendrick_base], self.peaklist['e_KMD_'+kendrick_base])]
            i += 1

        # combine bool arrays to matching bool array
        # where both criteria are met
        isMatch = amuApart * withinError

        # For each peak in the peaklist extract the mask from isMatch
        # where True, index the ion name (this is the name of the match)
        # put these matches in their own column in the peaklist
        match_column = []
        for i, name in enumerate(self.peaklist['ion']):
            match_mask = list(isMatch[i,:])
            matches = [p for p, s in itertools.izip(list(self.peaklist["ion"]), match_mask) if s]
            match_column.append(str(matches).replace("[","").replace("]","").replace("'",""))
        self.peaklist[match_col_name] = match_column


    def Find_unknown_formula(self, known_formula, unknown_mass, kendrick_base):

        """
        Returns estimated formula for unknown_mass 
        based on known_formula and kendrick_base.
        + known_formula. str.
        + unknown_mass. int.
        + kendrick_base. str.
        """

        unknown_mass = int(round(unknown_mass))

        # get mass and elements for known arguments and collate them.
        known_formula_mass = self.Mass(known_formula)
        formula_elements = self.Count_elements(known_formula)
        kendrick_base_exact_mass = self.Mass(kendrick_base)
        kendrick_base_elements = self.Count_elements(kendrick_base)


        all_elements = formula_elements.copy()
        all_elements.update(kendrick_base_elements)

        kmd_units = self.Kendrick_bases_apart(known_formula_mass,
                                              unknown_mass,
                                              kendrick_base_exact_mass)
        if not kmd_units:
            estimated = "error - Not integer kmd_units away."
        else:
            # new dictionary that multiplies the kendrick_bases
            # elements by how many repeating kmd bases there are
            kendrick_base_update = collections.Counter()
            for el in kendrick_base_elements:
                kendrick_base_update[el] = int(kendrick_base_elements[el] * -kmd_units)

            # update unknown_formula_elements to contain suggested formula
            unknown_formula_elements = collections.Counter()
            unknown_formula_elements.update(kendrick_base_update)
            unknown_formula_elements.update(formula_elements)

            for k,v in unknown_formula_elements.items():
                if v == 0:
                    del unknown_formula_elements[k]    # get rid of any 0 values

            # if the suggested formula have negative subscript it cant be real
            if sum(1 for number in unknown_formula_elements.values() if number < 0) > 0:
                estimated = "error - Can't have negative subscript in formula."
            else:
                estimated = ''.join("%s%r" % (key,val) for (key,val) in unknown_formula_elements.iteritems())
                estimated = self.Counted_elements_to_formula(self.Count_elements(estimated))

        return str(estimated)


#     def _Is_formula_realistic(self, suggested_formula):

#         """
#         Takes a suggested formula and returns true if it 
#         passes the conditions posed here. Takes into account 
#         realistic structure and visiblity by CIMS. returns Bool.
#         This is the function that decides if what the solver 
#         has returned is rubbish or not.
#         """

#         suggested_formula = self._Remove_reagent_ion(suggested_formula)
#         suggested_formula = self._Counted_elements_to_formula(suggested_formula)
#         list_of_compounds = self.cs.simple_search_by_formula(suggested_formula)

#         return_val = 1
#         if len(list_of_compounds) < 1:
#             return_val = 0
#         if self._Is_hydrocarbon(suggested_formula):
#             return_val = 0

#         return return_val


#     def _Matched(self, kendrick_base, known=True):

#         """
#         Returns list of KMD matches for the passed kendrick base
#         containin; known matches i.e. if a '-' is present in the name
#         indicating it has been assigned a formula; or unknown matches
#         if the '-' character is not present
#         """

#         matched = []
#         for matches in self.KMD_matches[kendrick_base]:
#             if known:
#                 matched.append([x for x in matches if "-" in x])
#             else:
#                 matched.append([x for x in matches if not "-" in x])

#         matched = list(set(self._Flatten(matched)))

#         return matched


#     def _Get_KMD_match_suggestions(self):

#         """
#         Works out what the matching peak could be based on the kendrick_base
#         and returns a list of the unknown name, unknown mass and chemspider objects
#         """

#         print "Identifying compounds. This may take some time."

#         # for each kendrick base
#         for kendrick_base in self.kendrick_bases:

#             self.suggested_formulas[kendrick_base] = [[] for x in self.ion]
#             self.suggested_errors[kendrick_base] = [[] for x in self.ion]
#             self.suggested_names[kendrick_base] = [[] for x in self.ion]

#             print "Looking at matches for %s" % kendrick_base

#             # find those KMD matches that are unknown
#             matched_unknowns = self._Matched(kendrick_base, known=False)

#             # for each of those uknowns
#             for i, unknown in enumerate(matched_unknowns):

#                 print "matching %d of %d unknowns" % (i+1, len(matched_unknowns))

#                 # find the KMD matches that are known
#                 matched_knowns = self._Matched(kendrick_base, known=True)

#                 # for each of those knowns
#                 guesses_formulas = []
#                 guesses_names = []
#                 guesses_errors = []

#                 for known in matched_knowns:

#                     unknown_index = self.ion.index(unknown)
#                     unknown_mass = self.x0[unknown_index]

#                     # get the suggested structure of the unknown that the known points to
#                     suggested_formula = self._Calc_unknown_formula(known, unknown_mass, kendrick_base)

#                     # igor if error is returned
#                     if not suggested_formula[0:5] == "error":

#                         # return true if structure is realistic
#                         realistic = self._Is_formula_realistic(suggested_formula)

#                         if not realistic:
#                             pass

#                         # Get formula without reagent ion
#                         suggested_formula_noRI = self._Counted_elements_to_formula(self._Remove_reagent_ion(suggested_formula))

#                         CScompounds = self.cs.simple_search_by_formula(suggested_formula_noRI)
#                         suggested_names = self._Get_visible_compounds(CScompounds, names=True)

#                         if suggested_names == []:
#                             pass
#                         else:

#                             error = self._Error_on_assignment(suggested_formula, unknown_mass)

#                             # put known and suggested pairs into list
#                             guesses_formulas.append((known, suggested_formula))
#                             guesses_errors.append((suggested_formula, error))
#                             guesses_names.append((known, suggested_names))

#                     else:
#                         pass

#                 # put lists of tuples of the known and suggested unknown
#                 # into an object variable at the same index as the unknown
#                 self.suggested_formulas[kendrick_base][unknown_index] = guesses_formulas
#                 self.suggested_errors[kendrick_base][unknown_index] = guesses_errors
#                 self.suggested_names[kendrick_base][unknown_index] = guesses_names

#         print "finished matching"

    # def Error_on_assignment(self, suggested_formula, unknown_mass):

    #     """Provides ppm error on assignment of unknown
    #     peak with sugggested formula.
    #     + suggested_formual. str.
    #     + unknown_mass. float.
    #      """

    #     exact_mass = self.Mass(suggested_formula)
    #     error = 1e6 * ((unknown_mass - exact_mass) / exact_mass)

    #     return error


#     def _Get_visible_compounds(self, list_of_csCompounds, names=False):

#         """Takes a list of cs.Compounds and returns their
#         names if they are visible by CIMS."""

#         ls = []
#         for compound in list_of_csCompounds:
#             try:
#                 condition_met = self._Condition_for_visible(compound)
#                 if condition_met:
#                     if names:
#                         ls.append(compound.common_name)
#                     else:
#                         ls.append(compound)
#                     if len(ls) > 4:
#                         break
#                 else:
#                     ls.append("Structure not visible by %s CIMS" % (self.reagent_ion))
#             except KeyError:
#                 ls.append("No Common Name")

#         return ls


#     def _Condition_for_visible(self, compound):

#         """
#         Condition for whether the suggested molecule is visible by CIMS.
#         + Can't be dimer.
#         + Can't have overall + or - charge.
#         + Cant have partial charge.
#         + No wierd character in name that is odd encoding of a greek letter
#             (indicates non typical oxidation state)
#         """

#         neg_count = 0
#         pos_count = 0
#         for char in compound.smiles:
#             if char == "-": neg_count += 1
#             if char == "+": pos_count += 1

#         return ("." not in compound.smiles) and (neg_count == pos_count) and ("$" not in compound.common_name) and ("{" not in compound.common_name) and ("^" not in compound.common_name)


    # def _Weighted_guesses(self, known_formula, suggested_formula):

    #     """
    #     Takes a list of tuple pairs where 0th element of the tuple
    #     is the suggesting known compound and the 1st element of the
    #     tuple is the unknown compound suggestion. The elements in the
    #     passed list represent how many times a suggestion is made.
    #     returns a dictionary of suggestion keys and frequency vals.
    #     """

    #     keys = set([x[1] for x in list_of_tuple_pairs])
    #     string_list = [x[1] for x in list_of_tuple_pairs]

    #     dat = {}
    #     for key in keys:
    #         dat[key] = string_list.count(key)

    #     return dat


    # def New_peaklist(self):

    #     """Return new peaklist with updated assignments.

    #     """

    #     new_peaklist = pd.DataFrame()

    #     for kendrick_base in self.suggested_formulas:

    #         suggested_formulas = self.suggested_formulas[kendrick_base]

    #         for i, entry in enumerate(suggested_formulas):

    #             # find highest frequency of a suggestion
    #             weighted_guess = self._Weighted_guesses(entry)
    #             try:
    #                 most_weight = max(weighted_guess, key=weighted_guess.get)
    #                 new_x0 = self.Mass(most_weight)

    #                 new_peaklist.loc[i, ['ion']] = most_weight+"-"
    #                 new_peaklist.loc[i, ['x0']] = new_x0

    #             except ValueError:
    #                 pass # weighted_guess returns {}

    #     return new_peaklist


#     def Assignment_info(self):

#         """Collect estimated assignment info into dataframe"""

#         assignment_info = pd.DataFrame(columns = ["ion", "kendrick base", "suggested by", "suggested formula", "error / ppm", "names"])

#         i = 0
#         for kendrick_base in self.kendrick_bases:

#             matched_unknowns = sorted(self._Matched(kendrick_base, known=False))

#             for matched_unknown in matched_unknowns:

#                 unknown_index = self.ion.index(matched_unknown)
#                 formulas = self.suggested_formulas[kendrick_base][unknown_index]
#                 errors = self.suggested_errors[kendrick_base][unknown_index]
#                 names = self.suggested_names[kendrick_base][unknown_index]

#                 for formula, error, name in zip(formulas, errors, names):

#                     assignment_info.loc[i] = [matched_unknown, kendrick_base, formula[0].encode('utf-8'), formula[1].encode('utf-8'), round(error[1],4), str([x.encode('utf-8') for x in name[1]]).replace("'","").replace("[","").replace("]","")]
#                     i +=1

#         assignment_info = assignment_info.sort_values(by=["ion","kendrick base"])

#         return assignment_info


#     def Run(self):

#         # Calculate mass defect
#         self._Calc_mass_defect(self._Mass_calculator(self.reagent_ion))

#         # # Calculate kendrick mass defects
#         [self._Calc_kendrick_mass_defect(base, self.kendrick_base_mass[i]) for i, base in enumerate(self.kendrick_bases)]

#         # Stage 1. Match kendrick mass defects - still unknown
#         [self._Match_peaks_on_kmd(base) for base in self.kendrick_bases]

#         # Stage2. Get suggested formulas and compounds for matches
# self._Get_KMD_match_suggestions()