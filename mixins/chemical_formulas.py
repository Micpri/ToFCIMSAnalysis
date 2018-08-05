import re
import pyteomics
import collections
import numpy as np
import pandas as pd
from cims_analysis.constants.elements import Elements

class chemical_formulas(Elements):

    """
    Contains methods for querying a chemical formula.
    """ 

    def __init__(self):
        Elements.__init__(self)

    
    def _Flatten(self, list_of_lists):

        """
        Flattens a list of lists into a list.
        + list_of_lists: list.
        """

        return [item for sublist in list_of_lists for item in sublist]


    def _Remove_substring(self, string, list_of_substrings):

        """
        Remove list of substrings from string.
        + list_of_substrings: list of str.
        """

        for substring in list_of_substrings:
            assert isinstance(substring, str), "List should not contain non string elements."
            string = string.replace(substring,"")
        return string


    def Count_elements(self, moiety):

        """
        Returns dict with keys of elements and values of number of elements.
        + moiety: str.
        """

        ls = []
        found = re.findall(r'([A-Z][a-z]*)(\d*)', moiety)
        for pair in found:
            letter = pair[0]
            number = pair[1]
            if number == "0":
                pass
            else:
                if len(number) == 0:
                    number = 1
                ls.append((unicode(pair[0]), int(number)))

        dic = collections.OrderedDict(sorted(dict(ls).items(), key=lambda x: x[0]))
        return dic


    def Counted_elements_to_formula(self, ordered_dict):

        """
        Takes an ordered dictionary of counted elements from 
        formula and returns a string.
        + ordered_dict: collections.OrderedDict().
        """
        assert isinstance(ordered_dict, collections.OrderedDict), "Argument should be of type collections.OrderedDict."
        string = ""
        for key in sorted(ordered_dict):
            string += key
            number = ordered_dict[key]
            if number == 0:
                string = string[:-1]
            elif number == 1:
                pass
            else:
                string += str(number)


        return string


    def Remove_reagent_ion(self, formula):

        """
        Takes a chemical formula string and removes the reagent ion.
        Reagent ion (X) must be succeeded by a . then the formula.
        e.g. I.C6H6O6, NO3.C12H18O6
        + formula: str.
        """

        return formula.split(".")[1]


    def Add_reagent_ion(self, formula, reagent_ion):

        """
        Takes a chemical formula string and removes the reagent ion.
        Reagent ion (X) must be succeeded by a . then the formula.
        e.g. I.C6H6O6, NO3.C12H18O6
        + formula: str.
        """

        return "%s.%s" % (reagent_ion, formula)


    def Mass(self, moiety):

        """
        Calculates exact mass for a given moiety.
        + moiety: str.
        """

        counted_elements = self.Count_elements(moiety)
        exact_mass = 0
        for key, value in counted_elements.iteritems():            
            exact_mass += (self.Get_mass(key) * value)
        return round(exact_mass, 6)


    def Mass_defect(self, exact_mass):

        """
        Calculates the mass defect for a given exact mass.
        + exact_mass: float.
        """

        integer_mass = int(np.round(exact_mass))
        mass_defect = integer_mass - exact_mass
        return round(mass_defect, 8)


    def Kendrick_mass_defect(self, moiety_exact_mass, kendrick_base_exact_mass, peak_assignment_error=20.0):

        """
        Calculates the Kendrick mass defect for a given exact mass and kenrdick base.
        + moiety_exact_mass: float.
        + kendrick_base_exact_mass: float. 
        + peak_assignment_error: float. units. ppm.
        """
        peak_assignment_error /= 1e6

        moiety_mass_defect = self.Mass_defect(moiety_exact_mass)

        kendrick_base_integer_mass = int(np.round(kendrick_base_exact_mass))
        kendrick_normalisation_factor = kendrick_base_integer_mass / kendrick_base_exact_mass
        
        kendrick_exact_mass = moiety_exact_mass * kendrick_normalisation_factor
        kendrick_integer_mass = int(np.round(kendrick_exact_mass))
        kendrick_mass_defect = round(kendrick_integer_mass - kendrick_exact_mass, 8)

        e = moiety_exact_mass * peak_assignment_error
        kendrick_e = round(e * kendrick_normalisation_factor, 8)

        return kendrick_mass_defect, kendrick_e


    def Element_ratios(self, moiety):

        """
        Returns O:C, H:C and N:C of moiety.
        + moiety: str.
        """

        counted_elements = self.Count_elements(moiety)

        try:
            OtoC = float(counted_elements["O"])/counted_elements["C"]
        except KeyError:
            OtoC = np.nan
        try:
            HtoC = float(counted_elements["H"])/counted_elements["C"]
        except KeyError:
            HtoC = np.nan
        try:
            NtoC = float(counted_elements["N"])/counted_elements["C"]
        except KeyError:
            NtoC = np.nan

        return OtoC, HtoC, NtoC


    def Is_hydrocarbon(self, moiety):

        """
        Returns true if moiety is a hydrocarbon.
        + moiety: str.
        """

        moiety = moiety.replace("C","").replace("H","")
        try:
            int(moiety)
            ishydrocarbon = True
        except ValueError:
            ishydrocarbon = False

        return ishydrocarbon


    def Osc(self, OtoC, HtoC, NtoC=0):
        
        """
        Calcuate average carbon oxidation state.
        OSc calculated from Kroll et al., 2011.
        DOI: 10.1038/NCHEM.948. Assumes all N is 
        organic nitrate (not true for NO2s, PANs etc.)
        + OtoC: float.
        + HtoC: float.
        + NtoC: float.
        """

        return np.round((2 * OtoC) - HtoC - (5 * NtoC), 1)


    def element_frequency(self, list_of_formulae, elements):
        
        """
        Calculates frequency of elements in list_of_moieties.
        + list_of_moieties: list of str.
        + elements: list of str.
        """

        # convert all strings into ordered dictionaries to count elements
        counted_moieties = [self.Count_elements(moiety) for moiety in list_of_formulae]

        # all frequency arrays must be of the same length, so identfy the longest.
        # Count element keys and frequency of their occurence (values).
        # frequencies = collections.Counter(_Flatten([x.keys() for x in counted_moieties]))
        size = max(self._Flatten([x.values() for x in counted_moieties]))

        # initialise freq_dist dataframe in which the distribution is stored
        freq_dist = pd.DataFrame(0, index=np.arange(size + 1), columns=elements)

        # for each counted_moiety then for each element from the elements:
        # if the element in present add a count to the freq_dist dataframe 
        # at the row of the counted_moiety dict value and column of the 
        # counted_moiety dict key.
        for counted_moiety in counted_moieties:
            for element in elements:
                if element in counted_moiety.keys():
                    freq_dist.loc[counted_moiety[element], element] +=1
                else:
                    freq_dist.loc[0,element] +=1

        return freq_dist