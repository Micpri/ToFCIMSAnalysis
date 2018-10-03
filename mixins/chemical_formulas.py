import re
import matplotlib
import pyteomics
import collections
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from ToFCIMSAnalysis.constants.elements import Elements

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


    def _RemoveSubstring(self, string, list_of_substrings):

        """
        Remove list of substrings from string.
        + list_of_substrings: list of str.
        """

        for substring in list_of_substrings:
            assert isinstance(substring, str), "List should not contain non string elements."
            string = string.replace(substring,"")
        return string


    def CountElements(self, moiety):

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


    def CountedElementsToFormula(self, ordered_dict):

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


    def RemoveReagentIon(self, formula):

        """
        Takes a chemical formula string and removes the reagent ion.
        Reagent ion (X) must be succeeded by a . then the formula.
        e.g. I.C6H6O6, NO3.C12H18O6
        + formula: str.
        """

        return formula.split(".")[1]


    def AddReagentIon(self, formula, reagent_ion):

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

        counted_elements = self.CountElements(moiety)
        exact_mass = 0
        for key, value in counted_elements.iteritems():            
            exact_mass += (self.GetMass(key) * value)
        return round(exact_mass, 6)


    def MassDefect(self, exact_mass):

        """
        Calculates the mass defect for a given exact mass.
        + exact_mass: float.
        """

        integer_mass = int(np.round(exact_mass))
        mass_defect = integer_mass - exact_mass
        return round(mass_defect, 8)


    def KendrickMassDefect(self, moiety_exact_mass, kendrick_base_exact_mass, peak_assignment_error=20.0):

        """
        Calculates the Kendrick mass defect for a given exact mass and kenrdick base.
        + moiety_exact_mass: float.
        + kendrick_base_exact_mass: float. 
        + peak_assignment_error: float. units. ppm.
        """
        peak_assignment_error /= 1e6

        moiety_mass_defect = self.MassDefect(moiety_exact_mass)

        kendrick_base_integer_mass = int(np.round(kendrick_base_exact_mass))
        kendrick_normalisation_factor = kendrick_base_integer_mass / kendrick_base_exact_mass
        
        kendrick_exact_mass = moiety_exact_mass * kendrick_normalisation_factor
        kendrick_integer_mass = int(np.round(kendrick_exact_mass))
        kendrick_mass_defect = round(kendrick_integer_mass - kendrick_exact_mass, 8)

        e = moiety_exact_mass * peak_assignment_error
        kendrick_e = round(e * kendrick_normalisation_factor, 8)

        return kendrick_mass_defect, kendrick_e


    def ElementRatios(self, moiety):

        """
        Returns O:C, H:C and N:C of moiety.
        + moiety: str.
        """

        counted_elements = self.CountElements(moiety)

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


    def IsHydrocarbon(self, moiety):

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


    def ElementDistributions(self, list_of_formulae, elements):
        
        """
        Calculates frequency of elements in list_of_moieties.
        + list_of_moieties: list of str.
        + elements: list of str.
        """

        # convert all strings into ordered dictionaries to count elements
        counted_moieties = [self.CountElements(moiety) for moiety in list_of_formulae]

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
                    freq_dist.loc[counted_moiety[element], element] += 1.0
                else:
                    freq_dist.loc[0,element] += 1.0

        return freq_dist


    def ElementDistributionsPlot(self, list_of_freq_df):
    
        """
        Displays data generated from self.ElementDistributions.
        freq_df = dataframe of numeric index (bins) and frequency count 
        of elements. Provided by chemical_formulas.element_frequency
        show_0 sets xlim as either 0 or 1.
        """
        
        matplotlib.rc('xtick', labelsize=20) 
        matplotlib.rc('ytick', labelsize=20) 
        
        # Get biggest bin from all dataframes
        xmax = max([df.shape[0] for df in list_of_freq_df])
        x = np.arange(0,xmax,1)

        # Get total number of elements asked for
        elements = list(set(self._Flatten([df.columns.values for df in list_of_freq_df])))
        element_i_key = {element: i for i, element in enumerate(elements)}
        
        # Start figure
        fig, ax = plt.subplots(nrows=len(elements), figsize=(13, len(elements)*3), sharex=True)
        fig.suptitle("Frequency of atoms");

        # FREQUENCY OF ATOMS CHON LEFT PANNEL ALL ROWS
        ncols_per_bin = len(list_of_freq_df)
        width = (1.0/ncols_per_bin)*0.8
        
        # Manually compute offsets for bars
        offsets = [0]
        if ncols_per_bin == 2:
            offsets = [-(width/ncols_per_bin), (width/ncols_per_bin)]
        if ncols_per_bin == 3:
            offsets = [-(width), 0, (width)]
        if ncols_per_bin > 3:
            raise ValueError("Too many datasets")
            
        cols = ['r','b','k']
        # for each dataframe
        for j, df in enumerate(list_of_freq_df):
            # for each element in the dataframe
            for element in list(set(df.columns)):
                # get the index of the element to plot
                i = element_i_key[element]
                ax[i].grid(zorder=1);
                ax[i].bar(x=[x+offsets[j] for x in df.index], height=df[element],
                          width=width, zorder=10, color=cols[j]);
                ax[i].set_ylabel(element, rotation=0, fontsize=20);
                ax[i].set_xlim(-1, xmax)

        return ax



    def OrganicCharacteristics(self, list_of_formulae):
    
        """
        Calculate O:C, H:C, N:C and average OSc for a list of ions.
        list_of_formulas = list of strings containing molecular formulas
        returns pandas dataframe
        """
        
        # initialise dataframe
        df = pd.DataFrame({'ion' : list_of_formulae})
        # Get unique elements in the formulaes given
        counted_elements = list(set(self._Flatten([self.CountElements(moiety) for 
                                                  moiety in list_of_formulae])))
        # initialise empty element columns in dataframe
        for element in counted_elements + ["O:C","H:C","N:C","OSc"]:
            df[element] = 0
        # loop over each row
        for i, moiety in enumerate(list_of_formulae):     
            # equivalent to one hot encoding 
            # for elements in formula
            ce = self.CountElements(moiety)
            for key in ce.keys():
                df.loc[i, key] = ce[key]         
            # calculate element_ratios and OSc
            OC, HC, NC = self.ElementRatios(moiety)
            df.loc[i, ["O:C","H:C","N:C"]] = OC, HC, NC
            if df.loc[i,'N'] == 0:
                # ignore N:C if N not present
                NC = 0
            df.loc[i, "OSc"] = self.Osc(OC, HC, NC)
        return df



    def OrganicCharacteristicsPlot(self, df, cmap="Blues", alphas=None, sizes=None, a=0.5):

        """
        Displays data generated from self.OrganicCharacteristics.
        df = dataframe of organic characteristics.
        returns figure axes.
        """

        import matplotlib
        matplotlib.rc('xtick', labelsize=20);
        matplotlib.rc('ytick', labelsize=20);

        fig, ax = plt.subplots(ncols=3, figsize=(15,4));

        ax[0].grid(zorder=1);
        ax[1].grid(zorder=1);
        ax[2].grid(zorder=1);

        ax[0].scatter(df["O:C"], df["H:C"], zorder=10, alpha=a, cmap=cmap, s=sizes, c=alphas, edgecolor="k");
        ax[1].scatter(df["C"],  df["O"],    zorder=10, alpha=a, cmap=cmap, s=sizes, c=alphas, edgecolor="k");
        ax[2].scatter(df["C"],  df["OSc"],  zorder=10, alpha=a, cmap=cmap, s=sizes, c=alphas, edgecolor="k");

        ax[0].set_xlabel("O:C", fontsize=20);
        ax[0].set_ylabel("H:C", fontsize=20);
        ax[1].set_xlabel("nC", fontsize=20);
        ax[1].set_ylabel("nO", fontsize=20);
        ax[2].set_xlabel("nC", fontsize=20);
        ax[2].set_ylabel("OSc", fontsize=20);

        plt.subplots_adjust(left=0.12, bottom=0.11,
                            right=0.90, top=0.94,
                            wspace=0.34, hspace=0.26);

        return ax