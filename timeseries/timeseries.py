import pandas as pd
import numpy as np
import scipy as sc
import tqdm

class TimeSeries():

    def __init__(self):
        pass


    def CountCycles(self, df, mask_column_name, new_cycle_column_name,
                    last_num_in_seq, first_num_in_seq):
    
        """
        This method inserts a column into a pandas dataframe. That column contains
        the cycle number i.e. the number of times a repeated pattern in a mask column
        of the dataframe that is repeated. This is used to count FIGAERO cycles and
        background cylces.
        df. pd.DataFrame. pandas dataframe containing time series data.
        mask_column_name. str. The name of the mask column in the df that contains the 
        repeating cycles.
        new_cycle_column_name. str. the column name for the counted cycles e.g. 'bg_count'.
        last_num_in_seq. int. defines the last mask value for a particular cycle. 
        first_num_in_seq. int. defines the first mask value for a particular cycle. 
        e.g. if the cycle goes sample(1), ramp(2), soak(3), cool(4), then
        last_num_in_seq = 4 and first_num_in_seq = 1.
        """
    
        # initialise new column name and a counter
        df[new_cycle_column_name] = 0
        count = 0
        # loop over every row in the dataframe
        for i in range(len(df[mask_column_name])):
            index = df.index[i]
            previous_index = df.index[i-1]
    
            if (df.loc[index, mask_column_name] == last_num_in_seq) & \
                (df.loc[previous_index, mask_column_name] == first_num_in_seq):
                count += 1
            df.loc[index, new_cycle_column_name] = count
    
        return df


    def FindBackgrounds(self, df, columns_to_bg, mask_column_name, bg_cycle_column_name,
                        bg_cycle_mask_val, method_of_mn='mean'):

        """
        Find the background values (identified by the mask) for each cycle and sets as a new column.
        df. pd.DataFrame. pandas dataframe of time series.
        columns_to_bg. list. list containing column names from df which are to have background found for them.
        mask_column_name. str. The name of the mask column in the df that contains the repeating cycles.
        bg_cycle_column_name. str. The name of the column that lists the number of bg cycles. This requires
        that CountCycles() has been run for the background data.
        bg_cycle_mask_val. int. The integer in the mask_column_name that gives the background period.
        method_of_mn. str. defines how the single value bg number is calculated from the array of bg values.
        returns df.
        Remember: the mass+"_bg" column created by this function just separates the background value from the 
        raw data. Once you have this column you might need to forward fill or interpolate between the points.
        """

        for mass in columns_to_bg:
            df[mass+"_bg"] = np.nan
            for i in np.arange(0, max(df[bg_cycle_column_name])):
                # Find bg values of that bg period (i) ie where bg_cycle_column_name is bg_cycle_mask_val
                bg_values = df.loc[(df[bg_cycle_column_name] == i) & (df[mask_column_name] == bg_cycle_mask_val), mass].values
                # work out single value
                if method_of_mn == 'min':
                    mn = np.nanmin(bg_values)
                else:
                    mn = np.nanmean(bg_values)

                # put in dataframe
                df.loc[(df[bg_cycle_column_name] == i) & (df[mask_column_name] == bg_cycle_mask_val), mass+"_bg"] = mn

        return df


    def IntegrateFIGAERO(self, df, columns_to_integrate, mask_column_name, figaero_cycle_column_name,
                        figaero_integrate_mask_val, gas_sample_mask_val):
        """
        Integrates the datapoints in columns_to_integrate of the df where the 
        figaero_integrate_mask_val values are found in the mask_column_name column of the df.
        This is performed for every cycle in the cycle_column_name. Requires CountCycles has been run.
        df. pandas dataframe.
        columns_to_integrate. list of strings. columns in dataframe e.g. ['mz_230', 'mz_360']
        mask_column_name. str. name of mask column.
        cycle_column_name. str. cycle column generated from CountCycles.
        figaero_integrate_mask_val. list of ints. these define the datapoints to integrate.
        gas_sample_mask_val. int. defines the period that the integrate value relates to.
        """
            
        # Set the gas sampling mask i.e. where gas phase sampling is happening.
        gas_sample_mask = (df[mask_column_name] == gas_sample_mask_val)
        # Set the figaero integrate mask i.e. where the figaero integration should take place.
        # This can either be a single int e.g. 3, when the ramp (3) is happening; or a list of ints
        # e.g. [3, 4] as we want to integrate both the ramp (3) and the soak (4).
        figaero_integrate_mask = df[mask_column_name].isin(figaero_integrate_mask_val)
        for mass in columns_to_integrate:
            df[mass+"_particle_integrated"] = np.nan
            for i in np.arange(0, max(df[figaero_cycle_column_name])):

                current_cycle_mask = df[figaero_cycle_column_name] == i

                current_integrate =  current_cycle_mask & figaero_integrate_mask
                previous_sample = current_cycle_mask & gas_sample_mask
                integrated_values = sc.trapz(df.loc[current_integrate, mass].values)

                first = np.where(df.index == previous_sample[previous_sample==True].index[-1])[0][0]
                last = np.where(df.index == previous_sample[previous_sample==True].index[0])[0][0]
                mid = first + ((last - first) / 2.0)
                df_index = previous_sample.index[int(mid)]
                df.loc[df_index, mass+"_particle_integrated"] = integrated_values

        return df