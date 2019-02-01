import pandas as pd
import numpy as np
import scipy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt

class TimeSeries():

    def __init__(self):
        pass


    def CountCycles(self, df, mask_column_name, new_cycle_column_name,
                    last_num_in_seq, first_num_in_seq):

        """
        This method inserts a column into a pandas dataframe. That column contains
        the cycle number i.e. the number of times a repeated pattern in a mask column
        of the dataframe is repeated. This is typically used to count FIGAERO cycles and
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

        # calculate difference in consecutive mask values
        df[mask_column_name+'_diff'] = np.append(np.diff(df[mask_column_name]), np.array([0]));
        # mark where the difference in sequence edges occurs with a 1
        new_cycle = (df[mask_column_name+'_diff'] == first_num_in_seq - last_num_in_seq).astype(int) & \
                    (df[mask_column_name] == last_num_in_seq).astype(int)
        # cumulative sum gives the cycle number
        df[new_cycle_column_name] = np.cumsum(new_cycle)

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

        for mass in tqdm(columns_to_bg):
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
                        figaero_integrate_mask_val, gas_sample_mask_val, bad_cycles=[]):
        
        """
        Integrates the datapoints in columns_to_integrate of the df where the 
        figaero_integrate_mask_val values are found in the mask_column_name column of the df.
        This is performed for every cycle in the cycle_column_name. Requires CountCycles has been run.
        df. pandas dataframe.
        columns_to_integrate. list of strings. columns in dataframe e.g. ['mz_230', 'mz_360']
        mask_column_name. str. name of mask column.
        figaero_cycle_column_name. str. cycle column generated from CountCycles.
        figaero_integrate_mask_val. list of ints. these define the datapoints to integrate.
        gas_sample_mask_val. int. defines the period that the integrate value relates to.
        """
            
        # Set the gas sampling mask i.e. where gas phase sampling is happening.
        gas_sample_mask = (df[mask_column_name] == gas_sample_mask_val)
        # Set the figaero integrate mask i.e. where the figaero integration should take place.
        # This can either be a single int e.g. 3, when the ramp (3) is happening; or a list of ints
        # e.g. [3, 4] as we want to integrate both the ramp (3) and the soak (4).
        figaero_integrate_mask = df[mask_column_name].isin(figaero_integrate_mask_val)
        for mass in tqdm(columns_to_integrate):
            df[mass+"_particle_integrated"] = np.nan
            for i in np.arange(0, max(df[figaero_cycle_column_name])):
                if i in bad_cycles:
                    pass
                else:
                    current_cycle_mask = df[figaero_cycle_column_name] == i
                    current_integrate = current_cycle_mask & figaero_integrate_mask
                    previous_sample = current_cycle_mask & gas_sample_mask
                    integrated_values = sc.trapz(df.loc[current_integrate, mass].values)
                    first = np.where(df.index == previous_sample[previous_sample==True].index[-1])[0][0]
                    last = np.where(df.index == previous_sample[previous_sample==True].index[0])[0][0]
                    mid = first + ((last - first) / 2.0)
                    df_index = previous_sample.index[int(mid)]
                    df.loc[df_index, mass+"_particle_integrated"] = integrated_values

        return df


    def GetTMaxes(self, df, columns_to_get_tmax, mask_column_name="state_name", bad_cycles=[],
                  thermogram_mask_val=[], figaero_cycle_column_name="figaero_cycle",
                  temperature_column_name="temperature"):

        """
        Returns pandas dataframe of column keys and list values with elements
        that are the max value for each cycle period.
        df. pandas dataframe.
        columns_to_get_tmax. list of strings. columns in dataframe e.g. ['mz_230', 'mz_360']
        thermogram_mask_val. list of ints. these define the datapoints to integrate
        figaero_cycle_column_name. str. cycle column generated from CountCycles.
        temperature_column_name. str. column containing temperature data.
        bad_cycles. list of ints. ignore getting tmax from these cycle numbers.
        """

        # Set the thermogram mask i.e. where the figaero integration should take place.
        # This can either be a single int e.g. 3, when the ramp (3) is happening; or a list of ints
        # e.g. [3, 4] as we want to integrate both the ramp (3) and the soak (4).
        thermogram_mask = df[mask_column_name].isin(thermogram_mask_val)
                
        tmaxes = pd.DataFrame()
        for mass in tqdm(columns_to_get_tmax):
            ts = []
            for i in np.arange(0, max(df[figaero_cycle_column_name])):
                if i in bad_cycles:
                    pass
                else:
                    ramp = (df[figaero_cycle_column_name] == i) & thermogram_mask
                    thermogram_df = df[ramp].copy()
                    thermogram_df = thermogram_df.set_index(temperature_column_name)
                    max_point = max(thermogram_df[mass])
                    tmax = thermogram_df[thermogram_df[mass] == max_point][mass].index.values[0]
                    ts.append(tmax)
            tmaxes[mass] = ts
        tmaxes.index.name = figaero_cycle_column_name
        
        return tmaxes


    def ExtractDesorptions(self, df, columns_to_extract, mask_column_name,
                           figaero_cycle_column_name, thermogram_mask_val,
                           temperature_dp, temperature_column_name, bad_cycles=[]):
        
        """
        Provides a dictionary of dataframes for each mass (column) in ts df, where the
        columns are the desorption number (taken from the figaero_cycle_column_name) 
        with a temperature index.
        df. pandas dataframe.
        columns_to_extract. list of strings. columns in dataframe e.g. ['mz_230', 'mz_360']
        bad_cycles. list of ints. ignore extracting desorptions from these cycle numbers.
        mask_column_name. str. name of mask column.
        figaero_cycle_column_name. str. cycle column generated from CountCycles.
        thermogram_mask_val. list of ints. these define the datapoints in the desorption.
        temperature_dp. int. sig fig for temperature rounding. You will need to play with this
        to find the best bin widths to use for your dataset.
        """
        
        def myround(x, base=1):
            if np.isnan(x):
                return np.nan
            else:
                return float(base * round(float(x)/base))
        # Set the thermogram mask i.e. where the figaero integration should take place.
        # This can either be a single int e.g. 3, when the ramp (3) is happening; or a list of ints
        # e.g. [3, 4] as we want to integrate both the ramp (3) and the soak (4).
        thermogram_mask = df[mask_column_name].isin(thermogram_mask_val)
        # extract temperature data
        temperature_data = df.loc[thermogram_mask].set_index(temperature_column_name)
         # dictionary to store dataframes
        desorptions = {}
        for mass in tqdm(columns_to_extract):
            desorptions[mass] = pd.DataFrame()        
            # loop over figaero cycles
            for cycle in np.unique(df[figaero_cycle_column_name])[:-1]: 
                if cycle in bad_cycles:
                    pass
                else:
                    # set the specific data we want to extract
                    temp_data = temperature_data.loc[(temperature_data[figaero_cycle_column_name] == cycle), mass]
                    # put that data into a temporary dataframe
                    temp_df = pd.DataFrame({cycle:temp_data.values}, index=temp_data.index)
                    # concatenate that into the dataframe for that mass
                    desorptions[mass] = desorptions[mass].append(temp_df)
            # reset index to get temperature as a column
            desorptions[mass].reset_index(inplace=True)
            # perform the rounding on the temperature column
            desorptions[mass]['temperature'] = [myround(x,temperature_dp) for x in desorptions[mass]['temperature']]
            # collect rows which are rounded to the same value
            desorptions[mass] = desorptions[mass].groupby('temperature').mean()

        return desorptions


    def PlotExtractDesorptions(self, df, figaero_cycles_to_plot=[], ax=None):
    
        """
        Quick plots series' of thermogram data and their mean.
        df. pandans.DataFrame. Must be of the format generated by  
        the ExtractDesorptions() method.
        figaero_cycles_to_plot. list of ints. 
        """
        
        ax = df.interpolate().mean(axis=1).plot(zorder=10, grid=True, marker="o", color="k",
                                                label="Mean", legend=True, figsize=(15,5));
        ax.fill_between(df.index,
                        y1=df.interpolate().mean(axis=1)-df.interpolate().std(axis=1),
                        y2=df.interpolate().mean(axis=1)+df.interpolate().std(axis=1),
                        zorder=1, alpha=0.30, color="k");
        if not figaero_cycles_to_plot:
            df.interpolate(axis=1).plot(zorder=1, ax=ax, grid=True, marker="+", ls="--");
        else:
            df[figaero_cycles_to_plot].interpolate(axis=1).plot(zorder=1, ax=ax, grid=True, marker="+", ls="--");
        
        ax.set_ylabel("Counts / Hz");
        
        return ax


    def MakeDiurnalDataset(self, df, resample_rule):

        """
        Takes time series in a pandas DataFrame with a datetime index.
        Returns dataframe with diurnal data generated from the time series'.
        df. pandas DataFrame. Must have datetime index.
        resample_rule. str. Provided to df.resample(). Gives time step of 
        returned diurnal dataframe.
        """

        diurnal = df.copy().resample(resample_rule).mean()
        diurnal['hour'] = diurnal.index.strftime("%H:%M")
        diurnal = diurnal.groupby('hour').describe()
        diurnal.index = pd.DatetimeIndex(diurnal.index)

        return diurnal


    def PlotCorrelationMatrix(self, df_corr, **kwargs):

        """
        Visualise the correlations between species.
        Returns matplotlib.pyplot Axis.
        df_corr. pandas DataFrame. correlation DataFrame e.g. df.corr()
        **kwargs passed to plt.pcolor().
        """
        
        fig, ax = plt.subplots(figsize=(12,10));
        pcolor = ax.pcolor(df_corr, edgecolors='k',vmin=-1,vmax=1,**kwargs);
        cbar = plt.colorbar(mappable=pcolor,ax=ax,shrink=0.75,ticks=np.linspace(-1,1,9));    
        cbar.ax.set_ylabel('r', rotation=0, fontsize=16);
        ax.set_yticks(np.linspace(0,len(df_corr.index)-1,len(df_corr.index))+0.5);
        ax.set_xticks(np.linspace(0,len(df_corr.columns)-1,len(df_corr.columns))+0.5);
        ax.set_yticklabels(df_corr.index);
        ax.set_xticklabels(df_corr.columns, rotation=90);
        
        return ax