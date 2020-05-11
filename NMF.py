class PerformNMF():

    """
    Perform non negtive matrix factorisation.
    """

    def __init__(self, NMF_param_grid):

        """
        NMF_param_grid contains hyperparameters for NMF. dict of hyperparameter keys.
        self.NMF_param_grid stores NMF_param_grid
        self.models empty list to contain models
        self.run_results empty list to contain results
        self.consensus_matricies empty dict. n_component keys and concencus matrix values
        """

        self.NMF_param_grid = NMF_param_grid
        self.models = []
        self.run_results = pd.DataFrame()
        self.consensus_matrices = {}


    def MakeModels(self):

        """Initialises models with parameters from self.NMF_parameter_grid"""

        self.models = []
        for params in tqdm(ParameterGrid(self.NMF_param_grid), desc="Make models"):
            self.models.append(NMF(**params))

        self.run_results = pd.DataFrame(columns=list(self.models[0].get_params().keys())+["W","H","err","RSS","mean RSS","connectivity_matrix","cophenetic"])


    def FindClusters(self, W):

        """
        The decomposed matrix W is of dimensions nxp
        where p is the number of factors and n number of time steps
        for each row (n) we find the highest value (i.e.
        contribution to which factor) and return the p.
        e.g. for the nth row, [0.1,0.5,0.2], p=1 has the highest
        contribution.
        W : 2d numpy array

        returns array of highest contributed p's of len(n)
        returns 1d numpy array
        """

        return W.argmax(axis=1) # based on max contribution


    def MakeConnectivityMatrix(self, clusters):

        """
        Takes list of highest contributed clusters (clusters)
        then makes an empty matrix in which clusters are connected
        with themselves. If the cluster number is the same, fill the
        matrix with a 1, otherwise leave it a zero.
        clusters: 1d numpy array

        returns 2d numpy array
        """
        # Make empty matrix
        connectivity_matrix = np.zeros((len(clusters), len(clusters)))
        # Iterate over each row and column.
        # if the numbers match, then 1 else, 2.
        for i, val1 in enumerate(clusters):
            for j, val2 in enumerate(clusters):
                if val1 == val2:
                    connectivity_matrix[i,j] = 1

        return connectivity_matrix


    def MakeReorderedConsensusMatrix(self, concencus_matrix):

        """
        Based on :
        https://www.atmos-meas-tech-discuss.net/amt-2019-404/amt-2019-404.pdf
        https://www.pnas.org/content/101/12/4164

        Reorder the concencus matrix
        concencus_matrix 2d numpy array
        returns 2d numpy array
        """

        M = pd.DataFrame(concencus_matrix)
        Y = 1 - M
        Z = linkage(squareform(Y), method='average')
        ivl = leaves_list(Z)
        ivl = ivl[::-1]
        reorderM = pd.DataFrame(M.values[:, ivl][ivl, :], index=M.columns[ivl], columns=M.columns[ivl])

        return reorderM


    def GetCCC(self, concencus_matrix):

        """
        Based on :
        https://www.atmos-meas-tech-discuss.net/amt-2019-404/amt-2019-404.pdf
        https://www.pnas.org/content/101/12/4164

        taken from https://github.com/marinkaz/nimfa/blob/master/nimfa/models/nmf.py

        concencus_matrix 2d np.array
        returns float
        """
        M = concencus_matrix
        # upper diagonal elements of consensus
        avec = np.array([M[i, j] for i in range(M.shape[0] - 1)
                        for j in range(i + 1, M.shape[1])])
        # consensus entries are similarities, conversion to distances
        Y = 1 - avec
        Z = linkage(Y, method='average')
        # cophenetic correlation coefficient of a hierarchical clustering
        # defined by the linkage matrix Z and matrix Y from which Z was
        # generated
        return cophenet(Z, Y)[0]


    def CalculateRSS(self, sum_data, W, H):

        """Calculate the residual sum of squares."""

        df = pd.DataFrame(
            data=np.dot(W, H).sum(axis=0),
            index=sum_data.index,
            columns=["reconstructed"]
        )
        df["original"] = sum_data
        rss = sum((df["original"] - df["reconstructed"])**2)

        return rss


    def MakeConsensusMatrices(self, NMF_param_grid, run_results):

        """
        Makes the concensus matrices.
        NMF_param_grid. dict of hyper parameter keys and values
        run_results. of type self.run_results
        returns dict to self.consensus_matrices
        """

        consens_mat_results = {}
        # find CCC for each n_components
        for n in tqdm(NMF_param_grid["n_components"], desc="CCC"):
            # Subselect chunk of run results for n n_components run
            chunk = run_results.loc[run_results["n_components"] == n].copy()
            connectivity_chunk = chunk["connectivity_matrix"]
            rss_chunk = chunk["RSS"]

            # Make the mean RSS
            mean_rss = connectivity_chunk.values.mean(axis=0)
            # Make the concensus matrix
            concens_matrix = connectivity_chunk.values.mean(axis=0)
            # Reorder the concensus matrix
            concens_matrix_reordered = self.MakeReorderedConsensusMatrix(concens_matrix)
            # get cophenetic correlation coefficient
            ccc = self.GetCCC(concens_matrix)

            # Store concensus matricies
            consens_mat_results["consensus_matrix {}".format(n)] = concens_matrix
            consens_mat_results["consensus_matrix_reordered {}".format(n)] = concens_matrix_reordered
            # Store cophenetic correlation coefficient
            run_results.loc[run_results["n_components"] == n, "cophenetic"] = ccc

        return consens_mat_results


    def RunModels(self, X):

        """
        Runs the NMF models stored in self.models
        Computes metrics
        Stores in self.run_results

        """

        for model in tqdm(self.models, desc="Run models"):

            # Get info of model
            params = model.get_params()
            W = model.fit_transform(X.T)
            H = model.components_

            # Put reconstructions into dict
            params["W"] = W
            params["H"] = H
            params["err"] = model.reconstruction_err_
            params["RSS"] = self.CalculateRSS(X.sum(axis=1), W, H)
            # Get factor to which the species contributes most
            clusters = params["W"].argmax(axis=1)
            # put connectivity matrices into parameters
            params["connectivity_matrix"] = self.MakeConnectivityMatrix(clusters)
            # store results
            self.run_results = self.run_results.append(pd.DataFrame([params]))

        # give run results an index
        self.run_results.reset_index(inplace=True)

        # Make consensus matrices
        self.consensus_matrices = self.MakeConsensusMatrices(self.NMF_param_grid, self.run_results)



    def PlotConcencusMatrices(self, NMF_obj, ax=None, **kwargs):

        """Plots concencus matrixces for each n component."""

        n_components = NMF_obj.NMF_param_grid["n_components"]
        if ax:
            pass
        else:
            fig, ax = plt.subplots(figsize=(5*len(n_components),10), ncols=np.nanmax(n_components)-1, nrows=2)
            for n in range(len(N_COMPONENTS)):
                im = ax[0,n].matshow(NMF_obj.consensus_matrices["consensus_matrix {}".format(n_components[n])])
                ax[1,n].matshow(NMF_obj.consensus_matrices["consensus_matrix_reordered {}".format(n_components[n])])
                ax[0,n].set_title("{} components".format(n_components[n]))

            cbar_ax = fig.add_axes([1.0, 0.15, 0.05, 0.7]);
            fig.colorbar(im, cax=cbar_ax);

        return ax

    def PlotCopheneticVsComponents(self, NMF_obj, ax=None, **kwargs):

        if ax:
            NMF_obj.run_results[["n_components","cophenetic"]].drop_duplicates().plot(ax=ax, x="n_components",y="cophenetic", marker="+")
        else:
            ax = NMF_obj.run_results[["n_components","cophenetic"]].drop_duplicates().plot(x="n_components",y="cophenetic")

        return ax


    def PlotRSSVsComponents(self, NMF_obj, ax=None):

        if ax:
            NMF_obj.run_results[["n_components","RSS"]].drop_duplicates().plot(ax=ax, x="n_components",y="RSS", marker="o", legend=False)
        else:
            ax = NMF_obj.run_results[["n_components","RSS"]].drop_duplicates().plot(x="n_components",y="RSS", marker="o", legend=False)

        return ax


    def PlotMultipleSolutions(self, NMF_obj, list_of_n_solns, inset_n_soln):

        """
        Plots a maximum of 3 different solutions and their components.
        Can choose which solution to add to inset.
        NMF_obj. NMF object from class PerformNMF()
        list_of_n_solns. list of ints
        inset_n_soln. int.
        returns matplotlib ax
        """

        solns1=list_of_n_solns[0]
        solns2=list_of_n_solns[1]
        solns3=list_of_n_solns[2]
        insetsolns=inset_n_soln

        cmap = [matplotlib.colors.rgb2hex(rgb[:3]) for rgb in cm.gist_rainbow(np.linspace(0, 1, solns3))]

        fig, ax = plt.subplots(figsize=(18,7), ncols=np.nanmax(list_of_n_solns)+1, nrows=3, sharex=True, sharey=False)
        plt.subplots_adjust(wspace=0.48)

        ax[0,0].set_ylabel("{} factors".format(solns1))
        ax[1,0].set_ylabel("{} factors".format(solns2))
        ax[2,0].set_ylabel("{} factors".format(solns3))
        for c in np.arange(solns1,solns3+1):
            ax[0,c].axis('off')
            if c > solns1:
                ax[1,c].axis('off')
            if c > solns2:
                ax[2,c].axis('off')

        axinset = fig.add_axes([0.75, 0.40, 0.25, 0.45])
        axinset.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)


        for i, c in enumerate(np.arange(0,solns1,1)):#enumerate([0,3,4,2,1]):#enumerate([3,2,0,1]):#enumerate(np.arange(0,solns,1)):

            ax[0,i].plot(NMF_obj.run_results.loc[NMF_obj.run_results["n_components"] == solns1, "H"].reset_index().loc[0,"H"][c], color=cmap[c], lw=5)
            ax[0,i].tick_params(axis='y', which='both', left=False, labelleft=False)

            twinx = ax[0,i].twinx()
            twinx.vlines(75, -0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(300,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(550,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(720,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.add_patch(patches.Rectangle(xy=(-0.2, -0.2), width=75, height=1.5, color="grey", alpha=0.3))
            twinx.set_ylim(0,1)
            twinx.tick_params(axis='y', which='both', right=False, labelright=False)


        for i, c in enumerate(np.arange(0,solns2,1)):#enumerate([0,3,4,5,2,1]):#enumerate([3,2,0,4,1]):#enumerate(np.arange(0,solns,1)):

            ax[1,i].plot(NMF_obj.run_results.loc[NMF_obj.run_results["n_components"] == solns2, "H"].reset_index().loc[0,"H"][c], color=cmap[c], lw=5)
            ax[1,i].tick_params(axis='y', which='both', left=False, labelleft=False)

            twinx = ax[1,i].twinx()
            twinx.vlines(75, -0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(300,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(550,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(720,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.add_patch(patches.Rectangle(xy=(-0.2, -0.2), width=75, height=1.5, color="grey", alpha=0.3))
            twinx.set_ylim(0,1)
            twinx.tick_params(axis='y', which='both', right=False, labelright=False)


        for i, c in enumerate(np.arange(0,solns3,1)):#enumerate([0,3,4,5,2,1,6]):#enumerate([3,2,0,5,1,4]):#enumerate(np.arange(0,solns,1)):

            ax[2,i].plot(NMF_obj.run_results.loc[NMF_obj.run_results["n_components"] == solns3, "H"].reset_index().loc[0,"H"][c], color=cmap[c], lw=5)
            ax[2,i].tick_params(axis='y', which='both', left=False, labelleft=False)
            ax[2,i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

            twinx = ax[2,i].twinx()
            twinx.vlines(75, -0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(300,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(550,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(720,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.add_patch(patches.Rectangle(xy=(-0.2, -0.2), width=75, height=1.5, color="grey", alpha=0.3))
            twinx.set_ylim(0,1)
            twinx.tick_params(axis='y', which='both', right=False, labelright=False)


        for i, c in enumerate(np.arange(0,insetsolns,1)):#enumerate([0,3,4,5,2,1,6]):#enumerate([3,2,0,5,1,4]):#enumerate(np.arange(0,solns,1)):

            #inset_dat = MinMaxScaler().fit_transform(simple_nmf_NOx20ppb.run_results.loc[simple_nmf_NOx20ppb.run_results["n_components"] == insetsolns, "H"].reset_index().loc[0,"H"][c].reshape(-1,1))
            inset_dat = NMF_obj.run_results.loc[NMF_obj.run_results["n_components"] == insetsolns, "H"].reset_index().loc[0,"H"][c].reshape(-1,1)
            axinset.plot(inset_dat, color=cmap[c], lw=3, zorder=20);
            axinset.tick_params(axis='y', which='both', left=False, labelleft=False)
            #axinset.set_ylim(0,1)

            twinx = axinset.twinx()
            twinx.vlines(75, -0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(300,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(550,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.vlines(720,-0.2, 1.6, linestyle=":", linewidth=1)
            twinx.add_patch(patches.Rectangle(xy=(-0.2, -0.2), width=75, height=1.5, color="grey", alpha=0.1))
            twinx.set_ylim(0,1)
            twinx.tick_params(axis='y', which='both', right=False, labelright=False)

           # ax[0,i].set_ylim(0,0.6)
           # ax[1,i].set_ylim(0,0.6)
           # ax[2,i].set_ylim(0,0.6)

        return ax
