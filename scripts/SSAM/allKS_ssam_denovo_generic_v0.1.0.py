
############################################################################################
### Automatic runscript for SSAM denovo. Just change the sample params block and go!
############################################################################################

print ("Loading libraries")

import numpy as np;

import pandas as pd;

import matplotlib.pyplot as plt;

import ssam;

import seaborn as sns;

import os;

from collections import defaultdict;

from sklearn import preprocessing;

from sklearn.decomposition import PCA;

import pickle;

from scipy.stats import pearsonr, spearmanr;

from matplotlib import colors;

from matplotlib.colors import to_rgba;

from datetime import datetime

files = [file for file in os.listdir('./all_csv_files') if file.endswith('.csv')]

print(files)

for sample in files: 

    now = datetime.now()

    date_time_str = now.strftime("%Y%m%d_%H%M%S")

    ############################################################################################
    ### sample parameters
    ############################################################################################

    print ("Reading parameters")

    #sample = "220KS_Decoded_LowThreshold.csv"

    #sample = "220KS_AP4_Decoded_LowThreshold_noPOU5F1_nSLC26A3_noFABP1_noBBC3_noANTXR1_noENG_noFCGR3A_ssam.csv"

    olympus_um_per_pixel = 0.1625

    df = pd.read_csv(os.path.join('all_csv_files',sample),usecols=['X', 'Y', 'gene'])
    df.columns=['target', 'x', 'y']
    df["x"]*= olympus_um_per_pixel
    df["y"]*= olympus_um_per_pixel
    print(df)

    outdir = "../Output_files/220KS"

    print ("Input spot file: " + sample )

    print ("Output folder: " + outdir + date_time_str + "/")

    os.mkdir(outdir + date_time_str + "/", mode = 0o777)

    outdir = outdir + date_time_str + "/"  # modify to include date

    um_per_pixel = 3


    ncores = 2 # Number of CPU processing cores used for kernel density estimation

    ############################################################################################
    ### processing parameters
    ############################################################################################

    # KDE params

    bw=2.5 # IMPORTANT... but leave as default? # this should be set according to molecule density. 2-4 is a good range to explore.

    outdir_kde = sample+"_kde_bw"+str(bw)+"pixPerum"+str(um_per_pixel)+"/"

    bw_ext='_bw'+str(bw)+'umPerPix'+str(um_per_pixel)

    # local max params # perhaps do not change this...

    local_max_min_expression=0.027 # def 0.027

    local_max_min_norm=0.08 # this should be investigated and set. No guidelines right now. def 0.2.

    local_max_search_size=3 # not sure of effect

    local_max_ext=bw_ext+"_exp"+str(local_max_min_expression)+"norm"+str(local_max_min_norm)

    # clustering

    clust_vec_min_cluster_size=20 # IMPORTANT!!# can be increased for only robest types... or reduced for rare

    clust_vec_pca_dims=30 # IMPORTANT!!# perhaps this should be set to the tSNE and UMAP components?

    clust_vec_resolution=0.3 # IMPORTANT!!# this is the Louvain resolution. Higher means fewer clusters. Lower means more clusters.

    clust_vec_maxcorr=0.95 # IMPORTANT!!# clusters with higher correlation than this will be merged

    clust_vec_metric='correlation'

    clust_vec_ext=local_max_ext+"_minsize"+str(clust_vec_min_cluster_size)+"pca"+str(clust_vec_pca_dims)+"res"+str(clust_vec_resolution)+"maxCorr"+str(clust_vec_maxcorr)

    # UMAP/tSNE params # probably doesnt require changing

    num_p_componants=clust_vec_pca_dims # used for tSNE and UMAP. Harmonise with clust_vec_pca_dims?

    tsne_s=10 # size of spots in scatter plot

    tsne_ext=clust_vec_ext+"_pca"+str(num_p_componants)+"s"+str(tsne_s)

    umap_ext=clust_vec_ext+"_pca"+str(num_p_componants)

    # cell type map filtering 

    filter_method = "local"

    filter_params = {
        "block_size": 151,
        "method": "mean",
        "mode": "constant",
        "offset": 0.2
    }

    filter_cmap_min_r=0.3 # IMPORTANT!!# minimum correlation to be considered for annotating cell type map

    filter_cmap_min_blob_area=4 # IMPORTANT!! # minimum size of annotated area to filter small local noise

    cmap_ext=clust_vec_ext+"_minR"+str(filter_cmap_min_r)+"minBlob"+str(filter_cmap_min_blob_area)

    # domain params

    domains_step=5

    domain_radius=50

    domain_n_clusters=12

    domain_merge_remote=True

    domain_merge_thres=0.7

    domain_local_norm_thresh=500

    domain_extension=cmap_ext+"_step"+str(domains_step)+"rad"+str(domain_radius)+"nclus"+str(domain_n_clusters)+"merge"+str(domain_merge_remote)+str(domain_merge_thres)+"norm"+str(domain_local_norm_thresh)

    ############################################################################################
    ### setup data
    ############################################################################################

    print("Reading spot file: um_per_pixel="+str(um_per_pixel))

    pos_dic = defaultdict(lambda: [])

    df.x = (df.x - df.x.min()) / um_per_pixel + 10

    df.y = (df.y - df.y.min()) / um_per_pixel + 10

    pos_dic = defaultdict(lambda: [])

    #xmin, ymin, zmin = 10000000, 10000000, 10000000

    #xmax, ymax, zmax = -10000000, -10000000, -10000000

    xmin = df.x.min()
    xmax = df.x.max()
    ymin = df.y.min()
    ymax = df.y.max()

    all_genes = sorted(df.target.unique())

    for gene in all_genes : 
        mask = df.target == gene
        gene_coordinates = [df[mask].x,df[mask].y]
        pos_dic[gene] = gene_coordinates


    # with open(sample) as f:
    #     f.readline()
    #     for line in f:
    #         e = line.strip().split(',')
    #         # the index of e should change based on the columns expected for x,y,(z) coordinates and the gene name g. 0-based index.
    #         x, y, g = e[1], e[2], e[0]
    #         x, y = [float(e) for e in [x, y]]
    #         if x > xmax:
    #             xmax = x
    #         if y > ymax:
    #             ymax = y
    #         if x < xmin:
    #             xmin = x
    #         if y < ymin:
    #             ymin = y
            
    # with open(sample) as f:
    #     f.readline()
    #     for line in f:
    #         e = line.strip().split(',')
    #         # the index of e should change based on the columns expected for x,y,(z) coordinates and the gene name g. 0-based index.
    #         x, y, g = e[1], e[2], e[0]
    #         x, y = [float(e) for e in [x, y]]
    #         x -= xmin
    #         y -= ymin
    #         x, y = [e*um_per_pixel + 10 for e in [x, y]]
    #         pos_dic[g].append([x, y])
            
    for g in pos_dic: pos_dic[g] = np.array(pos_dic[g]).T

    width = int((xmax - xmin) + 10)

    height = int((ymax - ymin) + 10)

    ds = ssam.SSAMDataset(all_genes, [pos_dic[gene] for gene in all_genes], width, height)

    ############################################################################################
    ### Run KDE and investigate expression and normalised expression thresholds
    ############################################################################################

    print("Running KDE: bw=" + str(bw))

    analysis = ssam.SSAMAnalysis(ds, ncores=ncores, save_dir=outdir_kde, verbose=True)

    analysis.run_fast_kde(bandwidth=bw, use_mmap=False)

    analysis.find_localmax(search_size=local_max_search_size, min_norm=local_max_min_norm, min_expression=local_max_min_expression)

    analysis.normalize_vectors()

    ### Plot local max

    plt.figure(figsize=[10, 10])

    ds.plot_l1norm(cmap="Greys", rotate=1)

    plt.scatter(ds.local_maxs[0], ds.local_maxs[1], c="blue", s=0.1)

    plt.tight_layout()

    plt.savefig(outdir+"localmax_ExpThres"+local_max_ext+'.png', bbox_inches='tight')

    plt.close()

    ### Expression threshold

    print("Defining expression threshold: local_max_min_expression=" + str(local_max_min_expression))

    viewport = 0.1

    gindices = np.arange(len(ds.genes))

    plt.figure(figsize=[20, 40])

    for i, gidx in enumerate(gindices[:6], start=1):
        ax = plt.subplot(5, 2, i)
        n, bins, patches = ax.hist(ds.vf[..., gidx][np.logical_and(ds.vf[..., gidx] > 0, ds.vf[..., gidx] < viewport)], bins=100, log=True, histtype=u'step')
        ax.set_xlim([0, viewport])
        ax.set_ylim([n[-1], n[0]])
        ax.axvline(local_max_min_expression, c='red', ls='--')
        ax.set_title(ds.genes[gidx])
        ax.set_xlabel("Expression")
        ax.set_ylabel("Count")

    plt.tight_layout()

    plt.savefig(outdir+"ExpThres"+local_max_ext+'.png', bbox_inches='tight')

    plt.close()

    ### Norm threshold

    print("Defining L1 norm/local maxima threshold: local_max_min_norm="+str(local_max_min_norm))

    gidx = 0

    plt.figure(figsize=[10, 5])

    n, _, _ = plt.hist(ds.vf_norm[np.logical_and(ds.vf_norm > 0, ds.vf_norm < 0.3)], bins=100, log=True, histtype='step')

    ax = plt.gca()

    ax.axvline(local_max_min_norm, c='red', ls='--')

    ax.set_xlabel("L1-norm")

    ax.set_ylabel("Count")

    plt.xlim([0, 0.3])

    plt.ylim([np.min(n), np.max(n) + 100000])

    plt.tight_layout()

    plt.savefig(outdir+"NormExp"+local_max_ext+'.png', bbox_inches='tight')

    plt.close()

    ############################################################################################
    ### Cluster vectors, perform tSNE/UMAP, make celltype map
    ############################################################################################

    ### PCA plot

    print("Performing PCA")

    pca = PCA().fit(analysis.dataset.normalized_vectors)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))

    plt.xlabel('number of components')

    plt.ylabel('cumulative explained variance')

    plt.axvline(x = clust_vec_pca_dims, color = 'b', label='Number of PCs used from clustering vectors')

    plt.savefig(outdir+'PCA'+cmap_ext+'.png', bbox_inches='tight')

    ### Cluster vectors

    print("Clustering vectors: min_cluster_size="+str(clust_vec_min_cluster_size)+" pca_dims="+str(clust_vec_pca_dims)+" resolution="+str(clust_vec_resolution)+" metric="+str(clust_vec_metric)+" max_correlation="+str(clust_vec_maxcorr))

    analysis.cluster_vectors(
        min_cluster_size=clust_vec_min_cluster_size,
        pca_dims=clust_vec_pca_dims,
        resolution=clust_vec_resolution,
        metric=clust_vec_metric,
        max_correlation=clust_vec_maxcorr)

    # post-filtering parameter for cell-type map

    print("Making cell type map: min_norm="+str(filter_method)+" filter_params="+str(filter_params)+" min_r="+str(filter_cmap_min_r)+" min_blob_area="+str(filter_cmap_min_blob_area))

    analysis.map_celltypes()

    analysis.filter_celltypemaps(min_norm=filter_method, filter_params=filter_params, min_r=filter_cmap_min_r, fill_blobs=True, min_blob_area=filter_cmap_min_blob_area)

    plt.figure(figsize=[10, 10])

    ds.plot_celltypes_map(rotate=3, set_alpha=False)

    plt.savefig(outdir+'celltype_map'+cmap_ext+'.png', bbox_inches='tight')

    plt.close()

    ### tSNE

    print("Running tSNE: pca_dims="+str(num_p_componants))

    plt.figure(figsize=[10, 10])

    #ds.plot_tsne(pca_dims=num_p_componants, metric="correlation", s=tsne_s, run_tsne=True, tsne_kwargs=dict(square_distances=True))

    ds.plot_tsne(pca_dims=num_p_componants, metric="correlation", s=tsne_s, run_tsne=True)

    plt.savefig(outdir+'celltype_tsne'+tsne_ext+'.png', bbox_inches='tight')

    plt.close()

    ### UMAP

    print("Running UMAP: pca_dims="+str(num_p_componants))

    plt.figure(figsize=[10, 10])

    ds.plot_umap(pca_dims=num_p_componants, metric="correlation",  random_state=0)

    plt.savefig(outdir+'celltype_umap'+umap_ext+'.png', bbox_inches='tight')

    plt.close()


    ############################################################################################
    ### Investigate cluster localisation and expression usig diagnostic plots
    ############################################################################################

    ### diagnostic plots

    def plot_diagnostic_plot(self, centroid_index, cluster_name=None, cluster_color=None, cmap=None, rotate=0, z=None, use_embedding="tsne", known_signatures=[], correlation_methods=[]):
            """
            Plot the diagnostic plot. This method requires `plot_tsne` or `plot_umap` was run at least once before.
            
            :param centroid_index: Index of the centroid for the diagnostic plot.
            :type centroid_index: int
            :param cluster_name: The name of the cluster.
            :type cluster_name: str
            :param cluster_color: The color of the cluster. Overrides `cmap` parameter.
            :type cluster_color: str or list(float)
            :param cmap: The colormap for the clusters. The cluster color is determined using the `centroid_index` th color of the given colormap.
            :type cmap: str or matplotlib.colors.Colormap
            :param rotate: Rotate the plot. Possible values are 0, 1, 2, and 3.
            :type rotate: int
            :param z: Z index to slice 3D vector norm and cell-type map plots.
                If not given, the slice at the middle will be used.
            :type z: int
            :param use_embedding: The type of the embedding for the last panel. Possible values are "tsne" or "umap".
            :type use_embedding: str
            :param known_signatures: The list of known signatures, which will be displayed in the 3rd panel. Each signature can be 3-tuple or 4-tuple,
                containing 1) the name of signature, 2) gene labels of the signature, 3) gene expression values of the signature, 4) optionally the color of the signature.
            :type known_signatures: list(tuple)
            :param correlation_methods: The correlation method used to determine max correlation of the centroid to the `known_signatures`. Each method should be 2-tuple,
                containing 1) the name of the correaltion, 2) the correaltion function (compatiable with the correlation methods available in `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_)
            :type correlation_methods: list(tuple)
            """
            from matplotlib import colors
            if z is None:
                z = int(self.vf_norm.shape[2] / 2)
            p, e = self.centroids[centroid_index], self.centroids_stdev[centroid_index]
            if cluster_name is None:
                cluster_name = "Cluster #%d"%centroid_index
            
            if cluster_color is None:
                if cmap is None:
                    cmap = plt.get_cmap("jet")
                cluster_color = cmap(centroid_index / (len(self.centroids) - 1))
            
            if len(correlation_methods) == 0:
                correlation_methods = [("r", corr), ]
            total_signatures = len(correlation_methods) * len(known_signatures) + 1
                    
            ax = plt.subplot(1, 4, 1)
            mask = self.filtered_cluster_labels == centroid_index
            plt.scatter(self.local_maxs[0][mask], self.local_maxs[1][mask], c=[cluster_color])
            self.plot_l1norm(rotate=rotate, cmap="Greys", z=z)
            
            ax = plt.subplot(1, 4, 2)
            ctmap = np.zeros([self.filtered_celltype_maps.shape[0], self.filtered_celltype_maps.shape[1], 4])
            ctmap[self.filtered_celltype_maps[..., z] == centroid_index] = to_rgba(cluster_color)
            ctmap[np.logical_and(self.filtered_celltype_maps[..., z] != centroid_index, self.filtered_celltype_maps[..., 0] > -1)] = [0.9, 0.9, 0.9, 1]
            if rotate == 1 or rotate == 3:
                ctmap = ctmap.swapaxes(0, 1)
            ax.imshow(ctmap, interpolation='nearest')
            if rotate == 1:
                ax.invert_xaxis()
            elif rotate == 2:
                ax.invert_xaxis()
                ax.invert_yaxis()
            elif rotate == 3:
                ax.invert_yaxis()
            
            ax = plt.subplot(total_signatures, 4, 3)
            ax.bar(self.genes, p, yerr=e, error_kw=dict(ecolor='lightgray'))
            ax.set_title(cluster_name)
            plt.xlim([-1, len(self.genes)])
            rects = ax.patches
            for rect, label in zip(rects, self.genes):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, height + 0.02, label, ha="center", va="bottom", rotation=90)
            plt.ylim(bottom=0)
            plt.xticks(rotation=90)
            
            subplot_idx = 0
            for signature in known_signatures:
                sig_title, sig_labels, sig_values = signature[:3]
                sig_colors_defined = False
                if len(signature) == 4:
                    sig_colors = signature[3]
                    sig_colors_defined = True
                for corr_label, corr_func in correlation_methods:
                    corr_results = [corr_func(p, sig_value) for sig_value in sig_values]
                    corr_results = [e[0] if hasattr(e, "__getitem__") else e for e in corr_results]
                    max_corr_idx = np.argmax(corr_results)
                    ax = plt.subplot(total_signatures, 4, 7+subplot_idx*4)
                    lbl = sig_labels[max_corr_idx]
                    if sig_colors_defined:
                        col = sig_colors[max_corr_idx]
                    else:
                        col = cluster_color
                    ax.bar(self.genes, sig_values[max_corr_idx], color=col)
                    ax.set_title("%s in %s (max %s, %.3f)"%(lbl, sig_title, corr_label, corr_results[max_corr_idx]))
                    plt.xlim([-1, len(self.genes)])
                    plt.xticks(rotation=90)
                    subplot_idx += 1
            
            if use_embedding == 'tsne':
                embedding = self.tsne
                fig_title = "t-SNE, %d vectors"%sum(self.filtered_cluster_labels == centroid_index)
            elif use_embedding == 'umap':
                embedding = self.umap
                fig_title = "UMAP, %d vectors"%sum(self.filtered_cluster_labels == centroid_index)
            good_vectors = self.filtered_cluster_labels[self.filtered_cluster_labels != -1]
            ax = plt.subplot(1, 4, 4)
            ax.scatter(embedding[:, 0][good_vectors != centroid_index], embedding[:, 1][good_vectors != centroid_index], c=[[0.8, 0.8, 0.8, 1],], s=80)
            ax.scatter(embedding[:, 0][good_vectors == centroid_index], embedding[:, 1][good_vectors == centroid_index], c=[cluster_color], s=80)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(fig_title)

    ds.normalized_vectors[ds.filtered_cluster_labels > -1].shape

    print("Making diagnostic plots")

    for idx in range(len(ds.centroids)):
        plt.figure(figsize=[40, 10])
    #    plot_diagnostic_plot(ds, idx, known_signatures=[("scRNA-seq", scrna_uniq_labels, scrna_centroids, scrna_colors), ], correlation_methods=[("r", pearsonr),("rho", spearmanr)], rotate=3)
        plot_diagnostic_plot(ds, idx, correlation_methods=[("r", pearsonr),("rho", spearmanr)], rotate=3, use_embedding = "tsne")
        plt.tight_layout()
        os.makedirs(outdir+'diagplots'+cmap_ext+'/', exist_ok=True)
        plt.savefig(outdir+'diagplots'+cmap_ext+'/diagplot_centroid_'+str(idx)+'_'+cmap_ext+'.png')
        plt.close()


    np.save(outdir+"celltypefile.npy", ds.filtered_celltype_maps)

    ############################################################################################
    ### print CSV of signatures
    ############################################################################################

    print("Writing local max gene expression TSV")

    gene_exp_df=pd.DataFrame(ds.normalized_vectors[ds.filtered_cluster_labels > -1], columns=ds.genes, index=ds.filtered_cluster_labels[ds.filtered_cluster_labels > -1])

    gene_exp_df.to_csv(outdir+'expressionHeatmap'+cmap_ext+'.tsv', sep='\t', header=True)

    cluster_labels = sorted(np.unique(ds.filtered_cluster_labels[ds.filtered_cluster_labels > -1]))

    signatures = pd.DataFrame(columns=ds.genes,index=cluster_labels)
    centroids =  ds.centroids
    signatures[:]=centroids
    signatures.to_csv(outdir+'cluster_centroids'+cmap_ext+'.tsv', sep='\t', header=True)
    ###

    print("Plotting local max gene expression heatmap")

    heatmap_vectors = np.zeros([np.sum(ds.filtered_cluster_labels != -1), len(ds.genes)], dtype=float)

    col_colors = np.zeros([np.sum(ds.filtered_cluster_labels != -1), 4])

    acc_idx = 0

    acc_sizes = []

    heatmap_clusters_index = range(len(ds.centroids))

    for cl_idx in heatmap_clusters_index:
        cl_vecs = ds.normalized_vectors[ds.filtered_cluster_labels == cl_idx]
        acc_sizes.append(cl_vecs.shape[0])
        cmap = plt.get_cmap("jet")
        col = cmap(cl_idx / (len(ds.centroids) - 1))
        heatmap_vectors[acc_idx:acc_idx+cl_vecs.shape[0], :] = cl_vecs
        col_colors[acc_idx:acc_idx+cl_vecs.shape[0]] = to_rgba(col)
        acc_idx = np.sum(acc_sizes)

    gene_exp_heatmap = heatmap_vectors.T

    gene_exp_heatmap = preprocessing.scale(gene_exp_heatmap)

    plt.figure(figsize=[15, 75])

    g = sns.clustermap(gene_exp_heatmap, figsize=[15, 75], cmap='bwr', row_cluster=True, col_cluster=True, xticklabels = 1000, vmin=-2.5, vmax=2.5, yticklabels=ds.genes)

    g.cax.set_visible(False)

    g.ax_heatmap.tick_params(labelright=False, labelleft=True, right=False)

    plt.savefig(outdir+'expressionHeatmap'+cmap_ext+'.png', bbox_inches='tight')

    plt.close()


    ############################################################################################
    ### Create domain map
    ############################################################################################

    ### Domain map

    print("Performing domain analysis n_clusters="+str(domain_n_clusters)+" merge_remote="+str(domain_merge_remote)+" merge_thres="+str(domain_merge_thres)+" norm_thres="+str(domain_local_norm_thresh))

    analysis.bin_celltypemaps(step=domains_step, radius=domain_radius)

    analysis.find_domains(n_clusters=domain_n_clusters, merge_remote=domain_merge_remote, merge_thres=domain_merge_thres, norm_thres=domain_local_norm_thresh)

    plt.figure(figsize=[20, 20])

    ds.plot_domains(rotate=3)

    plt.savefig(outdir+'domain_map'+domain_extension+'.png', bbox_inches='tight')

    plt.close()

    print("Done!")


