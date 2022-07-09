from persistable.plot import StatusbarHoverManager
from persistable.borrowed._hdbscan_boruvka import KDTreeBoruvkaAlgorithm, BallTreeBoruvkaAlgorithm
from persistable.borrowed.dense_mst import stepwise_dendrogram_with_core_distances
from persistable.aux import lazy_intersection
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree, BallTree
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import DisjointSet
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

_TOL = 1e-08
_DEFAULT_FINAL_K = 0.2

class Persistable :

    def __init__(self, X, metric = "minkowski", measure = None, maxk = None, leaf_size = 40, p = 2) :
        self._data = X
        self._p = p
        self._mpspace = _MetricProbabilitySpace(X, metric, measure, leaf_size, self._p)
        if maxk == None:
            maxk = int(_DEFAULT_FINAL_K * X.shape[0]) + 1
        self._maxk = maxk
        #self._mpspace.fit()
        self._mpspace.fit(maxk=maxk)
        self._connection_radius = self._mpspace.connection_radius()

    def parameter_selection(self, initial_k = 0.02, final_k = _DEFAULT_FINAL_K, n_parameters=50, firstn=10, fig_size=(10,3)):
        parameters = np.logspace(np.log10(initial_k), np.log10(final_k), num=n_parameters)
        sks = [ (self._connection_radius,k) for k in parameters ]
        pds = self._mpspace.lambda_linkage_prominence_vineyard(sks,k_indexed=True)
        _, ax = plt.subplots(figsize=fig_size)
        plt.xscale("log")
        plt.yscale("log")
        vineyard = _ProminenceVineyard(parameters,self._connection_radius,pds,k_varying=True, firstn=firstn)
        vineyard.plot_prominence_vineyard(ax)
        plt.ylim([np.quantile(np.array(vineyard._values),0.1),max(vineyard._values)])
        plt.show()

    def parameter_selection_s(self, n_parameters=50, firstn=10, fig_size=(10,3)):
        k0 = 0.02
        s1,s2 = self._mpspace.connection_radius([0.75,0.95])
        parameters = np.logspace(np.log10(s2), np.log10(s2*2), num=n_parameters)
        sks = [ (s,k0) for s in parameters ]
        pds = self._mpspace.lambda_linkage_prominence_vineyard(sks,k_indexed=True)
        _, ax = plt.subplots(figsize=fig_size)
        plt.xscale("log")
        plt.yscale("log")
        vineyard = _ProminenceVineyard(parameters,k0,pds,k_varying=False,firstn=firstn)
        vineyard.plot_prominence_vineyard(ax)
        plt.ylim([np.quantile(np.array(vineyard._values),0.1),max(vineyard._values)])
        plt.show()

    def persistence_diagram(self, s0, k0):
        hc = self._mpspace.lambda_linkage(s0,k0)
        return hc.persistence_diagram()

    def cluster(self,num_clusters,s0=None,k0=0.2,cluster_all=False,cluster_all_k=5):
        if num_clusters <= 1:
            warnings.warn("num_clusters must be greater than 1.")
            return
        if s0 == None:
            s0 = self._connection_radius
        hc = self._mpspace.lambda_linkage(s0,k0)
        #bd = hc.persistence_diagram()[0]
        bd = hc.persistence_diagram()
        pers = np.abs(bd[:,0] - bd[:,1])
        spers = np.sort(pers)
        if num_clusters >= bd.shape[0] :
            warnings.warn("num_clusters is larger than the number of gaps.")
            threshold = spers[0] / 2
        else :
            #print(np.abs(spers[-num_clusters] - spers[-(num_clusters+1)]))
            if np.abs(spers[-num_clusters] - spers[-(num_clusters+1)]) < _TOL:
                warnings.warn("The gap selected is too small to produce a reliable clustering.")
            threshold = (spers[-num_clusters] + spers[-(num_clusters+1)])/2
            #print(threshold)
        cl = hc.persistence_based_flattening(threshold)

        def _post_processing(dataset, labels, k) :
            neigh = KNeighborsClassifier(n_neighbors=k, p=self._p)
            neigh.fit(dataset[labels!=-1], labels[labels!=-1])
            res = labels.copy()
            res[labels==-1] = neigh.predict(dataset[labels==-1,:])
            return res
        
        if cluster_all:
            #labels = _post_processing(self._data, cl[1], k=cluster_all_k)
            labels = _post_processing(self._data, cl, k=cluster_all_k)
            return labels
        else:
            return cl

class _MetricProbabilitySpace :
    """Implements a finite metric probability space that can compute \
       its kernel density estimates"""

    def __init__(self, X, metric = 'minkowski', measure = None, leaf_size = 40, p = 2) :
        # if metric = 'precomputed' then assumes that X is a distance matrix
        # to do: check that input is correct
        self._metric = metric
        self._p = p
        self._leaf_size = leaf_size
        self._size = X.shape[0]
        if measure is None :
            self._measure = np.full(self._size, 1./self._size)
        else :
            self._measure = measure
        self._dimension = X.shape[1]
        self._metric = metric
        if metric != 'precomputed' :
            self._points = X
        else :
            self._points = np.array(range(self._size))
        self._fitted_nn = False
        self._fitted_density_estimates = False
        self._nn_distance = None
        self._nn_indices = None
        self._kernel_estimate = None
        self._maxk = None
        self._maxs = None
        self._tol = _TOL
        if metric in BallTree.valid_metrics:
            self._tree = BallTree(X, metric=metric, leaf_size=leaf_size, p = p)
        elif metric in KDTree.valid_metrics:
            self._tree = KDTree(X, metric=metric, leaf_size=leaf_size, p = p)
        elif metric == 'precomputed':
            self._dist_mat = X
        else :
            raise Exception("Metric given is not supported.")

    def fit(self, maxk = None) :
        self.fit_nn(maxk = maxk)
        self.fit_density_estimates()

    def fit_nn(self, maxk) :
        # to do: check input
        if maxk == None or maxk > self._size :
            maxk = self._size
        self._maxk = maxk
        if self._metric in BallTree.valid_metrics + KDTree.valid_metrics:
            k_neighbors = self._tree.query(\
                    self._points, self._maxk, return_distance = True, sort_results = True,
                    dualtree = True, breadth_first = True)
            k_neighbors = (np.array(k_neighbors[1]),np.array(k_neighbors[0]))
            maxs_given_by_maxk = np.min(k_neighbors[1][:,-1])
            self._maxs = maxs_given_by_maxk
            neighbors = k_neighbors[0]
            _nn_distance = k_neighbors[1]
        else :
            #warnings.warn("For now, for distance matrix we assume maxk = number of points.")
            self._maxk = self._size
            self._maxs = 0
            neighbors = np.argsort(self._dist_mat)
            _nn_distance = self._dist_mat[np.arange(len(self._dist_mat)), neighbors.transpose()].transpose()
        self._nn_indices = np.array(neighbors)
        self._nn_distance = np.array(_nn_distance)
        self._fitted_nn = True

    def fit_density_estimates(self) :
        self._fitted_density_estimates = True
        self._kernel_estimate = np.cumsum(self._measure[self._nn_indices], axis = 1)

    def kde_at_index_width(self, point_index, neighbor_index, width = None) :
        # to do: check input
        if width is None :
            width = self._nn_distance[point_index][neighbor_index]
        return self._kernel_estimate[point_index][neighbor_index]

    def kde(self, point_index, width) :
        # to do: check input (in particular that the index is in bound)
        width = np.array(width)
        # find the index (resp. indices) of the neighbor (resp. neighbors)
        # whose distance is (left) closest to width (resp. each element of width)
        pos = np.searchsorted(self._nn_distance[point_index], width, side = 'right')
        pos -= 1
        # number of local neighbors of the point
        n_neighbors = len(self._nn_distance[point_index])
        # check if the k value we computed is exact or only a lower bound
        # (in that case, annotate it in the out_of_range list)
        if n_neighbors < self._size :
            if width.ndim == 1 :
                # two conditions needed for out of bound
                out_of_range = np.where(pos == n_neighbors-1, True, False)
                if self._maxs > self._tol :
                    out_of_range_ = np.where(width > self._maxs, True, False)
                    out_of_range = np.logical_and(out_of_range, out_of_range_)
            else :
                out_of_range = (pos == n_neighbors-1 and self._nn_distance[pos] > self._maxs)
        else :
            if width.ndim == 1 :
                out_of_range = np.full(len(width),False)
            else :
                out_of_range = False
        return self.kde_at_index_width(point_index,pos,width), out_of_range

    def core_distance(self, point_index, s0, k0) :
        i_indices = []
        if s0 != np.inf:
            mu = s0/k0
            k_to_s = lambda y : s0 - mu * y
            for p in point_index:
                i_indices.append(lazy_intersection(self._kernel_estimate[p], self._nn_distance[p], s0, k0))
            i_indices = np.array(i_indices)
            out_of_range = i_indices[:,1]
            if np.any(out_of_range) :
                # to do: better message for second condition
                warnings.warn("Don't have enough neighbors to properly compute core scale, or point takes too long to appear.")
            i_indices = i_indices[:,0]
            op = lambda p, i : np.where(k_to_s(self._kernel_estimate[p,i-1]) <= self._nn_distance[p,i],\
                    k_to_s(self._kernel_estimate[p,i-1]),
                    self._nn_distance[p,i])
            return np.where(i_indices == 0, 0, op(point_index,i_indices))
        else :
            for p in point_index :
                i_indices.append(np.searchsorted(self._kernel_estimate[p], k0, side = 'left'))
            i_indices = np.array(i_indices)
            if self._maxk < self._size :
                out_of_range = np.where((i_indices >=\
                    np.apply_along_axis(len,-1,self._nn_indices[point_index])) &\
                    (np.apply_along_axis(len,-1,self._nn_indices[point_index]) < self._size), True, False)
                if np.any(out_of_range) :
                    warnings.warn("Don't have enough neighbors to properly compute core scale.")
            return self._nn_distance[(point_index, i_indices)]

    def lambda_linkage(self, s0, k0) :
        indices = np.arange(self._size)
        core_scales = np.minimum(s0,self.core_distance(indices, s0, k0))
        if self._metric in BallTree.valid_metrics:
            sl = BallTreeBoruvkaAlgorithm(self._tree, core_scales, self._nn_indices, leaf_size=self._leaf_size // 3, metric = self._metric, p = self._p).spanning_tree()
        elif self._metric in KDTree.valid_metrics:
            sl = KDTreeBoruvkaAlgorithm(self._tree, core_scales, self._nn_indices, leaf_size=self._leaf_size // 3, metric = self._metric, p = self._p).spanning_tree()
        else:
            sl = stepwise_dendrogram_with_core_distances(self._size, self._dist_mat, core_scales)
        merges = sl[:,0:2].astype(int)
        merges_heights = np.minimum(s0,sl[:,2])
        return _HierarchicalClustering(core_scales, merges, merges_heights, s0)

    def lambda_linkage_prominence_vineyard(self, sks, k_indexed) :

        def _prominences(bd) :
            return np.sort(np.abs(bd[:,0] - bd[:,1]))[::-1]

        prominence_diagrams = []
        for sk in sks :
            s0, k0 = sk
            hc = self.lambda_linkage(s0, k0)
            #persistence_diagram = hc.persistence_diagram()[0]
            persistence_diagram = hc.persistence_diagram()
            if k_indexed:
                mu = k0/s0
                s_to_k = lambda x : k0 - mu * x
                persistence_diagram = s_to_k(persistence_diagram)
            prominence_diagram = _prominences(persistence_diagram)
            prominence_diagrams.append(prominence_diagram)
        return prominence_diagrams

    def connection_radius(self,percentiles=1) :
        if self._metric in BallTree.valid_metrics:
            mst = BallTreeBoruvkaAlgorithm(self._tree, np.zeros(self._size), self._nn_indices, leaf_size=self._leaf_size // 3, metric = self._metric, p = self._p).spanning_tree()
        elif self._metric in KDTree.valid_metrics:
            mst = KDTreeBoruvkaAlgorithm(self._tree, np.zeros(self._size), self._nn_indices, leaf_size=self._leaf_size // 3, metric = self._metric, p = self._p).spanning_tree()
        elif self._metric == "precomputed":
            mst = stepwise_dendrogram_with_core_distances(self._size, self._dist_mat, np.zeros(self._size))
        return np.quantile(mst[:,2],percentiles)
        
class _HierarchicalClustering :

    def __init__(self, heights, merges, merges_heights, maxr) :
        self._merges = merges
        self._merges_heights = np.minimum(maxr,merges_heights)
        self._heights = np.minimum(maxr,heights)
        self._maxr = maxr

    def persistence_based_flattening(self, threshold) :
        end = self._maxr
        heights = self._heights
        merges = self._merges
        merges_heights = self._merges_heights
        n_points = heights.shape[0]
        n_merges = merges.shape[0]
        # this orders the point by appearance
        appearances = np.argsort(heights)
        # contains the current clusters
        uf = DisjointSet()
        # contains the birth time of clusters that are alive
        clusters_birth = {}
        clusters_died = {}
        # contains the flat clusters
        clusters = []
        # height index
        hind = 0
        # merge index
        mind = 0
        current_appearence_height = heights[appearances[0]]
        current_merge_height = merges_heights[0]
        while True :
            # while there is no merge
            while hind < n_points and heights[appearances[hind]] <= current_merge_height and heights[appearances[hind]] < end :
                # add all points that are born as new clusters
                uf.add(appearances[hind])
                clusters_birth[appearances[hind]] = heights[appearances[hind]]
                hind += 1
                if hind == n_points :
                    current_appearence_height = end
                else :
                    current_appearence_height = heights[appearances[hind]]
            # while there is no cluster being born
            while mind < n_merges and merges_heights[mind] < current_appearence_height and merges_heights[mind] < end :
                xy = merges[mind]
                x, y = xy
                rx = uf.__getitem__(x)
                ry = uf.__getitem__(y)
                # if both clusters are alive
                if rx not in clusters_died and ry not in clusters_died :
                    bx = clusters_birth[rx]
                    by = clusters_birth[ry]
                    # if both have lived for more than the threshold, have them as flat clusters
                    if bx + threshold <= merges_heights[mind] and by + threshold <= merges_heights[mind]:
                        clusters.append(list(uf.subset(x)))
                        clusters.append(list(uf.subset(y)))
                        uf.merge(x,y)
                        uf.add(mind + n_points)
                        uf.merge(x,mind + n_points)
                        rxy = uf.__getitem__(x)
                        clusters_died[rxy] = True
                    # otherwise, merge them
                    else :
                        # then merge them
                        del clusters_birth[rx]
                        del clusters_birth[ry]
                        uf.merge(x,y)
                        uf.add(mind + n_points)
                        uf.merge(x,mind + n_points)
                        rxy = uf.__getitem__(x)
                        clusters_birth[rxy] = min(bx, by)
                # if both clusters are already dead, just merge them into a dead cluster
                elif rx in clusters_died and ry in clusters_died :
                    uf.merge(x,y)
                    uf.add(mind + n_points)
                    uf.merge(x,mind + n_points)
                    rxy = uf.__getitem__(x)
                    clusters_died[rxy] = True
                # if only one of them is dead
                else :
                    # we make it so that ry already died and rx just died
                    if rx in clusters_died :
                        x, y = y, x
                        rx, ry = ry, rx
                    # if x has lived for longer than the threshold, have it as a flat cluster
                    if clusters_birth[rx] + threshold <= merges_heights[mind] :
                        clusters.append(list(uf.subset(x)))
                    # then merge the clusters into a dead cluster
                    uf.merge(x,y)
                    uf.add(mind + n_points)
                    uf.merge(x,mind + n_points)
                    rxy = uf.__getitem__(x)
                    clusters_died[rxy] = True
                mind += 1
                if mind == n_merges:
                    current_merge_height = end
                else :
                    current_merge_height = merges_heights[mind]
            if (hind == n_points or heights[appearances[hind]] >= end) and (mind == n_merges or merges_heights[mind] >= end) :
                break
        # go through all clusters that have been born but haven't been merged
        for x in range(n_points) :
            if x in uf._indices:
                rx = uf.__getitem__(x)
                if rx not in clusters_died :
                    if clusters_birth[rx] + threshold <= end:
                        clusters.append(list(uf.subset(x)))
                    clusters_died[rx] = True
        current_cluster = 0
        res = np.full(n_points, -1)
        for cl in clusters :
            for x in cl :
                if x < n_points :
                    res[x] = current_cluster
            current_cluster += 1
        return res

    def persistence_diagram(self, tol = _TOL) :
        end = self._maxr
        heights = self._heights
        merges = self._merges
        merges_heights = self._merges_heights
        n_points = heights.shape[0]
        n_merges = merges.shape[0]
        # this orders the point by appearance
        appearances = np.argsort(heights)
        # contains representative points for the clusters that are alive
        cluster_reps = np.full(heights.shape[0] + merges.shape[0], -1)
        # contains the persistence diagram
        pd = []
        # height index
        hind = 0
        # merge index
        mind = 0
        current_appearence_height = heights[appearances[0]]
        current_merge_height = merges_heights[0]
        while True :
            # while there is no merge
            while hind < n_points and heights[appearances[hind]] <= current_merge_height and heights[appearances[hind]] < end :
                # add all points that are born as new clusters
                cluster_reps[appearances[hind]] = appearances[hind]
                hind += 1
                if hind == n_points :
                    current_appearence_height = end
                else :
                    current_appearence_height = heights[appearances[hind]]
            # while there is no cluster being born
            while mind < n_merges and merges_heights[mind] < current_appearence_height and merges_heights[mind] < end :
                xy = merges[mind]
                x, y = xy
                rx = cluster_reps[x]
                ry = cluster_reps[y]
                bx = heights[rx]
                by = heights[ry]
                # assume x was born before y
                if bx > by:
                    x, y = y, x
                    bx, by = by, bx
                    rx, ry = ry, rx
                pd.append([by,merges_heights[mind]])
                cluster_reps[mind + n_points] = rx
                cluster_reps[ry] = -1
                mind += 1
                if mind == n_merges:
                    current_merge_height = end
                else :
                    current_merge_height = merges_heights[mind]
            if (hind == n_points or heights[appearances[hind]] >= end) and (mind == n_merges or merges_heights[mind] >= end) :
                break
        # go through all clusters that are still alive
        for i in range(heights.shape[0]):
            if cluster_reps[i] == i:
                pd.append([heights[i],end])
        pd = np.array(pd)
        return pd[np.abs(pd[:,0]-pd[:,1]) > tol]

class _ProminenceVineyard :
    
    def __init__(self, varying_parameters, fixed_parameter, prominence_diagrams, k_varying = True, firstn = 20) :
        if firstn == None:
            self._firstn = -1
        else:
            self._firstn = firstn
        self._parameters = varying_parameters
        self._prominence_diagrams = [pd[:firstn] for pd in prominence_diagrams]
        self._values = []
        self._fixed_parameter = fixed_parameter
        self.k_varying = k_varying

    def _vineyard_to_vines(self):
        times = self._parameters
        prominence_diagrams = self._prominence_diagrams
        num_vines = np.max([len(prom) for prom in prominence_diagrams])
        padded_prominence_diagrams = np.zeros((len(times),num_vines))
        for i in range(len(times)):
            padded_prominence_diagrams[i,:len(prominence_diagrams[i])] = prominence_diagrams[i]
        return [ (times,padded_prominence_diagrams[:,j]) for j in range(num_vines) ]

    def plot_prominence_vineyard(self, ax, interpolate=True, areas=True, points=False):

        def _vine_parts(times, prominences, tol = 1e-8):
            parts = []
            current_vine_part = []
            current_time_part = []
            part_number = 0
            for i in range(len(times)):
                if prominences[i] < tol :
                    if len(current_vine_part) > 0:
                        # we have constructed a non-trivial vine part that has now ended
                        if part_number != 0 :
                            # this is not the first vine part, so we prepend 0 to the vine and the previous time to the times
                            current_vine_part.insert(0,0)
                            current_time_part.insert(0,times[i-len(current_vine_part)])
                        # finish the vine part with a 0 and the time with the current time
                        current_vine_part.append(0)
                        current_time_part.append(times[i])
                        ## we save the current vine part and start over
                        parts.append( ( np.array(current_vine_part), np.array(current_time_part)))
                        part_number += 1
                        current_vine_part = []
                        current_time_part = []
                    # else, we haven't constructed a non-trivial vine part, so we just keep going
                elif i == len(times)-1 :
                    if part_number != 0 :
                        # this is not the first vine part, so we prepend 0 to the vine and the previous time to the times
                        current_vine_part.insert(0,0)
                        current_time_part.insert(0,times[i-len(current_vine_part)])
                    # finish the vine part with its value and the time with the current time
                    current_vine_part.append(prominences[i])
                    current_time_part.append(times[i])
                    # we save the final vine part and time
                    parts.append( ( np.array(current_vine_part), np.array(current_time_part)))
                else :
                    # we keep constructing the vine part, since the prominence is non-zero
                    current_vine_part.append(prominences[i])
                    current_time_part.append(times[i])
            return parts

        times = self._parameters
        #prominence_diagrams = self._prominence_diagrams
        vines = self._vineyard_to_vines()
        num_vines = len(vines)
        cscheme = lambda x : plt.cm.viridis(x)
        if self._firstn == -1 :
            colors = cscheme(np.linspace(0, 1, num_vines)[::-1])
        else :
            colors = list(cscheme(np.linspace(0, 1, self._firstn)[::-1]))
            last = colors[-1]
            colors.extend([last for _ in range(num_vines-self._firstn)])
        if self.k_varying:
            shm = StatusbarHoverManager(ax, "s0 = {:.3e}".format(self._fixed_parameter) + ", k0 = {:.3e}")
        else :
            shm = StatusbarHoverManager(ax, "s0 = {:.3e}, " +  ("k0 = {:.3e}".format(self._fixed_parameter)))
        if areas:
            for i in range(len(vines)-1):
                artist = ax.fill_between(times, vines[i][1], vines[i+1][1], color = colors[i])
                shm.add_artist_labels(artist, "gap " + str(i+1))
            ax.fill_between(times, vines[len(vines)-1][1], 0, color = colors[len(vines)-1])
            shm.add_artist_labels(artist, "gap " + str(i+1))
        for i,tv in enumerate(vines) :
            times, vine = tv
            for vine_part, time_part in _vine_parts(times,vine) :
                if interpolate:
                    artist = ax.plot(time_part,vine_part, c="black")
                    #shm.add_artist_labels(artist, "vine " + str(i+1))
                if points:
                    artist = ax.plot(time_part,vine_part, "o", c="black")
                    #shm.add_artist_labels(artist, "vine " + str(i+1))
                self._values.extend(vine_part)
