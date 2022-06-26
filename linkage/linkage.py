import numpy as np
import scipy as sp
import random
import warnings
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import DisjointSet
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from linkage.plot import StatusbarHoverManager
from linkage.from_hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm

_TOL = 1e-8
_INF = 1e15

class Persistable :

    def __init__(self, X, measure = None, maxk = None, leaf_size = 40, p = 2) :
        self._data = X
        self._p = p
        self._mpspace = _MetricProbabilitySpace(X, 'minkowski', measure, leaf_size, self._p)
        self._mpspace.fit()
        self._connection_radius = self._mpspace.connection_radius()
        self._maxk = maxk

    def parameter_selection(self, initial_k = 0.02, final_k = 0.2, n_parameters=50, color_firstn=10, fig_size=(10,3)):
        parameters = np.logspace(np.log10(initial_k), np.log10(final_k), num=n_parameters)
        sks = [ (self._connection_radius,k) for k in parameters ]
        pds = self._mpspace.lambda_linkage_prominence_vineyard(sks,k_indexed=True)
        _, ax = plt.subplots(figsize=fig_size)
        plt.xscale("log")
        plt.yscale("log")
        vineyard = _ProminenceVineyard(parameters,pds)
        vineyard.plot_prominence_vineyard(ax, color_firstn=color_firstn)
        plt.ylim([np.quantile(np.array(vineyard._values),0.1),max(vineyard._values)])
        plt.show()

    def cluster(self,num_clusters,k,s=None,cluster_all=False,cluster_all_k=5):
        if s == None:
            s = self._connection_radius
        hc = self._mpspace.lambda_linkage(s,k)
        bd = hc.persistence_diagram()[0]
        pers = np.abs(bd[:,0] - bd[:,1])
        if num_clusters >= bd.shape[0] :
            spers = np.sort(pers)
            threshold = spers[0] / 2
        else :
            spers = np.sort(pers)
            threshold = (spers[-num_clusters] + spers[-(num_clusters+1)])/2
        cl = hc.persistence_based_flattening(threshold)

        def _post_processing(dataset, labels, k) :
            neigh = KNeighborsClassifier(n_neighbors=k, p=self._p)
            neigh.fit(dataset[labels!=-1], labels[labels!=-1])
            res = labels.copy()
            res[labels==-1] = neigh.predict(dataset[labels==-1,:])
            return res
        
        if cluster_all:
            labels = _post_processing(self._data, cl[1], k=cluster_all_k)
            return cl[0], labels
        else:
            return cl

#    def __init__(self, X, measure = None, k0="auto", leaf_size = 40, p = 2) :
#        self._data = X
#        self._p = p
#        self._mpspace = _MetricProbabilitySpace(X, 'minkowski', measure, leaf_size, self._p)
#        if k0 == "auto":
#            #self.k0 = min((np.log10(self._mpspace._size) * 30)/self._mpspace._size, 1)
#            self.k0 = 0.05
#        # to do: the following was to be changed when the weights are not uniform!
#        self._maxk = int(self.k0 * self._mpspace._size)+1
#        self._mpspace.fit(maxk=self._maxk)
#        self._connection_radius = self._mpspace.connection_radius()

#    def parameter_selection(self, k0 = "auto", ss = "auto", n_parameters=50, color_firstn=10, k_indexed=True):
#        if k0 == "auto":
#            k0 = self.k0
#
#        if ss == "auto":
#            m = self._mpspace.connection_radius(percentiles=0.5)
#            s1, s2 = m, m*10
#        else :
#            s1, s2 = ss
#        parameters = np.logspace(np.log10(s1), np.log10(s2), num=n_parameters)
#        sks = [ (s, k0) for s in parameters ]
#        pds = self._mpspace.lambda_linkage_prominence_vineyard(sks,k_indexed)
#
#        fig, ax = plt.subplots(figsize=(10,3))
#        plt.xscale("log")
#        plt.yscale("log")
#
#        vineyard = _ProminenceVineyard(parameters,pds)
#        vineyard.plot_prominence_vineyard(ax, color_firstn=color_firstn)
#        vv = np.array(vineyard._values)
#        vv = vv[vv>=_TOL]
#
#        plt.ylim([np.quantile(vv,0.05),max(vineyard._values)])
#        plt.show()
#
#    def cluster(self, num_clusters, s0, k0="auto",cluster_all=False,cluster_all_k=5):
#        if k0 == "auto":
#            k0 = self.k0
#        cl = self._mpspace.lambda_linkage(s0,k0).persistence_based_flattening(num_clusters)
#
#        def _post_processing(dataset, labels, k) :
#            neigh = KNeighborsClassifier(n_neighbors=k, p=self._p)
#            neigh.fit(dataset[labels!=-1], labels[labels!=-1])
#            res = labels.copy()
#            res[labels==-1] = neigh._predict(dataset[labels==-1,:])
#            return res
#        
#        if cluster_all:
#            labels = _post_processing(self._data, cl[1], k=cluster_all_k)
#            return cl[0], labels
#        else:
#            return cl

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
        self._square_kernel_estimate = None
        self._kernel_estimate = None
        self._maxk = None
        self._maxs = None
        self._tol = _TOL
        if metric == "minkowski":
            self._tree = KDTree(X, metric=metric, leaf_size=leaf_size, p = p)
        #elif metric == 'precomputed':
        #    self.dist_mat = X
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
        if self._metric == 'minkowski' :
            k_neighbors = self._tree.query(\
                    self._points, self._maxk, return_distance = True, sort_results = True,
                    dualtree = True, breadth_first = True)
            k_neighbors = (np.array(k_neighbors[1]),np.array(k_neighbors[0]))
            maxs_given_by_maxk = np.min(k_neighbors[1][:,-1])
            self._maxs = maxs_given_by_maxk
            neighbors = k_neighbors[0]
            _nn_distance = k_neighbors[1]
        else :
            warnings.warn("For now, for distance matrix we assume maxk = number of points.")
            self._maxk = self._size
            self._maxs = 0
            neighbors = np.argsort(self.dist_mat)
            _nn_distance = self.dist_mat[np.arange(len(self.dist_mat)), neighbors.transpose()].transpose()
        self._nn_indices = np.array(neighbors)
        self._nn_distance = np.array(_nn_distance)
        self._fitted_nn = True

    def fit_density_estimates(self) :
        self._fitted_density_estimates = True
        self._square_kernel_estimate = np.cumsum(self._measure[self._nn_indices], axis = 1)
        self._kernel_estimate = self._square_kernel_estimate

    def kde_at_index_width(self, point_index, neighbor_index, width = None) :
        # to do: check input
        if width is None :
            width = self._nn_distance[point_index][neighbor_index]
        return self._square_kernel_estimate[point_index][neighbor_index]

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

        def _lazy_intersection(increasing, increasing2, f1) :
            # find first occurence of f1(increasing[i]) <= increasing2[i]
            first = 0
            last = len(increasing)-1
            if f1(increasing[first]) <= increasing2[first] :
                return first, False
            if f1(increasing[last]) > increasing2[last] :
                return last, True
            while first+1 < last :
                midpoint = (first + last)//2
                if f1(increasing[midpoint]) <= increasing2[midpoint] :
                    last = midpoint
                else:
                    first = midpoint
            return last, False

        mu = s0/k0
        k_to_s = lambda y : s0 - mu * y
        i_indices = []
        for p in point_index :
            i_indices.append(_lazy_intersection(self._kernel_estimate[p], self._nn_distance[p], k_to_s))
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

    def lambda_linkage(self, s0, k0) :
        indices = np.arange(self._size)
        core_scales = self.core_distance(indices, s0, k0)
        sl = KDTreeBoruvkaAlgorithm(self._tree, core_scales, self._nn_indices, leaf_size=self._leaf_size // 3).spanning_tree()
        merges = sl[:,0:2].astype(int)
        merges_heights = sl[:,2]
        #merges_heights[merges_heights >= _INF*2] = np.inf
        merges_heights[merges_heights <= _TOL*2] = 0
        return _HierarchicalClustering(self._points, core_scales, merges, merges_heights, s0)

    def lambda_linkage_prominence_vineyard(self, sks, k_indexed) :

        def _prominences(bd) :
            return np.sort(np.abs(bd[:,0] - bd[:,1]))[::-1]

        prominence_diagrams = []
        for sk in sks :
            s0, k0 = sk
            hc = self.lambda_linkage(s0, k0)
            persistence_diagram = hc.persistence_diagram()[0]
            if k_indexed:
                mu = k0/s0
                s_to_k = lambda x : k0 - mu * x
                persistence_diagram = s_to_k(persistence_diagram)
            prominence_diagram = _prominences(persistence_diagram)
            prominence_diagrams.append(prominence_diagram)
        return prominence_diagrams

    def connection_radius(self,percentiles=1) :
        mst = KDTreeBoruvkaAlgorithm(self._tree, np.zeros(len(self._points)), self._nn_indices, leaf_size=self._leaf_size // 3).spanning_tree()
        return np.quantile(mst[:,2],percentiles)
        
class _HierarchicalClustering :

    def __init__(self, X, heights, merges, merges_heights, maxr) :
        self._points = X
        #self._covariant = covariant
        #self.dend = dend
        self._merges = merges
        self._merges_heights = merges_heights
        self._heights = heights
        self._maxr = maxr
        #self._minr = minr

    def persistence_based_flattening(self, threshold) :
        #if threshold == None and num_clusters == None :
        #    raise Exception("Either threshold or num_clusters must be given.")
        #if threshold != None and num_clusters != None :
        #    warnings.warn("Both threshold and num_clusters given, using threshold.")
        #elif threshold == None :
        heights = self._heights.copy()
        merges_heights = self._merges_heights.copy()
        #if not self._covariant :
        #    heights = -heights - _TOL
        #    merges_heights = -merges_heights
        #else :
        #    heights = heights - _TOL
        heights = heights - _TOL
        # for numerical reasons, it may be that a point is merged before it appears,
        # we subtract _TOL, above, to make sure this doesn't happen
        appearances = np.argsort(heights)
        uf = DisjointSet()
        clusters_birth = {}
        clusters_died = {}
        clusters = []
        hind = 0
        mind = 0
        n_points = heights.shape[0]
        while True :
            while hind < n_points and heights[appearances[hind]] <= merges_heights[mind] :
                uf.add(appearances[hind])
                clusters_birth[appearances[hind]] = heights[appearances[hind]]
                hind += 1
            if hind == n_points :
                current_height = np.infty
            else :
                current_height = heights[appearances[hind]]
            while mind < merges_heights.shape[0] and merges_heights[mind] < current_height :
                xy = self._merges[mind]
                x, y = xy
                rx = uf.__getitem__(x)
                ry = uf.__getitem__(y)
                if rx not in clusters_died and ry not in clusters_died :
                    bx = clusters_birth[rx]
                    by = clusters_birth[ry]
                    if bx > merges_heights[mind] - threshold or by > merges_heights[mind] - threshold :
                        del clusters_birth[rx]
                        del clusters_birth[ry]
                        uf.merge(x,y)
                        uf.add(mind + n_points)
                        uf.merge(x,mind + n_points)
                        rxy = uf.__getitem__(x)
                        clusters_birth[rxy] = min(bx, by)
                    else :
                        # they both must die
                        if clusters_birth[rx] + threshold <= merges_heights[mind] :
                            clusters.append(list(uf.subset(x)))
                        if clusters_birth[ry] + threshold <= merges_heights[mind] :
                            clusters.append(list(uf.subset(y)))
                        uf.merge(x,y)
                        uf.add(mind + n_points)
                        uf.merge(x,mind + n_points)
                        rxy = uf.__getitem__(x)
                        clusters_died[rxy] = True
                elif rx in clusters_died and ry in clusters_died :
                    # both of them are dead
                    uf.merge(x,y)
                    uf.add(mind + n_points)
                    uf.merge(x,mind + n_points)
                    rxy = uf.__getitem__(x)
                    clusters_died[rxy] = True
                else :
                    if rx in clusters_died :
                        x, y = y, x
                        rx, ry = ry, rx
                    # ry already died and rx just died
                    if clusters_birth[rx] + threshold <= merges_heights[mind] :
                        clusters.append(list(uf.subset(x)))
                    uf.merge(x,y)
                    uf.add(mind + n_points)
                    uf.merge(x,mind + n_points)
                    rxy = uf.__getitem__(x)
                    clusters_died[rxy] = True
                mind += 1
            if mind == merges_heights.shape[0] :
                break
        death = np.inf
        #if self._covariant :
        #    death = np.inf
        #else :
        #    death = -self._minr
        for x in range(n_points) :
            rx = uf.__getitem__(x)
            if rx not in clusters_died :
                if clusters_birth[rx] + threshold <= death :
                    clusters.append(list(uf.subset(x)))
                clusters_died[rx] = True
        current_cluster = 0
        res = np.full(n_points, -1)
        for cl in clusters :
            for x in cl :
                if x < n_points :
                    res[x] = current_cluster
            current_cluster += 1
        return current_cluster, res

    def persistence_diagram(self) :
        # ti is the terminal index:
        # a point in the pd that never dies will have ti as its death index.
        #heights = self._heights.copy()
        heights = self._heights
        merges = self._merges
        merges_heights = self._merges_heights
        #covariant = self._covariant
        #if end == "infinity" :
        #    if covariant :
        #        ti = _INF
        #    else :
        #        ti = _TOL
        #else :
        #    if covariant :
        #        ti = self._maxr
        #    else :
        #        ti = self._minr
        ti = self._maxr
        num_points = heights.shape[0]
        num_merges = merges.shape[0]
        # initialize persistence diagram
        # in the end, pers_diag[i, 0] will be the birth,
        # and pers_diag[i, 1] will be the death
        # of the point represented by the datapoint i.
        # if pers_diag[i, :] = [-1, -1] at the end, we ignore it.
        pers_diag = np.empty(shape=(num_points, 2), dtype=np.float64)
        pers_diag.fill(-1)
        # initialize an array of cluster representatives
        # for the cluster i in the stepwise dendrogram,
        # cluster_reps[i] is a datapoint in that cluster
        cluster_reps = np.empty(shape=num_points + num_merges, dtype=np.int64)
        cluster_reps.fill(-1)
        # if the dendrogram is contravariant, 
        # we reindex by taking the reciprocal
        #if covariant == False:
        #    heights = np.reciprocal(heights)
        #    merges_heights[merges_heights < _TOL] = _TOL
        #    merges_heights[merges_heights > _INF] = _INF
        #    merges_heights = np.reciprocal(merges_heights)
        #    if ti <= _TOL:
        #        ti = _INF
        #    elif ti >= _INF :
        #        ti = _TOL
        #    else:
        #        ti = np.reciprocal(ti)
        for i in range(num_merges):
            cluster_0 = merges[i, 0]
            cluster_1 = merges[i, 1]
            # if both clusters are singletons
            if cluster_0 < num_points and cluster_1 < num_points:
                height_0 = heights[cluster_0]
                height_1 = heights[cluster_1]
                current_height = merges_heights[i]
                # if cluster_0 was just born, but cluster_1 was already alive
                if np.abs(height_0 - current_height) < _TOL and np.abs(height_1 - current_height) >= _TOL:
                    pers_diag[cluster_1, :] = [height_1, ti]
                    cluster_reps[num_points + i] = cluster_1
                # if cluster_1 was just born, but cluster_0 was already alive
                if np.abs(height_1 - current_height) < _TOL and np.abs(height_0 - current_height) >= _TOL:
                    pers_diag[cluster_0, :] = [height_0, ti]
                    cluster_reps[num_points + i] = cluster_0
                # if cluster_0 and cluster_1 were just born
                if np.abs(height_0 - current_height) < _TOL and np.abs(height_1 - current_height) < _TOL:
                    pers_diag[cluster_0, :] = [height_0, ti]
                    cluster_reps[num_points + i] = cluster_0
                # if cluster_0 and cluster_1 were both already alive
                if np.abs(height_0 - current_height) >= _TOL and np.abs(height_1 - current_height) >= _TOL:
                    # if cluster_1 is born first
                    if height_0 >= height_1:
                        pers_diag[cluster_0, :] = [height_0, current_height]
                        pers_diag[cluster_1, :] = [height_1, ti]
                        cluster_reps[num_points + i] = cluster_1
                    # if cluster_0 is born first
                    if height_0 < height_1:
                        pers_diag[cluster_0, :] = [height_0, ti]
                        pers_diag[cluster_1, :] = [height_1, current_height]
                        cluster_reps[num_points + i] = cluster_0                     
            # if cluster_0 is a singleton and cluster_1 is not
            if cluster_0 < num_points and cluster_1 >= num_points:
                height_0 = heights[cluster_0]
                rep_1 = cluster_reps[cluster_1]
                height_1 = pers_diag[rep_1, 0]
                current_height = merges_heights[i]
                # if cluster_0 was just born
                if np.abs(height_0 - current_height) < _TOL:
                    cluster_reps[num_points + i] = rep_1
                # if cluster_0 was already alive
                if np.abs(height_0 - current_height) >= _TOL:
                    # the singleton is younger than the cluster
                    if height_0 >= height_1:
                        pers_diag[cluster_0, :] = [height_0, current_height]
                        cluster_reps[num_points + i] = rep_1
                    # the singleton is older than the cluster
                    if height_0 < height_1:
                        pers_diag[cluster_0, :] = [height_0, ti]
                        pers_diag[rep_1, 1] = current_height
                        cluster_reps[num_points + i] = cluster_0
            # if cluster_1 is a singleton and cluster_0 is not
            if cluster_1 < num_points and cluster_0 >= num_points:
                height_1 = heights[cluster_1]
                rep_0 = cluster_reps[cluster_0]
                height_0 = pers_diag[rep_0, 0]
                current_height = merges_heights[i]
                # if cluster_1 was just born
                if np.abs(height_1 - current_height) < _TOL:
                    cluster_reps[num_points + i] = rep_0
                # if cluster_1 was already alive
                if np.abs(height_1 - current_height) >= _TOL:
                    # the singleton is younger than the cluster
                    if height_1 >= height_0:
                        pers_diag[cluster_1, :] = [height_1, current_height]
                        cluster_reps[num_points + i] = rep_0
                    # the singleton is older than the cluster
                    if height_1 < height_0:
                        pers_diag[cluster_1, :] = [height_1, ti]
                        pers_diag[rep_0, 1] = current_height
                        cluster_reps[num_points + i] = cluster_1
            # if neither cluster is a singleton
            if cluster_0 >= num_points and cluster_1 >= num_points:
                rep_0 = cluster_reps[cluster_0]
                height_0 = pers_diag[rep_0, 0]
                rep_1 = cluster_reps[cluster_1]
                height_1 = pers_diag[rep_1, 0]
                current_height = merges_heights[i]
                # cluster_0 is younger than cluster_1 
                if height_0 >= height_1:
                    pers_diag[rep_0, 1] = current_height
                    cluster_reps[num_points + i] = rep_1
                # cluster_1 is younger than cluster_0 
                if height_0 < height_1:
                    pers_diag[rep_1, 1] = current_height
                    cluster_reps[num_points + i] = rep_0
        # check if there are points in the dataset 
        # that never appeared in the dendrogram    
        appeared = np.zeros(shape=num_points, dtype=np.int64)
        for i in range(num_merges):
            cluster_0 = merges[i, 0]
            cluster_1 = merges[i, 1]
            if cluster_0 < num_points:
                appeared[cluster_0] = 1
            if cluster_1 < num_points:
                appeared[cluster_1] = 1
        for i in range(num_points):
            if appeared[i] == 0:
                pers_diag[i, :] = [heights[i], ti]
        # remove all rows from the persistence diagram that were never modified
        non_empty_indices = []
        for i in range(num_points):
            if pers_diag[i, 0] > -1:
                non_empty_indices.append(i)
        trimmed_pers_diag = np.empty(shape=(len(non_empty_indices), 2), dtype=np.float64)
        #representatives = np.empty(shape=(len(non_empty_indices), 1), dtype=np.int32)
        for i in range(len(non_empty_indices)):
            trimmed_pers_diag[i, 0] = pers_diag[non_empty_indices[i], 0]
            trimmed_pers_diag[i, 1] = pers_diag[non_empty_indices[i], 1]
        #if covariant == False:
        #    trimmed_pers_diag[:, [0, 1]] = np.reciprocal(trimmed_pers_diag[:, [0, 1]])
        trimmed_pers_diag[trimmed_pers_diag <= _TOL*2] = 0
        trimmed_pers_diag[trimmed_pers_diag >= _INF/2] = np.infty
        # set the death of the first born point to -infinity
        #if covariant == False and end == "infinity" :
        #    #trimmed_pers_diag[np.argmax(trimmed_pers_diag[:,0]),1] = -np.infty
        #    first_birth = np.max(trimmed_pers_diag[:,0])
        #    first_born = np.argwhere(trimmed_pers_diag[:,0] > first_birth - _TOL).flatten()
        #    # of the first born, take the last to die
        #    most_persistent = np.argmin(trimmed_pers_diag[first_born,1])
        #    index = first_born[most_persistent]
        #    trimmed_pers_diag[index,1] = -np.infty
        non_trivial_points = np.abs(trimmed_pers_diag[:,0] - trimmed_pers_diag[:,1]) > _TOL
        return trimmed_pers_diag[non_trivial_points], np.array(non_empty_indices)[non_trivial_points]

### PROMINENCE VINEYARD
class _ProminenceVineyard :
    
    def __init__(self, parameters, prominence_diagrams) :
        self._parameters = parameters
        self._prominence_diagrams = prominence_diagrams
        self._values = []

    def _vineyard_to_vines(self):
        times = self._parameters
        prominence_diagrams = self._prominence_diagrams
        num_vines = np.max([len(prom) for prom in prominence_diagrams])
        padded_prominence_diagrams = np.zeros((len(times),num_vines))
        for i in range(len(times)):
            padded_prominence_diagrams[i,:len(prominence_diagrams[i])] = prominence_diagrams[i]
        return [ (times,padded_prominence_diagrams[:,j]) for j in range(num_vines) ]

    def plot_prominence_vineyard(self, ax, color_firstn = 10, interpolate=True, areas=True, points=False):

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
        if color_firstn == None :
            colors = cscheme(np.linspace(0, 1, num_vines)[::-1])
        else :
            colors = list(cscheme(np.linspace(0, 1, color_firstn)[::-1]))
            last = colors[-1]
            colors.extend([last for _ in range(num_vines-color_firstn)])
        shm = StatusbarHoverManager(ax, "parameter", "prominence")
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
                    shm.add_artist_labels(artist, "vine " + str(i+1))
                if points:
                    artist = ax.plot(time_part,vine_part, "o", c="black")
                    shm.add_artist_labels(artist, "vine " + str(i+1))
                self._values.extend(vine_part)
