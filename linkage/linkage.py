import numpy as np
import scipy as sp
import random
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
#from scipy.spatial import KDTree
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import DisjointSet
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from linkage.plot import StatusbarHoverManager
from linkage.from_hdbscan._hdbscan_boruvka import KDTreeBoruvkaAlgorithm
#from linkage.from_hdbscan import *


TOL = 1e-8
INF = 1e15


### PERSISTABLE

class Persistable :
    def __init__(self, X, measure = None, kernel = 'square', maxk = None, leaf_size = 40, p = 2) :
        self.data = X
        self.p = p
        self.mpspace = MPSpace(X, 'minkowski', measure, leaf_size, self.p)
        self.mpspace.fit(maxk=maxk)
        self.connection_radius = self.mpspace.connection_radius()
        self.maxk = maxk
        self.kernel = kernel

    def parameter_selection(self, initial_k = 0.02, final_k = 0.2, n_parameters=50, color_firstn=10):
        parameters = np.logspace(np.log10(initial_k), np.log10(final_k), num=n_parameters)
        sks = [ (self.connection_radius, k) for k in parameters ]
        pds = self.mpspace.lambda_linkage_prominence_vineyard(sks)
        fig, ax = plt.subplots(figsize=(10,3))
        plt.xscale("log")
        plt.yscale("log")
        vineyard = ProminenceVineyard(parameters,pds)
        vineyard.plot_prominence_vineyard(ax, color_firstn=color_firstn)
        plt.ylim([np.quantile(np.array(vineyard.values),0.05),max(vineyard.values)])
        plt.show()

    def cluster(self,num_clusters,k,s=None,cluster_all=False,cluster_all_k=5):
        if s == None:
            s = self.connection_radius
        cl = self.mpspace.lambda_linkage(s,k).persistence_based_flattening(num_clusters)

        def postProcessing(dataset, labels, k) :
            neigh = KNeighborsClassifier(n_neighbors=k, p=self.p)
            neigh.fit(dataset[labels!=-1], labels[labels!=-1])
            res = labels.copy()
            res[labels==-1] = neigh.predict(dataset[labels==-1,:])
            return res
        
        if cluster_all:
            labels = postProcessing(self.data, cl[1], k=cluster_all_k)
            return cl[0], labels
        else:
            return cl



### GAMMA LINKAGE

class MPSpace :
    """Implements a finite metric probability space that can compute \
       its kernel density estimates"""

    POSSIBLE_KERNELS =  {'square', 'triangle', 'epanechnikov'}

    def __init__(self, X, metric = 'minkowski', measure = None, leaf_size = 40, p = 2) :
        # if metric = 'precomputed' then assumes that X is a distance matrix
        # to do: check that input is correct

        self.metric = metric
        self.p = p
        self.leaf_size = leaf_size

        self.size = X.shape[0]
        if measure is None :
            self.measure = np.full(self.size, 1./self.size)
            self.counting_measure = True

        self.dimension = X.shape[1]
        self.metric = metric
        if metric != 'precomputed' :
            self.points = X
        else :
            self.points = np.array(range(self.size))

        self.fit_on = None

        self.kernel = None
        self.fitted_nn = False
        self.fitted_density_estimates = False

        self.nn_distance = None
        self.nn_indices = None
        self.square_kernel_estimate = None
        self.delta = None
        self.kernel_estimate = None

        self.maxk = None
        self.maxs = None

        self.tol = TOL

        if metric == "minkowski":
            #self.tree = KDTree(X, leafsize=leaf_size)
            self.tree = KDTree(X, metric=metric, leaf_size=leaf_size, p = p)
        elif metric == 'precomputed':
            self.dist_mat = X
        else :
            raise Exception("Metric given is not supported.")


    def fit(self, maxk = None, kernel = 'square', fit_on = None) :
        self.fit_nn(maxk = maxk, fit_on = fit_on)
        self.fit_density_estimates(kernel = kernel)


    def fit_nn(self, maxk, fit_on) :
        # to do: check input
        if fit_on == None :
            fit_on = range(0,self.size)

        # fit_on can be just a number < 1
        if isinstance(fit_on, float) and fit_on < 1 :
            n_samples = int(self.size * fit_on)
            fit_on = random.sample(range(self.size),n_samples)
        # or > 1
        if isinstance(fit_on, int) and fit_on > 1 :
            n_samples = fit_on
            fit_on = random.sample(range(self.size),n_samples)

        self.fit_on = fit_on
        fit_on = self.points[fit_on]

        if maxk == None or maxk > self.size :
            maxk = self.size

        self.maxk = maxk
        
        if self.metric == 'minkowski' :
            #k_neighbors = self.tree.query(fit_on, k=self.maxk, p=self.p)
            k_neighbors = self.tree.query(\
                    fit_on, self.maxk, return_distance = True, sort_results = True,
                    dualtree = True, breadth_first = True)
            k_neighbors = (np.array(k_neighbors[1]),np.array(k_neighbors[0]))

            maxs_given_by_maxk = np.min(k_neighbors[1][:,-1])
            self.maxs = maxs_given_by_maxk
            neighbors = k_neighbors[0]
            nn_distance = k_neighbors[1]

        else :
            warnings.warn("For now, for distance matrix we assume maxk = number of points.")
            self.maxk = self.size
            self.maxs = 0
            neighbors = np.argsort(self.dist_mat)
            nn_distance = self.dist_mat[np.arange(len(self.dist_mat)), neighbors.transpose()].transpose()

        self.nn_indices = np.array(neighbors)
        self.nn_distance = np.array(nn_distance)
        self.fitted_nn = True


    def fit_density_estimates(self, kernel) :

        self.kernel = kernel
        self.fitted_density_estimates = True

        self.square_kernel_estimate = np.cumsum(self.measure[self.nn_indices], axis = 1)

        if kernel == 'square' :
            self.kernel_estimate = self.square_kernel_estimate
        else :
            with np.errstate(divide='ignore'):
                inv_width = np.where(self.nn_distance < self.tol, 0, np.divide(1.,self.nn_distance))
            if kernel == 'triangle' :
                self.delta = np.cumsum(self.measure[self.nn_indices] * self.nn_distance, axis = 1)
                self.kernel_estimate = self.square_kernel_estimate - inv_width * self.delta
            elif kernel == 'epanechnikov' :
                self.delta = np.cumsum(self.measure[self.nn_indices] * np.square(self.nn_distance), axis = 1)
                self.kernel_estimate = self.square_kernel_estimate - np.square(inv_width) * self.delta


    def kde_at_index_width(self, point_index, neighbor_index, width = None) :
        # to do: check input
        if width is None :
            width = self.nn_distance[point_index][neighbor_index]

        if self.kernel == 'square' :
            return self.square_kernel_estimate[point_index][neighbor_index]

        else :
            with np.errstate(divide='ignore'):
                inv_width = np.where(width < self.tol, 0, np.divide(1.,width))
   
            if self.kernel == 'triangle' :
                return self.square_kernel_estimate[point_index][neighbor_index] -\
                    inv_width * self.delta[point_index][neighbor_index]

            elif self.kernel == 'epanechnikov' :
                return self.square_kernel_estimate[point_index][neighbor_index] -\
                    np.square(inv_width) * self.delta[point_index][neighbor_index]
 

    def kde(self, point_index, width) :
        # to do: check input (in particular that the index is in bound)
        width = np.array(width)

        # find the index (resp. indices) of the neighbor (resp. neighbors)
        # whose distance is (left) closest to width (resp. each element of width)
        pos = np.searchsorted(self.nn_distance[point_index], width, side = 'right')
        pos -= 1

        # number of local neighbors of the point
        n_neighbors = len(self.nn_distance[point_index])

        # check if the k value we computed is exact or only a lower bound
        # (in that case, annotate it in the out_of_range list)
        if n_neighbors < self.size :
            if width.ndim == 1 :
                # two conditions needed for out of bound
                out_of_range = np.where(pos == n_neighbors-1, True, False)
                if self.maxs > self.tol :
                    out_of_range_ = np.where(width > self.maxs, True, False)
                    out_of_range = np.logical_and(out_of_range, out_of_range_)
            else :
                out_of_range = (pos == n_neighbors-1 and self.nn_distance[pos] > self.maxs)
        else :
            if width.ndim == 1 :
                out_of_range = np.full(len(width),False)
            else :
                out_of_range = False

        return self.kde_at_index_width(point_index,pos,width), out_of_range
           

    def core_distance(self, point_index, s0, k0) :

        def lazy_intersection(increasing, increasing2, f1) :
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
            i_indices.append(lazy_intersection(self.kernel_estimate[p], self.nn_distance[p], k_to_s))

        i_indices = np.array(i_indices)

        out_of_range = i_indices[:,1]
        if np.any(out_of_range) :
            # to do: better message for second condition
            warnings.warn("Don't have enough neighbors to properly compute core scale, or point takes too long to appear.")

        i_indices = i_indices[:,0]

        op = lambda p, i : np.where(k_to_s(self.kernel_estimate[p,i-1]) <= self.nn_distance[p,i],\
                k_to_s(self.kernel_estimate[p,i-1]),
                self.nn_distance[p,i])

        return np.where(i_indices == 0, 0, op(point_index,i_indices))




    def core_scale(self, point_index, gamma) :
        """Given a curve gamma (that takes an r and returns s,t,k) and a
        list of (indices of) points in the space, returns the r-time at which
        the points are born."""

        # point_index can be a list
        point_index = np.array(point_index)

        if gamma.s_component.is_constant :
            return self.core_scale_constant_s(point_index, gamma)
        elif gamma.k_component.is_constant :
            return self.core_scale_varying_s_constant_k(point_index, gamma)
        else :
            return self.core_scale_varying_s_k(point_index, gamma)


    def core_scale_constant_s(self, point_index, gamma) :
        s0 = gamma.s_component.func(gamma.minr)
        kde_s0 = np.vectorize(lambda i : self.kde(i,s0))
        kdes, out_of_range = kde_s0(point_index)

        if np.any(out_of_range) :
            warnings.warn("Don't have enough neighbors to properly calculate core scale.")

        return gamma.k_component.inverse(kdes)


    def core_scale_varying_s_constant_k(self, point_index, gamma) :

        k0 = gamma.k_component.func(gamma.minr)

        if k0 < TOL :
            return np.zeros((len(point_index)))

        if k0 > 1 :
            warnings.warn("The curve doesn't intersect the shadow.")
            zero = np.vectorize(lambda x : 0)
            return zero(point_index)

        if self.kernel == 'square' and self.counting_measure :
            # square kernel with couting measure and constant k
            i_indices = int(np.ceil(k0 * self.size)) - 1

            if i_indices + 1 > self.maxk :

                if self.maxk < self.size :
                    # to do: check that the boundary cases are correct here
                    out_of_range = np.where((i_indices + 1 >\
                            np.apply_along_axis(len,-1,self.nn_indices[point_index])) &\
                            (i_indices + 1 < self.size), True, False)
                    if np.any(out_of_range) :
                        warnings.warn("Don't have enough neighbors to properly compute core scale.")
        else :
            i_indices = []
            for p in point_index :
                i_indices.append(np.searchsorted(self.kernel_estimate[p],k0, side = 'left'))

            i_indices = np.array(i_indices)

            if self.maxk < self.size :
                out_of_range = np.where((i_indices >=\
                    np.apply_along_axis(len,-1,self.nn_indices[point_index])) &\
                    (np.apply_along_axis(len,-1,self.nn_indices[point_index]) < self.size), True, False)

                if np.any(out_of_range) :
                    warnings.warn("Don't have enough neighbors to properly compute core scale.")

        if self.kernel == 'square' :
            return gamma.s_component.inverse(self.nn_distance[(point_index, i_indices)])
        if self.kernel == 'triangle' :
            op = lambda p, i : np.divide(self.delta[p,i-1], (self.square_kernel_estimate[p,i-1] - k0))
        elif self.kernel == 'epanechnikov' :
            op = lambda p, i : np.sqrt(np.divide(self.delta[p,i-1], self.square_kernel_estimate[p,i-1] - k0))

        return gamma.s_component.inverse(np.where(i_indices == 0, 0, op(point_index,i_indices)))


    def core_scale_varying_s_k(self, point_index, gamma) :

        def lazy_intersection(increasing, increasing2, f2) :
            # find first occurence of increasing[i] >= f2(increasing2[i])
            first = 0
            last = len(increasing)-1

            if increasing[first] >= f2(increasing2[first]) :
                return first, False
            if increasing[last] < f2(increasing2[last]) :
                return last, True

            while first+1 < last :
                midpoint = (first + last)//2
                if increasing[midpoint] >= f2(increasing2[midpoint]) :
                    last = midpoint
                else:
                    first = midpoint

            return last, False


        k_s_inv = lambda d : gamma.k_component.func(gamma.s_component.func_inv(d))
        #k_s_inv = np.vectorize(lambda d : gamma.k_component.func(gamma.s_component.func_inv(d)))

        i_indices = []
        for p in point_index :
            i_indices.append(lazy_intersection(self.kernel_estimate[p], self.nn_distance[p], k_s_inv))
            #i_indices.append(lazy_intersection_2(self.kernel_estimate[p], self.nn_distance[p], k_s_inv))

        i_indices = np.array(i_indices)

        out_of_range = i_indices[:,1]
        if np.any(out_of_range) :
            # to do: better message for second condition
            warnings.warn("Don't have enough neighbors to properly compute core scale, or point takes too long to appear.")

        i_indices = i_indices[:,0]

        if self.kernel == 'square' :

            op = lambda p, i : np.where(self.kernel_estimate[p,i-1] >= k_s_inv(self.nn_distance[p,i]),\
                    gamma.s_component.func(gamma.k_component.func_inv(self.kernel_estimate[p,i-1])),
                    self.nn_distance[p,i])

            return gamma.s_component.inverse(np.where(i_indices == 0, 0, op(point_index,i_indices)))
        else :

            # to do: set tolerance so user can choose it, and handle nonconvergence in some controlled way
            op_ = lambda p, i : sp.optimize.brentq(lambda s : self.kde(p, s)[0] -\
                                       gamma.k_component.func(gamma.s_component.func_inv(s)),
                                   self.nn_distance[p,i-1], self.nn_distance[p,i], disp = True)

            op = lambda p, i : 0 if i == 0 else op_(p,i)

            return gamma.s_component.inverse(np.array(list(map(op, point_index, i_indices))))


    def lambda_linkage(self, s0, k0) :

        indices = np.arange(self.size)
        core_scales = self.core_distance(indices, s0, k0)
   
        sl = KDTreeBoruvkaAlgorithm(self.tree, core_scales, self.nn_indices, leaf_size=self.leaf_size // 3).spanning_tree()
        merges = sl[:,0:2].astype(int)
        merges_heights = sl[:,2]
        merges_heights[merges_heights <= TOL*2] = 0

        return HierarchicalClustering(self.points, True, core_scales, merges, merges_heights, 0, s0)



    def lambda_linkage_prominence_vineyard(self, sks) :

        def prominences(bd : np.array) -> np.array :
            return np.sort(np.abs(bd[:,0] - bd[:,1]))[::-1]

        prominence_diagrams = []
        
        for sk in sks :
            s0, k0 = sk
            hc = self.lambda_linkage(s0, k0)
            persistence_diagram = hc.PD()[0]
            prominence_diagram = prominences(persistence_diagram)
            prominence_diagrams.append(prominence_diagram)
            
        return prominence_diagrams


    def connection_radius(self,factor=1,percentile=1) :
        mst = KDTreeBoruvkaAlgorithm(self.tree, np.zeros(len(self.points)), self.nn_indices, leaf_size=self.leaf_size // 3).spanning_tree()
        return factor * np.quantile(mst[:,2],percentile)
    
        
class HierarchicalClustering :
    """Implements a hierarchical clustering of a dataset"""

    def __init__(self, X, covariant, heights, merges, merges_heights, minr, maxr) :
        self.points = X
        self.covariant = covariant
        #self.dend = dend
        self.merges = merges
        self.merges_heights = merges_heights
        self.heights = heights
        self.maxr = maxr
        self.minr = minr


    def persistence_based_flattening(self, num_clusters) :
        #if threshold == None and num_clusters == None :
        #    raise Exception("Either threshold or num_clusters must be given.")
        #if threshold != None and num_clusters != None :
        #    warnings.warn("Both threshold and num_clusters given, using threshold.")
        #elif threshold == None :

        bd = self.PD(end="infinity")[0]
        pers = np.abs(bd[:,0] - bd[:,1])
        if num_clusters >= bd.shape[0] :
            spers = np.sort(pers)
            threshold = spers[0] / 2
        else :
            spers = np.sort(pers)
            threshold = (spers[-num_clusters] + spers[-(num_clusters+1)])/2

        heights = self.heights.copy()
        merges_heights = self.merges_heights.copy()

        if not self.covariant :
            heights = -heights - TOL
            merges_heights = -merges_heights
        else :
            heights = heights - TOL

        # for numerical reasons, it may be that a point is merged before it appears,
        # we subtract TOL, above, to make sure this doesn't happen

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
                xy = self.merges[mind]
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
        #if self.covariant :
        #    death = np.inf
        #else :
        #    death = -self.minr
        
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
 


    def PD(self, end = None) :
        # ti is the terminal index:
        # a point in the pd that never dies will have ti as its death index.
        heights = self.heights.copy()
        merges = self.merges.copy()
        merges_heights = self.merges_heights.copy()

        covariant = self.covariant
        if end == "infinity" :
            if covariant :
                ti = INF
            else :
                ti = TOL
        else :
            if covariant :
                ti = self.maxr
            else :
                ti = self.minr

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
        if covariant == False:
            
            heights = np.reciprocal(heights)
            merges_heights[merges_heights < TOL] = TOL
            merges_heights[merges_heights > INF] = INF
            merges_heights = np.reciprocal(merges_heights)
            
            if ti <= TOL:
                ti = INF
            elif ti >= INF :
                ti = TOL
            else:
                ti = np.reciprocal(ti)
        
        for i in range(num_merges):
            
            cluster_0 = merges[i, 0]
            cluster_1 = merges[i, 1]
            
            # if both clusters are singletons
            if cluster_0 < num_points and cluster_1 < num_points:
                
                height_0 = heights[cluster_0]
                height_1 = heights[cluster_1]
                current_height = merges_heights[i]
                
                # if cluster_0 was just born, but cluster_1 was already alive
                if np.abs(height_0 - current_height) < TOL and np.abs(height_1 - current_height) >= TOL:
                    
                    pers_diag[cluster_1, :] = [height_1, ti]
                    cluster_reps[num_points + i] = cluster_1
                    
                # if cluster_1 was just born, but cluster_0 was already alive
                if np.abs(height_1 - current_height) < TOL and np.abs(height_0 - current_height) >= TOL:
                    
                    pers_diag[cluster_0, :] = [height_0, ti]
                    cluster_reps[num_points + i] = cluster_0
                    
                # if cluster_0 and cluster_1 were just born
                if np.abs(height_0 - current_height) < TOL and np.abs(height_1 - current_height) < TOL:
                    
                    pers_diag[cluster_0, :] = [height_0, ti]
                    cluster_reps[num_points + i] = cluster_0
                    
                # if cluster_0 and cluster_1 were both already alive
                if np.abs(height_0 - current_height) >= TOL and np.abs(height_1 - current_height) >= TOL:
                    
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
                if np.abs(height_0 - current_height) < TOL:
                    cluster_reps[num_points + i] = rep_1
                    
                # if cluster_0 was already alive
                if np.abs(height_0 - current_height) >= TOL:
                        
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
                if np.abs(height_1 - current_height) < TOL:
                    cluster_reps[num_points + i] = rep_0
                    
                # if cluster_1 was already alive
                if np.abs(height_1 - current_height) >= TOL:
                        
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
            
        if covariant == False:
            trimmed_pers_diag[:, [0, 1]] = np.reciprocal(trimmed_pers_diag[:, [0, 1]])
          
        trimmed_pers_diag[trimmed_pers_diag <= TOL*2] = 0
        trimmed_pers_diag[trimmed_pers_diag >= INF/2] = np.infty

        # set the death of the first born point to -infinity
        if covariant == False and end == "infinity" :
            
            #trimmed_pers_diag[np.argmax(trimmed_pers_diag[:,0]),1] = -np.infty
            
            first_birth = np.max(trimmed_pers_diag[:,0])
            
            first_born = np.argwhere(trimmed_pers_diag[:,0] > first_birth - TOL).flatten()
            
            # of the first born, take the last to die
            most_persistent = np.argmin(trimmed_pers_diag[first_born,1])
            
            index = first_born[most_persistent]
            trimmed_pers_diag[index,1] = -np.infty

        non_trivial_points = np.abs(trimmed_pers_diag[:,0] - trimmed_pers_diag[:,1]) > TOL

        return trimmed_pers_diag[non_trivial_points], np.array(non_empty_indices)[non_trivial_points]
    
    def measure_based_flattening(self, m):
        
        # compute poset of persistent clusters
        X = self.PC(min_m = m, max_m = m)
        
        return X.measure_based_flattening_PC(m = m)
    
    def measure_based_flattening_several_m(self, min_m, max_m):
        
        # compute poset of persistent clusters
        X = self.PC(min_m = min_m, max_m = max_m)
        
        # compute all measure-based flattenings
        num_points = self.heights.shape[0]
        labels = np.empty(shape=(max_m - min_m + 1, num_points), dtype=np.int64)
        ids = {}
        
        for m in range(min_m, max_m + 1):
            labels_m, ids_m = X.measure_based_flattening_PC(m = m, verbose = True)
            labels[m - min_m, :] = labels_m
            ids[m] = ids_m
            
        critical_m = [(min_m, len(ids[min_m]))]
            
        for m in range(min_m + 1, max_m + 1):
            
            if len(ids[m-1]) != len(ids[m]):
                critical_m.append((m, len(ids[m])))
            else:
                if set(ids[m-1]) != set(ids[m]):
                    critical_m.append((m, len(ids[m])))
            
        return labels, critical_m
            

    # Returns the PC of the hierarchical clustering, 
    # after the measure-based pruning with m=2.
    # From this, one can compute the measure-based flattening with any m
    # such that min_m <= m <= max_m.
    def PC(self, min_m, max_m):

        covariant = self.covariant
        if covariant :
            ti = self.maxr
        else :
            ti = self.minr

        num_points = self.heights.shape[0]
        num_merges = self.merges.shape[0]
        
        pc_of_points = np.empty(shape=num_points, dtype=np.int64)
        pc_of_points.fill(-1)
        
        X = PCTree(pers_clusters = {}, 
                   pc_of_points = pc_of_points,
                   min_m = min_m, max_m = max_m,
                   roots = [])
        
        # Initialize an array of cluster identifiers.
        # For the cluster i in the stepwise dendrogram,
        # cluster_ids[i] will be the identifier of the persistent cluster
        # to which i belongs.
        cluster_ids = np.empty(shape=num_merges, dtype=np.int64)
        cluster_ids.fill(-1)
        
        for i in range(num_merges):
            
            cluster_0 = self.merges[i, 0]
            cluster_1 = self.merges[i, 1]
            
            # if both clusters are singletons
            if cluster_0 < num_points and cluster_1 < num_points:
                
                # add persistent cluster to the PC            
                pers_cluster = PersCluster(identifier = i,
                                           index = self.merges_heights[i],
                                           min_m = min_m, max_m = max_m)
                
                X.pers_clusters[i] = pers_cluster
                
                # cluster i in the stepwise dendrogram belongs to
                # the persistent cluster i
                cluster_ids[i] = i
                
                # both singletons belong to the persistent cluster i
                pc_of_points[cluster_0] = i
                pc_of_points[cluster_1] = i
                
                
            # if cluster_0 is not a singleton and cluster_1 is a singleton
            if cluster_0 >= num_points and cluster_1 < num_points:
                
                # find the persistent cluster to which cluster_0 belongs
                ident = cluster_ids[cluster_0 - num_points]
                pc = X.pers_clusters[ident]
                
                current_index = self.merges_heights[i]
                
                # update the score of pc
                pc.update_score_summands(current_index)
                
                # update the index where pc was last visited
                pc.index = current_index
                
                # pc has increased in size, since cluster_1 was added
                pc.size += 1
                
                # cluster_1 belongs to pc
                pc_of_points[cluster_1] = ident
                
                # cluster i in the stepwise dendrogram belongs to
                # the persistent cluster ident
                cluster_ids[i] = ident
                
            # if cluster_1 is not a singleton and cluster_0 is a singleton
            if cluster_1 >= num_points and cluster_0 < num_points:
                
                # find the persistent cluster to which cluster_1 belongs
                ident = cluster_ids[cluster_1 - num_points]
                pc = X.pers_clusters[ident]
                
                current_index = self.merges_heights[i]
                
                # update the score of pc
                pc.update_score_summands(current_index)
                
                # update the index where pc was last visited
                pc.index = current_index
                
                # pc has increased in size, since cluster_0 was added
                pc.size += 1
                
                # cluster_0 belongs to pc
                pc_of_points[cluster_0] = ident
                
                # cluster i in the stepwise dendrogram belongs to
                # the persistent cluster ident
                cluster_ids[i] = ident
                
            # if both clusters are not singletons
            if cluster_0 >= num_points and cluster_1 >= num_points:
                
                # find the persistent cluster to which cluster_0 belongs
                ident_0 = cluster_ids[cluster_0 - num_points]
                pc_0 = X.pers_clusters[ident_0]
                
                # find the persistent cluster to which cluster_1 belongs
                ident_1 = cluster_ids[cluster_1 - num_points]
                pc_1 = X.pers_clusters[ident_1]
                
                current_index = self.merges_heights[i]
                
                # update the score of pc_0
                pc_0.update_score_summands(current_index)
                    
                # update the score of pc_1
                pc_1.update_score_summands(current_index)
                    
                # Since pc_0 and pc_1 have merged,
                # they create a child in X
            
                pers_cluster = PersCluster(identifier = i,
                                           parents = [ident_0, ident_1],
                                           size = pc_0.size + pc_1.size,
                                           index = current_index,
                                           min_m = min_m, max_m = max_m)
                
                X.pers_clusters[i] = pers_cluster
                
                pc_0.child = i
                pc_1.child = i
                
                # cluster i in the stepwise dendrogram belongs to
                # the persistent cluster i
                cluster_ids[i] = i
                
        # find the roots of the PC        
        for ident in X.pers_clusters.keys():
            if X.pers_clusters[ident].child == None:
                X.roots.append(ident)
                
        # we have to finish computing the scores of root elements
        
        current_index = ti
        
        for root in X.roots:

            # find the persistent cluster to which root belongs
            pc = X.pers_clusters[root]
            
            # update the score of pc
            pc.update_score_summands(current_index)
                
        return X


### CURVES

class ParametrizedInterval :

    def __init__(self, dom_min, dom_max, cod_min, cod_max, func, covariant, func_inv = None) :
        # to do: check input

        self.covariant = covariant

        self.dom_min = dom_min
        self.dom_max = dom_max
        self.cod_min = cod_min
        self.cod_max = cod_max
        
        self.func = func
        # could be None
        self.func_inv = func_inv
        if func_inv == None :
            self.is_constant = True
        else :
            self.is_constant = False


    def inverse(self, r) :
        # to do: we don't assume the curve is invertible
        # in the constant case self.cod_min == self.cod_max,
        # make sure that the conditions are mutually exclusive
        # and their union is everything

        condlist = [ r < self.cod_min, r >= self.cod_max ]

        if self.covariant :
            choicelist = [ self.dom_min, self.dom_max ]
        else :
            choicelist = [ self.dom_max, self.dom_min ]

        #to do: handle non-invertible curves better
        if self.func_inv == None :
            return np.select(condlist,choicelist, default = 0)
        else:
            return np.select(condlist,choicelist, default = self.func_inv(r))


    def linear(dom_min, dom_max, cod_min, cod_max, slope, intercept, slope_inv = None, intercept_inv = None) :

        def line(slope, intercept, r) :
            return slope * r + intercept

        dom_min = dom_min
        dom_max = dom_max
        cod_min = cod_min
        cod_max = cod_max

        func = lambda r : line(slope,intercept, r)
        if slope_inv != None :
            func_inv = lambda r : line(slope_inv,intercept_inv, r)
        else :
            func_inv = None

        if slope >= 0 :
            covariant = True
        else :
            covariant = False

        return ParametrizedInterval(dom_min, dom_max, cod_min, cod_max, func, covariant, func_inv = func_inv)


    def linear_increasing(dom_min, dom_max, cod_min, cod_max) :
        slope = (cod_max - cod_min)/(dom_max - dom_min)
        intercept = cod_min - slope * dom_min

        slope_inv = 1./slope
        intercept_inv = -intercept * slope_inv

        return ParametrizedInterval.linear(dom_min, dom_max, cod_min, cod_max,\
                slope, intercept, slope_inv = slope_inv, intercept_inv = intercept_inv)


    def linear_decreasing(dom_min, dom_max, cod_min, cod_max) :
        slope = (cod_min - cod_max)/(dom_max - dom_min)
        intercept = cod_max - slope * dom_min
        
        slope_inv = 1./slope
        intercept_inv = -intercept * slope_inv

        return ParametrizedInterval.linear(dom_min, dom_max, cod_min, cod_max,\
                slope, intercept, slope_inv = slope_inv, intercept_inv = intercept_inv)


    def constant(dom_min, dom_max, const, covariant) :
        def constant(const, r) :
            return const

        # to do: test
        dom_min = dom_min
        dom_max = dom_max
        cod_min = const
        cod_max = const

        func = lambda r : constant(const, r)

        return ParametrizedInterval(dom_min, dom_max, cod_min, cod_max, func, covariant)


    def identity() :
        def identity(r) :
            return r

        dom_min = 0
        dom_max = np.infty
        cod_min = 0
        cod_max = np.infty

        func = identity
        func_inv = func

        return ParametrizedInterval(dom_min, dom_max, cod_min, cod_max, func, True, func_inv = func_inv)


    def times(alpha) :
        def times(alpha,r) :
            return alpha * r

        dom_min = 0
        dom_max = np.infty
        cod_min = 0
        cod_max = np.infty

        func = lambda r : times(alpha, r)
        func_inv = lambda r : times(1./alpha,r)

        return ParametrizedInterval(dom_min, dom_max, cod_min, cod_max, func, True, func_inv = func_inv)


class GammaCurve :
    
    def __init__(self, s_component, t_component, k_component) :
        # to do: actually, it does make sense for both components to be constant
        if s_component.is_constant and k_component.is_constant :
            raise Exception("Both components shouldn't be constant.")
        if s_component.is_constant :
            self.covariant = not k_component.covariant
        else :
            self.covariant = s_component.covariant

        self.s_component = s_component
        self.t_component = t_component
        self.k_component = k_component
        #self.k_s_inv = None

        # to do: check that domains coincide (and min should be 0)
        self.maxr = s_component.dom_max
        self.minr = 0


    def linear_interpolator_alpha_k_indexed(k, s, alpha = 1) :
        k_component = ParametrizedInterval.linear_increasing(0,k,0,k)
        s_component = ParametrizedInterval.linear_decreasing(0,k,0,s)
        t_component = ParametrizedInterval.linear_decreasing(0,k,0,alpha * s)
        return GammaCurve(s_component,t_component,k_component)


    def linear_interpolator_alpha_s_indexed(k, s, alpha = 1) :
        k_component = ParametrizedInterval.linear_decreasing(0,s,0,k)
        s_component = ParametrizedInterval.linear_increasing(0,s,0,s)
        t_component = ParametrizedInterval.linear_increasing(0,s,0,alpha * s)
        return GammaCurve(s_component,t_component,k_component)


    def constant_k_alpha_s_indexed(k,alpha = 1, maxs = np.infty) :
        k_component = ParametrizedInterval.constant(0,maxs,k,False)
        s_component = ParametrizedInterval.identity()
        t_component = ParametrizedInterval.times(alpha)
        return GammaCurve(s_component,t_component,k_component)


    def constant_s_t_k_indexed(s,t,maxk = np.infty) :
        k_component = ParametrizedInterval.identity()
        s_component = ParametrizedInterval.constant(0,maxk,s,False)
        t_component = ParametrizedInterval.constant(0,maxk,t,False)
        return GammaCurve(s_component,t_component,k_component)


### PROMINENCE VINEYARD
    
class ProminenceVineyard :
    
    def __init__(self, parameters, prominence_diagrams) :
        self.parameters = parameters
        self.prominence_diagrams = prominence_diagrams
        # to do: delete these
        self.largest_gaps = []
        self.largest_gap_parameters = []
        self.values = []
        
    def vineyard_to_vines(self):
        times = self.parameters
        prominence_diagrams = self.prominence_diagrams
        num_vines = np.max([len(prom) for prom in prominence_diagrams])
        padded_prominence_diagrams = np.zeros((len(times),num_vines))
        for i in range(len(times)):
            padded_prominence_diagrams[i,:len(prominence_diagrams[i])] = prominence_diagrams[i]
    
        return [ (times,padded_prominence_diagrams[:,j]) for j in range(num_vines) ]
    
    def plot_prominence_vineyard(self, ax, color_firstn = 10, interpolate=True, areas=True, points=False):

        times = self.parameters
        prominence_diagrams = self.prominence_diagrams

        def vine_parts(times, prominences, tol = 1e-8):
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

        vines = self.vineyard_to_vines()
       
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
    
            for vine_part, time_part in vine_parts(times,vine) :
                if interpolate:
                    artist = ax.plot(time_part,vine_part, c="black")
                    shm.add_artist_labels(artist, "vine " + str(i+1))
                if points:
                    artist = ax.plot(time_part,vine_part, "o", c="black")
                    shm.add_artist_labels(artist, "vine " + str(i+1))
                self.values.extend(vine_part)
        
            


    def find_largest_gaps(self, max_ell):
        # why are we assigning these to object fields instead of just returning them?
        largest_gaps, largest_gap_parameters = self.gaps(max_ell)
        self.largest_gaps = largest_gaps
        self.largest_gap_parameters = largest_gap_parameters
        
    def gaps(self,max_ell):
        N = len(self.prominence_diagrams)
        largest_gaps = []
        largest_gap_parameters = []
        
        prominences = np.zeros(shape=(max_ell, N), dtype=np.float64)
        for ell in range(max_ell):
            for i in range(N):
                if self.prominence_diagrams[i].shape[0] > ell:
                    prominences[ell, i] = self.prominence_diagrams[i][ell]
                    
        for ell in range(max_ell-1):
            largest_gaps.append(np.amax(prominences[ell,:] - prominences[ell+1,:]))
            largest_gap_parameters.append(self.parameters[np.argmax(prominences[ell,:] - prominences[ell+1,:])])
            
        return largest_gaps, largest_gap_parameters
    
    def plot_largest_gaps(self, min_ell, max_ell):
        plt.xticks(ticks=range(min_ell, max_ell, 2))
        plt.bar(range(min_ell, max_ell), [self.largest_gaps[ell-1] for ell in range(min_ell, max_ell)])
        plt.show()
