import numpy as np
import scipy as sp
import random
import warnings
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree, BallTree
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


TOL = 1e-15
INF = 1e15

### GAMMA LINKAGE


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


class MPSpace :
    """Implements a finite metric probability space that can compute \
       its kernel density estimates"""

    POSSIBLE_KERNELS =  {'square', 'triangle', 'epanechnikov'}

    def __init__(self, X, metric = 'minkowski', measure = None, leaf_size = 40, p = 2) :
        # if metric = 'precomputed' then assumes that X is a distance matrix
        # to do: check that input is correct

        self.metric = metric
        self.p = p

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

        if metric in KDTree.valid_metrics :
            self.tree = KDTree(X, metric=metric, leaf_size=leaf_size, p = p)
        elif metric in BallTree.valid_metrics :
            self.tree = BallTree(X, metric=metric, leaf_size=leaf_size)
        elif metric == 'precomputed':
            self.dist_mat = X
        else :
            raise Exception("Metric given is not supported.")


    def fit(self, maxk = None, maxs = 0, kernel = 'square', fit_on = None) :
        self.fit_nn(maxk = maxk, maxs = maxs, fit_on = fit_on)
        self.fit_density_estimates(kernel = kernel)


    def fit_nn(self, maxk, maxs, fit_on) :
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

        if maxk == None :
            maxk = self.size
        if maxk == 1 and maxs == 0 :
            warnings.warn("Fitting with k = 1 and s = 0.")
        if maxk > self.size :
            warnings.warn("Trying to fit with k > |data set|. Changing to k = |data set|.")
            maxk = self.size

        self.maxs = maxs
        self.maxk = maxk
        
        if self.metric != 'precomputed' :
            # to do: check if dualtree or breadth_first set to False is faster
            k_neighbors = self.tree.query(\
                    fit_on, self.maxk, return_distance = True, sort_results = True,
                    dualtree = True, breadth_first = True)
            k_neighbors = (np.array(k_neighbors[1]),np.array(k_neighbors[0]))

            maxs_given_by_maxk = np.min(k_neighbors[1][:,-1])

            neighbors = []
            nn_distance = []

            if maxs < maxs_given_by_maxk :
                self.maxs = maxs_given_by_maxk
                neighbors = k_neighbors[0]
                nn_distance = k_neighbors[1]

            else :
                s_neighbors = self.tree.query_radius(\
                        fit_on, maxs, return_distance = True, sort_results = True)

                for i in range(len(fit_on)) :
                    # can this be done more efficiently at a lower level?
                    if len(k_neighbors[0][i]) >= len(s_neighbors[0][i]) :
                        neighbors.append(k_neighbors[0][i])
                        nn_distance.append(k_neighbors[1][i])
                    else :
                        neighbors.append(s_neighbors[0][i])
                        nn_distance.append(s_neighbors[1][i])
        else :
            warnings.warn("For now, for distance matrix we assume maxk = number of points.")
            self.maxk = self.size
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
           

    # to do: use multiple cores when have lots of point indices
    def core_scale(self, point_index, gamma) :
        """Given a curve gamma (that takes an r and returns s,t,k) and a
        list of (indices of) points in the space, returns the r-time at which
        the points are born."""

        # point_index can be a list
        point_index = np.array(point_index)

        if gamma.s_component.is_constant :
            #warnings.warn("s component is constant.")
            return self.core_scale_constant_s(point_index, gamma)

        elif gamma.k_component.is_constant :
            #warnings.warn("k component is constant.")
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
            #warnings.warn("k is constant and 0, output is just single-linkage.")
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

        k_s_inv = lambda d : gamma.k_component.func(gamma.s_component.func_inv(d))

        i_indices = []
        for p in point_index :
            i_indices.append(lazy_intersection(self.kernel_estimate[p], self.nn_distance[p], k_s_inv))

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


    def its_shadow(self, gridk = 1.1, grids = 1, n_bins = 250) :
        # to do: check input

        if not self.fitted_density_estimates :
            raise Exception("Must fit before computing shadow.")

        # create grid if not given explicitly
        if isinstance(gridk, float) or isinstance(gridk, int) :
            n_bins = n_bins
            max_grid_k = gridk
            gridk = np.array(range(0,n_bins))/(n_bins-1) * max_grid_k
            max_grid_s = grids
            grids = np.array(range(0,n_bins))/(n_bins-1) * max_grid_s

        shadow = np.zeros((len(gridk),len(grids)))

        mask = np.full((len(gridk),len(grids)),False)

        for i in range(len(self.fit_on)) :

            estimates, out_of_bounds = self.kde(i, grids)

            k_indices = np.searchsorted(gridk, estimates, side = 'left')
            k_indices -= 1

            shadow[(k_indices, range(0,len(k_indices)))] += self.measure[i]

            for s_index, b in enumerate(out_of_bounds) :
                if b :
                    mask[k_indices[s_index]][s_index] = True

        shadow = shadow[::-1].cumsum(axis = 0)[::-1]
        normalize_by = np.sum(self.measure[self.fit_on])
        shadow /= normalize_by

        mask = np.logical_or.accumulate(mask)

        return Shadow(gridk, grids, shadow, mask)


    def gamma_linkage(self, gamma, consistent = False, intrinsic_dim = 1) :
        covariant = gamma.covariant

        if self.metric == "precomputed" :
            sl_dist = self.dist_mat.copy()
        else :
            sl_dist = pairwise_distances(self.points, metric = self.metric, p = self.p)

        indices = np.arange(self.size)
        core_scales = self.core_scale(indices, gamma)

        sl_dist = gamma.t_component.inverse(sl_dist)

        if not covariant :
            sl_dist = np.minimum(sl_dist, core_scales)
            sl_dist = np.minimum(sl_dist.T,core_scales).T
            sl_dist[sl_dist < TOL] = TOL
            sl_dist = np.reciprocal(sl_dist)
        else :
            sl_dist = np.maximum(sl_dist, core_scales)
            sl_dist = np.maximum(sl_dist.T,core_scales).T
            sl_dist[sl_dist > INF] = INF

        sl = linkage(squareform(sl_dist, checks=False), 'single')

        merges = sl[:,0:2].astype(int)
        merges_heights = sl[:,2]
      
        if not covariant :
            merges_heights = np.reciprocal(merges_heights)

        merges_heights[merges_heights >= INF/2] = np.infty
        merges_heights[merges_heights <= TOL*2] = 0

        ret = HierarchicalClustering(self.points, covariant, core_scales, merges, merges_heights, gamma.minr, gamma.maxr)

        if consistent :
            d = intrinsic_dim
            if d == 1 :
                if self.kernel == "square" :
                    cons = 2
                elif self.kernel == "triangle" :
                    cons = 1
                elif self.kernel == "epanechnikov" :
                    cons = 4/3
            else :
                if self.kernel == "square" :
                    cons = (np.pi ** (d/2)) / math.gamma(d/2 + 1)
                elif self.kernel == "triangle" :
                    cons = (2 * np.pi**((d-1)/2))/(math.gamma((d-1)/2) * d * (d+1))
                elif self.kernel == "epanechnikov" :
                    cons = (2 * np.pi**((d-1)/2) * 2)/(math.gamma((d-1)/2) * d * (d+2))

            v_s = lambda s : cons * s**d

            inverse_rescaling = np.vectorize(lambda r : gamma.k_component.func(r) / v_s(gamma.s_component.func(r)))

            # new covariance is False
            ret.reindex(inverse_rescaling, 0, np.inf, False)

        return ret


    def connection_radius(self) :
        gamma = Gamma_curve.constant_k_alpha_s_indexed(0)
        return self.gamma_linkage(gamma).start_and_end()[1]


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


    def persistence_based_flattening(self, threshold = None, num_clusters = None) :
        if threshold == None and num_clusters == None :
            raise Exception("Either threshold or num_clusters must be given.")
        if threshold != None and num_clusters != None :
            warnings.warn("Both threshold and num_clusters given, using threshold.")
        elif threshold == None :
            bd = self.PD()[0]
            pers = np.abs(bd[:,0] - bd[:,1])
            if num_clusters + 1 > bd.shape[0] :
                threshold = 0
            else :
                threshold = np.sort(pers)[-(num_clusters+1)] + TOL


        appearances = np.argsort(self.heights)
        heights = self.heights.copy()

        merges_heights = self.merges_heights.copy()
        if not self.covariant :
            heights = -heights 
            merges_heights = -merges_heights

        uf = UnionFind()
        clusters_birth = {}
        #clusters_death = {}
        clusters_died = {}
        clusters = []
        hind = 0
        mind = 0
        n_points = heights.shape[0]
        while True :
            while hind < n_points and heights[appearances[hind]] <= merges_heights[mind] :
                uf.find(appearances[hind])
                
                clusters_birth[appearances[hind]] = heights[appearances[hind]]
                hind += 1

            if hind == n_points :
                current_height = np.infty
            else :
                current_height = heights[appearances[hind]]

            while mind < merges_heights.shape[0] and merges_heights[mind] < current_height :
                xy = self.merges[mind]
                x, y = xy
                rx = uf.find(x)
                ry = uf.find(y)

                if rx not in clusters_died and ry not in clusters_died :

                    bx = clusters_birth[rx]
                    by = clusters_birth[ry]

                    if bx > merges_heights[mind] - threshold or by > merges_heights[mind] - threshold :
                        del clusters_birth[rx]
                        del clusters_birth[ry]
                        uf.union(x,y)
                        uf.union(x,mind + n_points)
                        rxy = uf.find(x)
                        clusters_birth[rxy] = min(bx, by)

                    else :
                        # they both must die

                        if clusters_birth[rx] + threshold <= merges_heights[mind] :
                            clusters.append(uf.equivalence_class(x))
                        if clusters_birth[ry] + threshold <= merges_heights[mind] :
                            clusters.append(uf.equivalence_class(y))

                        uf.union(x,y)
                        uf.union(x,mind + n_points)
                        rxy = uf.find(x)
                        clusters_died[rxy] = True

                elif rx in clusters_died and ry in clusters_died :
                    # both of them are dead
                    uf.union(x,y)
                    uf.union(x,mind + n_points)
                    rxy = uf.find(x)
                    clusters_died[rxy] = True

                else :
                    if rx in clusters_died :
                        x, y = y, x
                        rx, ry = ry, rx
                    # ry already died and rx just died

                    if clusters_birth[rx] + threshold <= merges_heights[mind] :
                        clusters.append(uf.equivalence_class(x))

                    uf.union(x,y)
                    uf.union(x,mind + n_points)
                    rxy = uf.find(x)
                    clusters_died[rxy] = True

                mind += 1

            if mind == merges_heights.shape[0] :
                break


        if self.covariant :
            death = np.inf
        else :
            death = -self.minr
        
        for x in range(n_points) :
            rx = uf.find(x)
            if rx not in clusters_died :
                if clusters_birth[rx] + threshold <= death :
                    clusters.append(uf.equivalence_class(x))
                clusters_died[rx] = True

        current_cluster = 0

        res = np.full(n_points, -1)

        for cl in clusters :
            for x in cl :
                if x < n_points :
                    res[x] = current_cluster

            current_cluster += 1

        return current_cluster, res
 

    def reindex(self, inverse_rescaling, new_min, new_max, new_covariance) :
        self.minr = new_min
        self.maxr = new_max
        self.covariant = new_covariance
        self.merges_heights = inverse_rescaling(self.merges_heights)
        self.heights = inverse_rescaling(self.heights)


    def start_and_end(self) :
        #returns the first and last things that happen in the hierarchical clustering
        if self.covariant :
            return np.min(self.heights), np.max(self.merges_heights)
        else :
            return np.max(self.heights), np.min(self.merges_heights)


    def interleaving_distance(self, hc) :
        """Computes the interleaving distance between self and the given hierarchical clustering.\
            Assumes that self and the given hierarchical clustering are defined over the same set."""
        heights1 = self.heights
        heights2 = hc.heights

        merges1 = self.merges
        merges2 = hc.merges
        merges_heights1 = self.merges_heights.copy()
        merges_heights2 = hc.merges_heights.copy()
        if not self.covariant :
            merges_heights1 = - merges_heights1
            merges_heights2 = - merges_heights2
        nmerges1 = len(merges1)
        nmerges2 = len(merges2)

        npoints = len(self.heights)
        #to do: fail if different number of points

        dist = np.amax(np.abs(heights1 - heights2))

        # compute the assymetric interleaving distance from 1 to 2
        # to do: we assume that everything merges together right after the last thing that happens
        # maybe this should be done at the level of dendrograms?
        i = 0
        epsilon1 = dist
        uf1 = UnionFind()
        uf1_ = UnionFind()
        for xy,r,n in zip(merges1, merges_heights1, range(nmerges1)):
            x,y = xy
            uf1_.union(x,y)
            uf1_.union(x,n + npoints)
            while i < nmerges2 and merges_heights2[i] < r + epsilon1 :
                uf1.union(merges2[i,0], merges2[i,1])
                uf1.union(merges2[i,0], i + npoints)
                i = i + 1

            rx = uf1_.find(x)
            ry = uf1_.find(y)
            while i < nmerges2 and uf1.find(rx) != uf1.find(ry) :
                epsilon1 = merges_heights2[i] - r
                uf1.union(merges2[i,0], merges2[i,1])
                uf1.union(merges2[i,0], i + npoints)
                i = i + 1

        i = 0
        epsilon2 = epsilon1 
        uf2 = UnionFind()
        uf2_ = UnionFind()
        for xy,r in zip(merges2, merges_heights2):
            x,y = xy
            uf2_.union(x,y)
            uf2_.union(x,n + npoints)
            while i < nmerges1 and merges_heights1[i] < r + epsilon2 :
                uf2.union(merges1[i,0], merges1[i,1])
                uf2.union(merges1[i,0], i + npoints)
                i = i + 1

            rx = uf2_.find(x)
            ry = uf2_.find(y)
            while i < nmerges1 and uf2.find(rx) != uf2.find(ry) :
                epsilon2 = merges_heights1[i] - r
                uf2.union(merges1[i,0], merges1[i,1])
                uf2.union(merges1[i,0], i + npoints)
                i = i + 1

        return epsilon2


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

        if covariant == False and end == "infinity" :
            trimmed_pers_diag[np.argmax(trimmed_pers_diag[:,0]),1] = -np.infty

        non_trivial_points = np.abs(trimmed_pers_diag[:,0] - trimmed_pers_diag[:,1]) > TOL

        return trimmed_pers_diag[non_trivial_points], np.array(non_empty_indices)[non_trivial_points]


class Shadow :

    # returns an empty shadow
    def __init__(self, gridk, grids, matrix, mask) :
        self.gridk = gridk
        self.grids = grids
        self.matrix = matrix
        self.mask = mask 


### UNION FIND

class UnionFind:

    def __str__(self) :
        return 'par: ' + str(self.parent) + '\n' +\
               'rnk: ' + str(self.rank) + '\n' +\
               'siz: ' + str(self.size) + '\n' +\
               'rep: ' + str(self.representatives)


    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.size = {}
        self.representatives = set()
        self.next = {}


    def __copy__(self):
        new_uf = UnionFind()

        new_uf.parent = self.parent.copy()
        new_uf.rank = self.rank.copy()
        new_uf.size = self.size.copy()
        new_uf.representatives = self.representatives.copy()
        new_uf.next = self.next.copy()

        return new_uf


    def class_size(self, obj) :
        root = self.find(obj)
        return self.size[root]


    def class_representatives(self) :
        return self.representatives


    def insert_object(self, obj):
        if not obj in self.parent :
            self.parent[obj] = obj
            self.rank[obj] = 0
            self.size[obj] = 1
            self.representatives.add(obj)
            self.next[obj] = obj


    def find(self, obj):
        if not obj in self.parent :
            self.insert_object(obj)
            return obj

        if self.parent[obj] != obj :
            self.parent[obj] = self.find(self.parent[obj])
        return self.parent[obj]


    def union(self, obj1, obj2):
        root1 = self.find(obj1)
        root2 = self.find(obj2)

        if root1 == root2 :
            return

        if self.rank[obj1] < self.rank[obj2] :
            root1, root2 = root2, root1

        self.parent[root2] = root1
        self.size[root1] = self.size[root1] + self.size[root2]
        self.representatives.remove(root2)
        self.next[root1], self.next[root2] = self.next[root2], self.next[root1]

        if self.rank[root1] == self.rank[root2] :
            self.rank[root1] = self.rank[root1] + 1

    def equivalence_class(self, obj) :
        next_obj = self.next[obj]
        cl = [obj]
        while next_obj != obj :
            cl.append(next_obj)
            next_obj = self.next[next_obj]

        return cl



### CURVES

def line(slope, intercept, r) :
    return slope * r + intercept

def constant(const, r) :
    return const

def identity(r) :
    return r

def times(alpha,r) :
    return alpha * r

class Parametrized_interval :

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

        return Parametrized_interval(dom_min, dom_max, cod_min, cod_max, func, covariant, func_inv = func_inv)


    def linear_increasing(dom_min, dom_max, cod_min, cod_max) :
        slope = (cod_max - cod_min)/(dom_max - dom_min)
        intercept = cod_min - slope * dom_min

        slope_inv = 1./slope
        intercept_inv = -intercept * slope_inv

        return Parametrized_interval.linear(dom_min, dom_max, cod_min, cod_max,\
                slope, intercept, slope_inv = slope_inv, intercept_inv = intercept_inv)


    def linear_decreasing(dom_min, dom_max, cod_min, cod_max) :
        slope = (cod_min - cod_max)/(dom_max - dom_min)
        intercept = cod_max - slope * dom_min
        
        slope_inv = 1./slope
        intercept_inv = -intercept * slope_inv

        return Parametrized_interval.linear(dom_min, dom_max, cod_min, cod_max,\
                slope, intercept, slope_inv = slope_inv, intercept_inv = intercept_inv)


    def constant(dom_min, dom_max, const, covariant) :
        # to do: test
        dom_min = dom_min
        dom_max = dom_max
        cod_min = const
        cod_max = const

        func = lambda r : constant(const, r)

        return Parametrized_interval(dom_min, dom_max, cod_min, cod_max, func, covariant)


    def identity() :
        dom_min = 0
        dom_max = np.infty
        cod_min = 0
        cod_max = np.infty

        func = identity
        func_inv = func

        return Parametrized_interval(dom_min, dom_max, cod_min, cod_max, func, True, func_inv = func_inv)


    def times(alpha) :
        dom_min = 0
        dom_max = np.infty
        cod_min = 0
        cod_max = np.infty

        func = lambda r : times(alpha, r)
        func_inv = lambda r : times(1./alpha,r)

        return Parametrized_interval(dom_min, dom_max, cod_min, cod_max, func, True, func_inv = func_inv)


class Gamma_curve :
    
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
        k_component = Parametrized_interval.linear_increasing(0,k,0,k)
        s_component = Parametrized_interval.linear_decreasing(0,k,0,s)
        t_component = Parametrized_interval.linear_decreasing(0,k,0,alpha * s)
        return Gamma_curve(s_component,t_component,k_component)


    def linear_interpolator_alpha_s_indexed(k, s, alpha = 1) :
        k_component = Parametrized_interval.linear_decreasing(0,s,0,k)
        s_component = Parametrized_interval.linear_increasing(0,s,0,s)
        t_component = Parametrized_interval.linear_increasing(0,s,0,alpha * s)
        return Gamma_curve(s_component,t_component,k_component)


    def constant_k_alpha_s_indexed(k,alpha = 1, maxs = np.infty) :
        k_component = Parametrized_interval.constant(0,maxs,k,False)
        s_component = Parametrized_interval.identity()
        t_component = Parametrized_interval.times(alpha)
        return Gamma_curve(s_component,t_component,k_component)


    def constant_s_t_k_indexed(s,t,maxk = np.infty) :
        k_component = Parametrized_interval.identity()
        s_component = Parametrized_interval.constant(0,maxk,s,False)
        t_component = Parametrized_interval.constant(0,maxk,t,False)
        return Gamma_curve(s_component,t_component,k_component)


### PLOTTING SHADOWS

def latex_float(f):
    # https://stackoverflow.com/a/13490601/2171328
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        #return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
        return r"{0}e{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def ticks_in_scientific_notation(tks) :
    #return [ "$" + latex_float(t) + "$" for t in tks]
    return [ latex_float(t) for t in tks]


def plot_shadow(shadow, n_ticks = 11, n_shadows = 10, maxt = None, gammas = None, n_samples_curve_domain = 10000, h_size = 10, v_size = 5) :

    #sns.set(rc={'text.usetex': True})

    # set size of final picture
    sns.set(rc={'figure.figsize':(h_size,v_size)})

    # draw heat map
    ax = sns.heatmap(np.flip(shadow.matrix,0), cmap = sns.color_palette("Blues", n_shadows), mask = np.flip(shadow.mask,0), rasterized=True)
    ax.set_facecolor('Grey')
    old_xticks = ax.get_xticks()
    new_xticks = np.linspace(np.min(old_xticks), np.max(old_xticks), n_ticks)
    new_xlabels = ticks_in_scientific_notation(np.linspace(shadow.grids[0], shadow.grids[-1], n_ticks))
    ax.set_xticks(new_xticks)
    ax.set_xticklabels(new_xlabels)

    old_yticks = ax.get_yticks()
    new_yticks = np.linspace(np.min(old_yticks), np.max(old_yticks), n_ticks)
    new_ylabels = ticks_in_scientific_notation(np.linspace(shadow.gridk[-1], shadow.gridk[0], n_ticks))
    ax.set_yticks(new_yticks)
    ax.set_yticklabels(new_ylabels, rotation = "horizontal")

    # plot t at which everything gets merged together if given as parameter
    if maxt != None :
        plt.axvline(x = maxt * len(shadow.grids)/shadow.grids[-1])
    
    # plot gamma if given as a parameter
    if gammas != None :
    
        mk = shadow.gridk[-1]
        ms = shadow.grids[-1]
    
        for gamma in gammas :
            # to do: there are more cases to consider in theory
            if gamma.maxr == np.infty :
                if gamma.s_component.is_constant :
                    k_component_func = np.vectorize(gamma.k_component.func)
                    s_component_func = np.vectorize(gamma.s_component.func)
                    maxr_graph = gamma.k_component.func_inv(mk)
                else :
                    k_component_func = np.vectorize(lambda x : gamma.k_component.func(gamma.s_component.func_inv(x)))
                    s_component_func = np.vectorize(lambda x : x)
                    maxr_graph = ms
            else :
                maxr_graph = gamma.maxr
                k_component_func = np.vectorize(gamma.k_component.func)
                s_component_func = np.vectorize(gamma.s_component.func)
                
            sample_domain = np.linspace(0, maxr_graph, n_samples_curve_domain)
        
            k_bins = len(shadow.gridk)
            s_bins = len(shadow.grids)
        
            scalex = lambda r : s_bins/ms * r
            scaley = lambda r : -k_bins/mk * r + k_bins
    
            x_coord = scalex(s_component_func(sample_domain))
            y_coord = scaley(k_component_func(sample_domain))
   
            ax.plot(x_coord, y_coord, '--k')

    return ax
