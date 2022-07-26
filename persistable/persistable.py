# Authors: Luis Scoccola and Alexander Rolle
# License: 3-clause BSD

from operator import ne
from turtle import color
from .plot import PersistablePlot
from .prominence_vineyard import ProminenceVineyard
from .borrowed._hdbscan_boruvka import (
    KDTreeBoruvkaAlgorithm,
    BallTreeBoruvkaAlgorithm,
)
from .borrowed.prim_mst import mst_linkage_core_vector
from .borrowed.dense_mst import stepwise_dendrogram_with_core_distances
from .borrowed.dist_metrics import DistanceMetric
from .aux import lazy_intersection
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import KDTree, BallTree
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import DisjointSet
from scipy.stats import mode
from joblib import Parallel, delayed


_TOL = 1e-08
_DEFAULT_FINAL_K = 0.2


class Persistable:
    def __init__(
        self,
        X,
        metric="minkowski",
        measure=None,
        max_neighbors=None,
        leaf_size: int = 40,
        **kwargs
    ):
        # keep dataset
        self._data = X
        # if metric is minkowski but no p was passed, assume p = 2
        if metric == "minkowski" and "p" not in kwargs:
            kwargs["p"] = 2
        # if no measure was passed, assume normalized counting measure
        if measure is None:
            measure = np.full(X.shape[0], 1.0 / X.shape[0])
        self._mpspace = _MetricProbabilitySpace(X, metric, measure, leaf_size, **kwargs)
        # if no max_neighbors for fitting mpspace was passed, compute a reasonable one
        if max_neighbors is None:
            if X.shape[0] < 100:
                max_neighbors = X.shape[0]
            else:
                max_neighbors = min(int(np.log10(X.shape[0])) * 100, X.shape[0])
        else:
            max_neighbors = min(max_neighbors, X.shape[0])
        # keep max_k (normalized max_neighbors)
        self._maxk = max_neighbors / X.shape[0]
        # fit the mpspace
        self._mpspace.fit(max_neighbors)
        default_percentile = 0.95
        # compute and keep robust connection radius
        self._connection_radius = self._mpspace.connection_radius(default_percentile)

        ## to be passed/computed later:
        # user-selected bounds for the prominence vineyard
        self._vineyard_parameter_bounds = {}
        # user-selected start and end for a line
        self._line_parameters = None
        # user-selected number of clusters
        self._n_clusters = None
        # the computed prominence vineyard
        self._vineyard = None
        # the PersistablePlot object implementing the GUI
        self._plot = None

    def quick_cluster(
        self,
        n_neighbors: int = 30,
        n_clusters_range=[3, 15],
        extend_clustering_by_hill_climbing=True,
        n_iterations_extend_cluster=5,
        n_neighbors_extend_cluster=10,
    ):
        k = n_neighbors / self._mpspace._size
        s = self._connection_radius * 2
        hc = self._mpspace.lambda_linkage([0, k], [s, 0])
        pd = hc.persistence_diagram()

        def _prominences(bd):
            return np.sort(np.abs(bd[:, 0] - bd[:, 1]))[::-1]

        proms = _prominences(pd)
        if n_clusters_range[1] >= len(proms):
            return self.cluster(n_clusters_range[1], [0, k], [s, 0])
        logproms = np.log(proms)
        # print("logproms")
        # print(logproms)
        peaks = logproms[:-1] - logproms[1:]
        # print("peaks")
        # print(peaks)
        min_clust = n_clusters_range[0] - 1
        max_clust = n_clusters_range[1] - 1
        # print("highest peak")
        # print(peaks[min_clust:max_clust][np.argmax(peaks[min_clust:max_clust])])
        num_clust = np.argmax(peaks[min_clust:max_clust]) + min_clust + 1
        # print(num_clust)
        return self.cluster(
            num_clust,
            [0, k],
            [s, 0],
            extend_clustering_by_hill_climbing=extend_clustering_by_hill_climbing,
            n_iterations_extend_cluster=n_iterations_extend_cluster,
            n_neighbors_extend_cluster=n_neighbors_extend_cluster,
        )

    def cluster(
        self,
        n_clusters=None,
        start=None,
        end=None,
        extend_clustering_by_hill_climbing=True,
        n_iterations_extend_cluster=5,
        n_neighbors_extend_cluster=10,
    ):
        if start is None:
            if self._line_parameters is None:
                raise Exception("No parameters for the line were given!")
            else:
                start, end = self._line_parameters
                n_clusters = self._n_clusters
        else:
            start, end = np.array(start), np.array(end)
            if start.shape != (2,) or end.shape != (2,):
                raise Exception("start and end must both be points on the plane.")
        if n_clusters <= 1:
            raise Exception("n_clusters must be greater than 1.")
        hc = self._mpspace.lambda_linkage(start, end)
        bd = hc.persistence_diagram()
        pers = np.abs(bd[:, 0] - bd[:, 1])
        # TODO: use sort from largest to smallest and make the logic below simpler
        spers = np.sort(pers)
        if n_clusters >= bd.shape[0]:
            if n_clusters > bd.shape[0]:
                warnings.warn(
                    "n_clusters is larger than the number of gaps, using n_clusters = number of gaps."
                )
            threshold = spers[0] / 2
        else:
            if np.abs(spers[-n_clusters] - spers[-(n_clusters + 1)]) < _TOL:
                warnings.warn(
                    "The gap selected is too small to produce a reliable clustering."
                )
            threshold = (spers[-n_clusters] + spers[-(n_clusters + 1)]) / 2
        cl = hc.persistence_based_flattening(threshold)
        if extend_clustering_by_hill_climbing:
            cl = self._mpspace.extend_clustering_by_hill_climbing(
                cl, n_iterations_extend_cluster, n_neighbors_extend_cluster
            )
        return cl

    def parameter_selection(
        self,
        max_dim=20,
        max_k=None,
        bounds_s=None,
        granularity=50,
        n_jobs=4,
        # colormap="binary",
    ):
        self._plot = PersistablePlot(
            self.update_vineyard_parameter_bounds,
            self.update_line_parameters,
            self.prominence_vineyard,
        )
        ss, ks, max_dim, hf = self.hilbert_function(
            max_dim=max_dim,
            max_k=max_k,
            bounds_s=bounds_s,
            granularity=granularity,
            n_jobs=n_jobs,
        )
        self._plot.plot_hilbert_function(ss, ks, max_dim, hf)

    def prominence_vineyard(
        self,
        start_end1=None,
        start_end2=None,
        n_parameters=50,
        first_n_vines=20,
    ):
        if start_end1 is None or start_end2 is None:
            if len(self._vineyard_parameter_bounds.values()) < 4:
                raise Exception("No parameters chosen!")
            else:
                start1 = self._vineyard_parameter_bounds["start1"]
                end1 = self._vineyard_parameter_bounds["end1"]
                start2 = self._vineyard_parameter_bounds["start2"]
                end2 = self._vineyard_parameter_bounds["end2"]
        else:
            start1, end1 = start_end1
            start2, end2 = start_end2
        if (
            start1[0] >= end1[0]
            or start1[1] <= end1[1]
            or start2[0] >= end2[0]
            or start2[1] <= end2[1]
        ):
            raise Exception("Chosen parameters will result in non-monotonic lines!")
        starts = list(
            zip(
                np.linspace(start1[0], start2[0], n_parameters),
                np.linspace(start1[1], start2[1], n_parameters),
            )
        )
        ends = list(
            zip(
                np.linspace(end1[0], end2[0], n_parameters),
                np.linspace(end1[1], end2[1], n_parameters),
            )
        )
        startends = list(zip(starts, ends))
        pds = self._mpspace.lambda_linkage_prominence_vineyard(startends)
        self._vineyard = ProminenceVineyard(startends, pds, firstn=first_n_vines)
        return self._vineyard

    def hilbert_function(
        self,
        max_dim=20,
        max_k=None,
        bounds_s=None,
        granularity=50,
        n_jobs=4,
        # colormap="binary",
    ):
        if max_k is None:
            max_k = self._maxk
        elif max_k > self._maxk:
            max_k = min(max_k, self._maxk)
            warnings.warn(
                "Not enough neighbors to compute chosen max_k, using max_k="
                + str(max_k)
                + " instead."
            )
        if bounds_s is None:
            first_s = self._connection_radius / 5
            last_s = self._connection_radius * 2
        else:
            first_s = bounds_s[0]
            last_s = bounds_s[1]
        # how many more ss than ks (note that getting more ss is very cheap)
        more_s_than_k = 10
        ss = np.linspace(
            first_s,
            last_s,
            granularity * more_s_than_k,
        )
        ks = np.linspace(0, max_k, granularity)
        hf = self._mpspace.hilbert_function(ks, ss, n_jobs=n_jobs)
        return ss, ks, max_dim, hf
        # self._plot.plot_hilbert_function(ss, ks, max_dim, hf, colormap=colormap)

    def update_line_parameters(self, gap, line_index):
        self._line_parameters = self._vineyard._parameters[line_index]
        self._n_clusters = gap
        return self._line_parameters, gap

    def update_vineyard_parameter_bounds(self, point):
        if "start1" not in self._vineyard_parameter_bounds:
            self._vineyard_parameter_bounds["start1"] = point
        elif "end1" not in self._vineyard_parameter_bounds:
            st1 = self._vineyard_parameter_bounds["start1"]
            if point[0] < st1[0] or point[1] > st1[1]:
                return self._vineyard_parameter_bounds
            self._vineyard_parameter_bounds["end1"] = point
        elif "start2" not in self._vineyard_parameter_bounds:
            self._vineyard_parameter_bounds["start2"] = point
        elif "end2" not in self._vineyard_parameter_bounds:
            st2 = self._vineyard_parameter_bounds["start2"]
            if point[0] < st2[0] or point[1] > st2[1]:
                return self._vineyard_parameter_bounds
            self._vineyard_parameter_bounds["end2"] = point
        else:
            self._vineyard_parameter_bounds = {}
            self.update_vineyard_parameter_bounds(point)
        return self._vineyard_parameter_bounds

    def persistence_diagram(self, s0, k0):
        hc = self._mpspace.lambda_linkage([0, k0], [s0, 0])
        return hc.persistence_diagram()


class _MetricProbabilitySpace:
    """Implements a finite metric probability space that can compute \
       its kernel density estimates"""

    def __init__(self, X, metric, measure, leaf_size=40, **kwargs):
        self._metric = metric
        self._kwargs = kwargs
        self._leaf_size = leaf_size
        self._size = X.shape[0]
        self._measure = measure
        self._dimension = X.shape[1]
        self._metric = metric
        if metric != "precomputed":
            self._points = X
        else:
            self._points = np.array(range(self._size))
        self._fitted_nn = False
        self._fitted_density_estimates = False
        self._nn_distance = None
        self._nn_indices = None
        self._kernel_estimate = None
        self._max_neighbors = None
        self._maxs = None
        self._tol = _TOL
        if metric in KDTree.valid_metrics:
            self._tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
        elif metric in BallTree.valid_metrics:
            self._tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
        elif metric == "precomputed":
            self._dist_mat = X
        else:
            raise Exception("Metric given is not supported.")

    def fit(self, max_neighbors):
        self.fit_nn(max_neighbors)
        self.fit_density_estimates()

    def fit_nn(self, max_neighbors):
        self._max_neighbors = max_neighbors
        if self._metric in BallTree.valid_metrics + KDTree.valid_metrics:
            k_neighbors = self._tree.query(
                self._points,
                self._max_neighbors,
                return_distance=True,
                sort_results=True,
                dualtree=True,
                breadth_first=True,
            )
            k_neighbors = (np.array(k_neighbors[1]), np.array(k_neighbors[0]))
            maxs_given_by_max_neighbors = np.min(k_neighbors[1][:, -1])
            self._maxs = maxs_given_by_max_neighbors
            neighbors = k_neighbors[0]
            _nn_distance = k_neighbors[1]
        else:
            self._max_neighbors = self._size
            self._maxs = 0
            neighbors = np.argsort(self._dist_mat)
            _nn_distance = self._dist_mat[
                np.arange(len(self._dist_mat)), neighbors.transpose()
            ].transpose()
        self._nn_indices = np.array(neighbors)
        self._nn_distance = np.array(_nn_distance)
        self._fitted_nn = True

    def fit_density_estimates(self):
        self._fitted_density_estimates = True
        self._kernel_estimate = np.cumsum(self._measure[self._nn_indices], axis=1)

    def core_distance(self, point_index, s_intercept, k_intercept):
        i_indices = []
        if s_intercept != np.inf:
            mu = s_intercept / k_intercept
            k_to_s = lambda y: s_intercept - mu * y
            for p in point_index:
                i_indices.append(
                    lazy_intersection(
                        self._kernel_estimate[p],
                        self._nn_distance[p],
                        s_intercept,
                        k_intercept,
                    )
                )
            i_indices = np.array(i_indices)
            out_of_range = i_indices[:, 1]
            if np.any(out_of_range):
                warnings.warn(
                    "Don't have enough neighbors to properly compute core scale, or point takes too long to appear."
                )
            i_indices = i_indices[:, 0]
            op = lambda p, i: np.where(
                k_to_s(self._kernel_estimate[p, i - 1]) <= self._nn_distance[p, i],
                k_to_s(self._kernel_estimate[p, i - 1]),
                self._nn_distance[p, i],
            )
            return np.where(i_indices == 0, 0, op(point_index, i_indices))
        else:
            for p in point_index:
                i_indices.append(
                    np.searchsorted(self._kernel_estimate[p], k_intercept, side="left")
                )
            i_indices = np.array(i_indices)
            if self._max_neighbors < self._size:
                out_of_range = np.where(
                    (
                        i_indices
                        >= np.apply_along_axis(len, -1, self._nn_indices[point_index])
                    )
                    & (
                        np.apply_along_axis(len, -1, self._nn_indices[point_index])
                        < self._size
                    ),
                    True,
                    False,
                )
                if np.any(out_of_range):
                    warnings.warn(
                        "Don't have enough neighbors to properly compute core scale."
                    )
            return self._nn_distance[(point_index, i_indices)]

    def lambda_linkage(self, start, end):
        if start[0] > end[0] or start[1] < end[1]:
            raise Exception("Lambda linkage parameters do not give a monotonic line!")

        def _startend_to_intercepts(start, end):
            if end[0] == np.infty:
                k_intercept = start[1]
                s_intercept = np.infty
            else:
                slope = (end[1] - start[1]) / (end[0] - start[0])
                k_intercept = -start[0] * slope + start[1]
                s_intercept = -k_intercept / slope
            return s_intercept, k_intercept

        hc_start = start[0]
        hc_end = end[0]
        indices = np.arange(self._size)
        s_intercept, k_intercept = _startend_to_intercepts(start, end)
        core_distances = self.core_distance(indices, s_intercept, k_intercept)
        core_distances = np.minimum(hc_end, core_distances)
        core_distances = np.maximum(hc_start, core_distances)
        if self._metric in KDTree.valid_metrics:
            if self._dimension > 60:
                X = self._points
                if not X.flags["C_CONTIGUOUS"]:
                    X = np.array(X, dtype=np.double, order="C")
                dist_metric = DistanceMetric.get_metric(self._metric, **self._kwargs)
                sl = mst_linkage_core_vector(X, core_distances, dist_metric)
            else:
                sl = KDTreeBoruvkaAlgorithm(
                    self._tree,
                    core_distances,
                    self._nn_indices,
                    leaf_size=self._leaf_size // 3,
                    metric=self._metric,
                    # p=self._p,
                    **self._kwargs
                ).spanning_tree()
        elif self._metric in BallTree.valid_metrics:
            if self._dimension > 60:
                X = self._points
                if not X.flags["C_CONTIGUOUS"]:
                    X = np.array(X, dtype=np.double, order="C")
                dist_metric = DistanceMetric.get_metric(self._metric, **self._kwargs)
                sl = mst_linkage_core_vector(X, core_distances, dist_metric)
            else:
                sl = BallTreeBoruvkaAlgorithm(
                    self._tree,
                    core_distances,
                    self._nn_indices,
                    leaf_size=self._leaf_size // 3,
                    metric=self._metric,
                    **self._kwargs
                ).spanning_tree()
        else:
            sl = stepwise_dendrogram_with_core_distances(
                self._size, self._dist_mat, core_distances
            )
        merges = sl[:, 0:2].astype(int)
        merges_heights = np.minimum(hc_end, sl[:, 2])
        merges_heights = np.maximum(hc_start, sl[:, 2])
        return _HierarchicalClustering(
            core_distances, merges, merges_heights, hc_start, hc_end
        )

    def lambda_linkage_vineyard(self, startends, n_jobs, tol=_TOL):
        run_in_parallel = lambda startend: self.lambda_linkage(
            startend[0], startend[1]
        ).persistence_diagram(tol=tol)
        # return [run_in_parallel(startend) for startend in startends]
        return Parallel(n_jobs=n_jobs)(
            delayed(run_in_parallel)(startend) for startend in startends
        )

    def lambda_linkage_prominence_vineyard(self, startends, n_jobs=4, tol=_TOL):
        def _prominences(bd):
            return np.sort(np.abs(bd[:, 0] - bd[:, 1]))[::-1]

        pds = self.lambda_linkage_vineyard(startends, n_jobs, tol=tol)
        return [_prominences(pd) for pd in pds]

    def hilbert_function(self, ks, ss, n_jobs, tol=_TOL):
        n_s = len(ss)
        n_k = len(ks)
        tol = ss[1] - ss[0]
        startends = [[[0, k], [np.infty, k]] for k in ks[:-1]]
        pds = self.lambda_linkage_vineyard(startends, n_jobs=n_jobs, tol=tol)
        hf = np.zeros((n_k - 1, n_s - 1))
        for i, pd in enumerate(pds):
            for bar in pd:
                b, d = bar
                start = np.searchsorted(ss[:-1], b)
                end = np.searchsorted(ss[:-1], d)
                hf[i, start:end] += 1
        return hf

    def connection_radius(self, percentiles=1):
        hc = self.lambda_linkage([0, 0], [np.infty, 0])
        return np.quantile(hc._merges_heights, percentiles)

    def extend_clustering_by_hill_climbing(self, labels, n_iterations, n_neighbors):
        old_labels = labels
        for _ in range(n_iterations):
            new_labels = []
            for x in range(self._size):
                if old_labels[x] == -1:
                    neighbors_labels = old_labels[self._nn_indices[x, :n_neighbors]]
                    neighbors_labels = neighbors_labels[neighbors_labels != -1]
                    if len(neighbors_labels) == 0:
                        new_labels.append(-1)
                    else:
                        new_labels.append(mode(neighbors_labels)[0][0])
                else:
                    new_labels.append(old_labels[x])
            new_labels = np.array(new_labels)
            old_labels = new_labels
        return new_labels


class _HierarchicalClustering:
    def __init__(self, heights, merges, merges_heights, start, end):
        # assumes heights and merges_heights are between start and end
        self._merges = merges
        self._merges_heights = merges_heights
        self._heights = heights
        self._start = start
        self._end = end

    def persistence_based_flattening(self, threshold):
        end = self._end
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
        while True:
            # while there is no merge
            while (
                hind < n_points
                and heights[appearances[hind]] <= current_merge_height
                and heights[appearances[hind]] < end
            ):
                # add all points that are born as new clusters
                uf.add(appearances[hind])
                clusters_birth[appearances[hind]] = heights[appearances[hind]]
                hind += 1
                if hind == n_points:
                    current_appearence_height = end
                else:
                    current_appearence_height = heights[appearances[hind]]
            # while there is no cluster being born
            while (
                mind < n_merges
                and merges_heights[mind] < current_appearence_height
                and merges_heights[mind] < end
            ):
                xy = merges[mind]
                x, y = xy
                rx = uf.__getitem__(x)
                ry = uf.__getitem__(y)
                # if both clusters are alive
                if rx not in clusters_died and ry not in clusters_died:
                    bx = clusters_birth[rx]
                    by = clusters_birth[ry]
                    # if both have lived for more than the threshold, have them as flat clusters
                    if (
                        bx + threshold <= merges_heights[mind]
                        and by + threshold <= merges_heights[mind]
                    ):
                        clusters.append(list(uf.subset(x)))
                        clusters.append(list(uf.subset(y)))
                        uf.merge(x, y)
                        uf.add(mind + n_points)
                        uf.merge(x, mind + n_points)
                        rxy = uf.__getitem__(x)
                        clusters_died[rxy] = True
                    # otherwise, merge them
                    else:
                        # then merge them
                        del clusters_birth[rx]
                        del clusters_birth[ry]
                        uf.merge(x, y)
                        uf.add(mind + n_points)
                        uf.merge(x, mind + n_points)
                        rxy = uf.__getitem__(x)
                        clusters_birth[rxy] = min(bx, by)
                # if both clusters are already dead, just merge them into a dead cluster
                elif rx in clusters_died and ry in clusters_died:
                    uf.merge(x, y)
                    uf.add(mind + n_points)
                    uf.merge(x, mind + n_points)
                    rxy = uf.__getitem__(x)
                    clusters_died[rxy] = True
                # if only one of them is dead
                else:
                    # we make it so that ry already died and rx just died
                    if rx in clusters_died:
                        x, y = y, x
                        rx, ry = ry, rx
                    # if x has lived for longer than the threshold, have it as a flat cluster
                    if clusters_birth[rx] + threshold <= merges_heights[mind]:
                        clusters.append(list(uf.subset(x)))
                    # then merge the clusters into a dead cluster
                    uf.merge(x, y)
                    uf.add(mind + n_points)
                    uf.merge(x, mind + n_points)
                    rxy = uf.__getitem__(x)
                    clusters_died[rxy] = True
                mind += 1
                if mind == n_merges:
                    current_merge_height = end
                else:
                    current_merge_height = merges_heights[mind]
            if (hind == n_points or heights[appearances[hind]] >= end) and (
                mind == n_merges or merges_heights[mind] >= end
            ):
                break
        # go through all clusters that have been born but haven't been merged
        for x in range(n_points):
            if x in uf._indices:
                rx = uf.__getitem__(x)
                if rx not in clusters_died:
                    if clusters_birth[rx] + threshold <= end:
                        clusters.append(list(uf.subset(x)))
                    clusters_died[rx] = True
        current_cluster = 0
        res = np.full(n_points, -1)
        for cl in clusters:
            for x in cl:
                if x < n_points:
                    res[x] = current_cluster
            current_cluster += 1
        return res

    def persistence_diagram(self, tol=_TOL):
        end = self._end
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
        while True:
            # while there is no merge
            while (
                hind < n_points
                and heights[appearances[hind]] <= current_merge_height
                and heights[appearances[hind]] < end
            ):
                # add all points that are born as new clusters
                cluster_reps[appearances[hind]] = appearances[hind]
                hind += 1
                if hind == n_points:
                    current_appearence_height = end
                else:
                    current_appearence_height = heights[appearances[hind]]
            # while there is no cluster being born
            while (
                mind < n_merges
                and merges_heights[mind] < current_appearence_height
                and merges_heights[mind] < end
            ):
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
                pd.append([by, merges_heights[mind]])
                cluster_reps[mind + n_points] = rx
                cluster_reps[ry] = -1
                mind += 1
                if mind == n_merges:
                    current_merge_height = end
                else:
                    current_merge_height = merges_heights[mind]
            if (hind == n_points or heights[appearances[hind]] >= end) and (
                mind == n_merges or merges_heights[mind] >= end
            ):
                break
        # go through all clusters that are still alive
        for i in range(heights.shape[0]):
            if cluster_reps[i] == i:
                pd.append([heights[i], end])
        pd = np.array(pd)
        return pd[np.abs(pd[:, 0] - pd[:, 1]) > tol] - self._start
