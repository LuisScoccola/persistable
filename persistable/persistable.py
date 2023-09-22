# Authors: Luis Scoccola and Alexander Rolle
# License: 3-clause BSD

from ._vineyard import Vineyard
from .borrowed._hdbscan_boruvka import (
    KDTreeBoruvkaAlgorithm,
    BallTreeBoruvkaAlgorithm,
)
from .borrowed.prim_mst import mst_linkage_core_vector
from .borrowed.dense_mst import stepwise_dendrogram_with_core_distances
from .borrowed.dist_metrics import DistanceMetric
from .auxiliary import lazy_intersection
from .subsampling import close_subsample_fast_metric, close_subsample_distance_matrix
from .persistence_diagram_h0 import persistence_diagram_h0
from .signed_betti_numbers import (
    signed_betti,
    rank_decomposition_2d_rectangles,
    rank_decomposition_2d_rectangles_to_hooks,
)
import numpy as np
import warnings
from sklearn.neighbors import KDTree, BallTree
from scipy.cluster.hierarchy import DisjointSet
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    minimum_spanning_tree as sparse_matrix_minimum_spanning_tree,
)
from scipy.stats import mode
from joblib import Parallel, delayed
from joblib.parallel import cpu_count



kdtree_valid_metrics = KDTree.valid_metrics
balltree_valid_metrics = BallTree.valid_metrics

_TOL = 1e-08
# starting when we consider a dataset large
_MANY_POINTS = 40000




def parallel_computation(function, inputs, n_jobs, debug=False, threading=False):
    if n_jobs == 1:
        return [function(inp) for inp in inputs]
    else:
        verbose = 11 if debug else 0
        n_jobs = min(cpu_count(), n_jobs)
        if threading:
            return Parallel(n_jobs=n_jobs, backend="threading", verbose=verbose)(
                delayed(function)(inp) for inp in inputs
            )
        else:
            return Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(function)(inp) for inp in inputs
            )


class Persistable:
    """Density-based clustering on finite metric spaces.

    Persistable has two main clustering methods: ``cluster()`` and ``quick_cluster()``.
    The methods are similar, the main difference being that ``quick_cluster()`` takes
    parameters that are sometimes easier to set. The parameters for ``cluster()``
    are usually set by using the graphical user interface implemented by the
    ``PersistableInteractive`` class.

    X: ndarray (n_samples, n_features)
        A numpy vector of shape (samples, features) or a distance matrix.

    metric: string, optional, default is "minkowski"
        A string determining which metric is used to compute distances
        between the points in X. It can be a metric in ``KDTree.valid_metrics``
        or ``BallTree.valid_metrics`` (which can be found by
        ``from sklearn.neighbors import KDTree, BallTree``) or ``"precomputed"``
        if X is a distance matrix.

    measure: None or ndarray(n_samples), default is None
        A numpy vector of length (samples) of non-negative numbers, which is
        intepreted as a measure on the data points.
        If None, the uniform measure where each point has weight 1/samples is used.
        If the measure does not sum to 1, it is normalized.

    subsample: None or int, optional, default is None
        Number of datapoints to subsample. The subsample is taken to have a measure
        that approximates the original measure on the full dataset as best as possible,
        in the Prokhorov sense. If metric is ``minkowski`` and the dimensionality is
        not too big, computing the sample takes time O( log(size_subsample) * size_data ),
        otherwise it takes time O( size_subsample * size_data ).

    n_neighbors: int or string, optional, default is "auto"
        Number of neighbors for each point in X used to initialize
        datastructures used for clustering. If set to ``"all"`` it will use
        the number of points in the dataset, if set to ``"auto"`` it will find
        a reasonable default.

    debug: bool, optional, default is False
        Whether to print debug messages.

    threading: bool, optional, default is False
        Whether to use python threads for parallel computation with ``joblib``.
        If false, the backend ``loky`` is used. In this case, using threads is
        significantly slower because of the GIL, but the backend ``loky`` does
        not work well in some systems.

    n_jobs: int, default is 1
        Number of processes or threads to use to fit the data structures, for exaple
        to compute the nearest neighbors of all points in the dataset.

    ``**kwargs``:
        Passed to ``KDTree`` or ``BallTree``.

    """

    def __init__(
        self,
        X,
        metric="minkowski",
        measure=None,
        subsample=None,
        n_neighbors="auto",
        debug=False,
        threading=False,
        n_jobs=4,
        **kwargs
    ):
        self._subsample = None

        # if no measure was passed, assume normalized counting measure
        if measure is None:
            measure = np.full(X.shape[0], 1.0 / X.shape[0])
        else:
            measure = np.array(measure)
            assert np.all(measure >= 0)
            # rescale so that total measure is 1
            measure = measure / np.sum(measure)

        if subsample is not None:
            if type(subsample) != int:
                raise ValueError("subsample must be either None or an integer.")
            ms = _MetricSpace(
                X, metric, threading=threading, debug=debug, n_jobs=n_jobs, **kwargs
            )
            if subsample >= X.shape[0]:
                subsample = int(X.shape[0] / 4)
                warnings.warn(
                    "subsample is greater than or equal to number of datapoints, using "
                    + str(subsample)
                    + " instead."
                )
            self._subsample = subsample

            subsample_euclidean = metric == "minkowski"

            subsample_indices, subsample_representatives = ms.close_subsample(
                subsample, euclidean=subsample_euclidean
            )
            self._subsample = subsample_indices.shape[0]

            X = X.copy()
            if metric == "precomputed":
                X = X[subsample_indices, :][:, subsample_indices]
            else:
                X = X[subsample_indices, :]
            self._subsample_representatives = subsample_representatives

            # compute measure for subsample
            new_measure = np.zeros(self._subsample)
            for i, _ in enumerate(self._subsample_representatives):
                new_measure[self._subsample_representatives[i]] += measure[i]
            measure = new_measure

        # if metric is minkowski but no p was passed, assume p = 2
        if metric == "minkowski" and "p" not in kwargs:
            kwargs["p"] = 2

        # if no n_neighbors for fitting mpspace was passed, compute a reasonable one
        if n_neighbors == "auto":
            if X.shape[0] < 100:
                n_neighbors = X.shape[0]
            else:
                n_neighbors = min(int(np.log10(X.shape[0])) * 100, X.shape[0])
        elif n_neighbors == "all":
            n_neighbors = X.shape[0]
        elif type(n_neighbors) == int and n_neighbors >= 1:
            n_neighbors = min(n_neighbors, X.shape[0])
        else:
            raise ValueError(
                "n_neighbors must be either auto, all, or a positive integer."
            )
        # keep max_k (normalized n_neighbors)
        self._maxk = n_neighbors / X.shape[0]
        self._dataset_is_large = X.shape[0] > _MANY_POINTS

        # construct the filtration
        self._mpspace = _MetricProbabilitySpace(
            X,
            metric,
            measure,
            n_neighbors,
            threading=threading,
            debug=debug,
            n_jobs=n_jobs,
            **kwargs
        )

        self._bifiltration = _DegreeRipsBifiltration(
            self._mpspace,
            debug=debug,
            threading=threading,
        )

    def quick_cluster(
        self,
        n_neighbors: int = 30,
        n_clusters_range=np.array([3, 15]),
    ):
        """Find parameters automatically and cluster dataset passed at initialization.

        This function will find the best number of clusterings in the range passed
        by the user, according to a certain measure of goodness of clustering
        based on prominence of modes of the underlying distribution.

        n_neighbors: int, optional, default is 30
            Number of neighbors used as a maximum density threshold
            when doing density-based clustering.

        n_clusters_range: (int, int), optional, default is [3, 15]
            A two-element list or tuple representing an integer
            range of possible numbers of clusters to consider when finding the
            optimum number of clusters.

        returns:
            A numpy array of length the number of points in the dataset containing
            integers from -1 to the number of clusters minus 1, representing the
            labels of the final clustering. The label -1 represents noise points,
            i.e., points deemed not to belong to any cluster by the algorithm.

        """
        k = n_neighbors / self._mpspace.size()
        default_percentile = 0.95
        s = self._bifiltration.connection_radius(default_percentile) * 2

        hc = self._bifiltration.lambda_linkage([0, k], [s, 0])
        pd = hc.persistence_diagram()
        if pd.shape[0] == 0:
            return np.full(self._mpspace.size(), -1)

        def _prominences(bd):
            return np.sort(np.abs(bd[:, 0] - bd[:, 1]))[::-1]

        proms = _prominences(pd)
        if n_clusters_range[1] >= len(proms):
            return self.cluster(n_clusters_range[1], [0, k], [s, 0])
        logproms = np.log(proms)
        peaks = logproms[:-1] - logproms[1:]
        min_clust = n_clusters_range[0] - 1
        max_clust = n_clusters_range[1] - 1
        num_clust = np.argmax(peaks[min_clust:max_clust]) + min_clust + 1
        return self.cluster(
            num_clust,
            [0, k],
            [s, 0]
        )

    def cluster(
        self,
        n_clusters,
        start,
        end,
        flattening_mode="conservative",
        keep_low_persistence_clusters=False,
    ):
        """Cluster dataset passed at initialization.

        n_clusters: int
            Integer determining how many clusters the final clustering
            must have. Note that the final clustering can have fewer clusters
            if the selected parameters do not allow for so many clusters.

        start: (float, float)
            Two-element list, tuple, or numpy array representing a point on
            the positive plane determining the start of the segment in the
            two-parameter hierarchical clustering used to do persistence-based
            clustering.

        end: (float, float)
            Two-element list, tuple, or numpy array representing a point on
            the positive plane determining the end of the segment in the
            two-parameter hierarchical clustering used to do persistence-based
            clustering.

        flattening_mode: string, optional, default is "conservative"
            If "exhaustive", flatten the hierarchical clustering using the approach
            of 'Persistence-Based Clustering in Riemannian Manifolds' Chazal, Guibas,
            Oudot, Skraba.
            If "conservative", use the more stable approach of
            'Stable and consistent density-based clustering' Rolle, Scoccola.
            The conservative approach usually results in more unclustered points.

        keep_low_persistence_clusters: bool, optional, default is False
            Only has effect if ``flattening_mode`` is set to "exhaustive".
            Whether to keep clusters that are born below the persistence threshold
            associated to the selected n_clusters. If set to True, the number of clusters
            can be larger than the selected one.

        returns:
            A numpy array of length the number of points in the dataset containing
            integers from -1 to the number of clusters minus 1, representing the
            labels of the final clustering. The label -1 represents noise points,
            i.e., points deemed not to belong to any cluster by the algorithm.

        """

        start, end = np.array(start), np.array(end)
        if start.shape != (2,) or end.shape != (2,):
            raise ValueError("start and end must both be points on the plane.")
        if n_clusters < 1:
            raise ValueError("n_clusters must be greater than 0.")
        hc = self._bifiltration.lambda_linkage(start, end)
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
        cl = hc.persistence_based_flattening(
            threshold,
            flattening_mode=flattening_mode,
            keep_low_persistence_clusters=keep_low_persistence_clusters,
        )

        if self._subsample is not None:
            new_cl = np.full(self._subsample_representatives.shape[0], -1)
            for i, _ in enumerate(self._subsample_representatives):
                new_cl[i] = cl[self._subsample_representatives[i]]
            cl = new_cl

        return cl

    def _dbscan_cluster(self, xy):
        x,y = xy
        hc = self._bifiltration.lambda_linkage([0,y], [np.inf, y])
        cl = hc.cut(x)

        if self._subsample is not None:
            new_cl = np.full(self._subsample_representatives.shape[0], -1)
            for i, _ in enumerate(self._subsample_representatives):
                new_cl[i] = cl[self._subsample_representatives[i]]
            cl = new_cl

        return cl

    def _find_end(self):
        return self._bifiltration.find_end()

    def _default_granularity(self):
        if self._mpspace.size() > _MANY_POINTS:
            return 3
        elif self._mpspace.size() < 5000:
            return 80
        else:
            return 30

    def _hilbert_function(
        self,
        min_s,
        max_s,
        max_k,
        min_k,
        granularity,
        reduced=False,
        n_jobs=1,
    ):
        return self._bifiltration.hilbert_function_on_regular_grid(
            min_s, max_s, max_k, min_k, granularity, reduced=reduced, n_jobs=n_jobs
        )

    def _rank_invariant(
        self, min_s, max_s, max_k, min_k, granularity, reduced=False, n_jobs=1
    ):
        return self._bifiltration.rank_invariant_on_regular_grid(
            min_s, max_s, max_k, min_k, granularity, reduced=reduced, n_jobs=n_jobs
        )

    def _linear_vineyard(
        self, start_end1, start_end2, n_parameters, reduced=False, n_jobs=1
    ):
        return self._bifiltration.linear_vineyard(
            start_end1, start_end2, n_parameters, reduced=reduced, n_jobs=n_jobs
        )


class _DegreeRipsBifiltration:
    def __init__(self, mpspace, debug=False, threading=False):
        self._debug = debug
        self._threading = threading
        self._mpspace = mpspace

    def _core_distance(self, point_index, s_intercept, k_intercept, max_k=None):

        kernel_estimate = self._mpspace.kernel_estimate()
        nn_distance = self._mpspace.nn_distance()

        max_k = k_intercept if max_k is None else max_k
        if s_intercept != np.inf:
            i_indices_and_finished_at_last_index = []
            mu = s_intercept / k_intercept
            k_to_s = lambda y: s_intercept - mu * y
            max_k_larger_last_kernel_estimate = []
            for p in point_index:
                i_indices_and_finished_at_last_index.append(
                    lazy_intersection(
                        kernel_estimate[p],
                        nn_distance[p],
                        s_intercept,
                        k_intercept,
                    )
                )
                max_k_larger_last_kernel_estimate.append(
                    (kernel_estimate[p, -1] < max_k)
                )
            max_k_larger_last_kernel_estimate = np.array(
                max_k_larger_last_kernel_estimate
            )
            i_indices_and_finished_at_last_index = np.array(
                i_indices_and_finished_at_last_index
            )
            i_indices = i_indices_and_finished_at_last_index[:, 0]
            finished_at_last_index = i_indices_and_finished_at_last_index[:, 1]
            # check if for any points we don't have enough neighbors to properly compute its core scale
            # for this, the lazy intersection must have finished at the last index and the max_k
            # of the line segment chosen must be larger than the max kernel estimate for the point
            if np.any(
                np.logical_and(
                    finished_at_last_index, max_k_larger_last_kernel_estimate
                )
            ):
                warnings.warn(
                    "Don't have enough neighbors to properly compute core scale, or point takes too long to appear."
                )

            def op(p, i):
                return np.where(
                    k_to_s(kernel_estimate[p, i - 1]) <= nn_distance[p, i],
                    k_to_s(kernel_estimate[p, i - 1]),
                    nn_distance[p, i],
                )

            return np.where(i_indices == 0, 0, op(point_index, i_indices))
        else:
            i_indices = []
            for p in point_index:
                idx = np.searchsorted(kernel_estimate[p], k_intercept, side="left")
                if idx == nn_distance[p].shape[0]:
                    idx -= 1
                i_indices.append(idx)
            i_indices = np.array(i_indices)
            # TODO: properly check and warn of not enough n_neighbors or
            # explicitly ensure that the following does not happen:
            # if self._n_neighbors < self._size:
            #    out_of_range = np.where(
            #        (
            #            i_indices
            #            >= np.apply_along_axis(len, -1, self._nn_indices[point_index])
            #        )
            #        & (
            #            np.apply_along_axis(len, -1, self._nn_indices[point_index])
            #            < self._size
            #        ),
            #        True,
            #        False,
            #    )
            #    if np.any(out_of_range):
            #        warnings.warn(
            #            "Don't have enough neighbors to properly compute core scale."
            #        )
            return nn_distance[(point_index, i_indices)]

    def find_end(self, tolerance=1e-4):
        maxk = self._mpspace.max_fitted_density()

        dataset_is_large = self._mpspace.size() > _MANY_POINTS
        if dataset_is_large:
            default_percentile = 0.95
            return self.connection_radius(default_percentile) * 4, maxk

        def pers_diag(k):
            return self.lambda_linkage([0, k], [np.infty, k]).persistence_diagram()

        lower_bound = 0
        upper_bound = maxk

        i = 0
        while True:
            current_k = (lower_bound + upper_bound) / 2
            i += 1

            pd = pers_diag(current_k)
            pd = np.array(pd)
            if pd.shape[0] == 0:
                raise Exception(
                    "Empty persistence diagram found when trying to find end of bifiltration."
                )
            # persistence diagram has more than one class
            elif pd.shape[0] > 1:
                lower_bound = current_k
                if np.abs(current_k - maxk) < _TOL:
                    pd = pers_diag(lower_bound)
                    return [np.max(pd[pd[:, 1] != np.infty][:, 1]), current_k]
            # persistence diagram has exactly one class
            else:
                upper_bound = current_k

            if np.abs(lower_bound - upper_bound) < tolerance:
                pd = pers_diag(lower_bound)
                return [np.max(pd[pd[:, 1] != np.infty][:, 1]), current_k]

    def connection_radius(self, percentiles=1):
        hc = self.lambda_linkage([0, 0], [np.infty, 0])
        return np.quantile(hc.merges_heights(), percentiles)

    def _lambda_linkage_vertical(self, s_intercept, k_start, k_end):
        if k_end > k_start:
            raise ValueError("Parameters do not give a monotonic line.")

        indices = np.arange(self._mpspace.size())
        k_births = self._mpspace.density_estimate(
            indices, s_intercept, max_density=k_start
        )
        # clip
        k_births = np.maximum(k_end, np.minimum(k_start, k_births))
        # make it covariant
        k_births = k_start - k_births

        res_hierarchical_clustering = (
            self._mpspace.hierarchical_clustering_filtered_rips_graph(
                k_births, s_intercept
            )
        )

        hc_start = 0
        hc_end = k_start - k_end
        res_hierarchical_clustering.clip(hc_start, hc_end)

        return res_hierarchical_clustering

    def _lambda_linkage_skew(self, start, end):
        def _startend_to_intercepts(start, end):
            if end[0] == np.infty or start[1] == end[1]:
                k_intercept = start[1]
                s_intercept = np.infty
            else:
                slope = (end[1] - start[1]) / (end[0] - start[0])
                k_intercept = -start[0] * slope + start[1]
                s_intercept = -k_intercept / slope
            return s_intercept, k_intercept

        hc_start = start[0]
        hc_end = end[0]
        indices = np.arange(self._mpspace.size())
        s_intercept, k_intercept = _startend_to_intercepts(start, end)
        max_k = start[1]
        core_distances = self._core_distance(indices, s_intercept, k_intercept, max_k)

        core_distances = np.minimum(hc_end, core_distances)
        core_distances = np.maximum(hc_start, core_distances)

        single_linkage_hc = self._mpspace.generalized_single_linkage(core_distances)

        single_linkage_hc.clip(hc_start, hc_end)

        return single_linkage_hc

    def lambda_linkage(self, start, end):
        if start[0] > end[0] or start[1] < end[1]:
            raise ValueError("Parameters do not give a monotonic line.")

        if start[0] == end[0]:
            s_intercept = start[0]
            k_start = start[1]
            k_end = end[1]
            return self._lambda_linkage_vertical(s_intercept, k_start, k_end)
        else:
            return self._lambda_linkage_skew(start, end)

    def lambda_linkage_vineyard(self, startends, reduced=False, tol=_TOL, n_jobs=1):
        run_in_parallel = lambda startend: self.lambda_linkage(
            startend[0], startend[1]
        ).persistence_diagram(tol=tol, reduced=reduced)

        return parallel_computation(
            run_in_parallel,
            startends,
            n_jobs,
            debug=self._debug,
            threading=self._threading,
        )

    def linear_vineyard(
        self, start_end1, start_end2, n_parameters, reduced=False, n_jobs=1
    ):
        start1, end1 = start_end1
        start2, end2 = start_end2
        if (
            start1[0] > end1[0]
            or start1[1] < end1[1]
            or start2[0] > end2[0]
            or start2[1] < end2[1]
        ):
            raise ValueError(
                "Parameters chosen for vineyard will result in non-monotonic lines!"
            )
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
        pds = self.lambda_linkage_vineyard(startends, reduced=reduced, n_jobs=n_jobs)
        return Vineyard(startends, pds)

    def _rank_invariant(self, ss, ks, reduced=False, n_jobs=1):
        # go on one more step to compute rank invariant at the end of the grid
        ss = list(ss)
        ks = list(ks)
        ss.append(ss[-1] + _TOL)
        ks.append(ks[-1] - _TOL)
        n_s = len(ss)
        n_k = len(ks)
        ks = np.array(ks)
        startends_horizontal = [[[ss[0], k], [ss[-1], k]] for k in ks]
        startends_vertical = [[[s, ks[0]], [s, ks[-1]]] for s in ss]
        startends = startends_horizontal + startends_vertical

        def run_in_parallel(startend):
            return self.lambda_linkage(startend[0], startend[1])

        hcs = parallel_computation(
            run_in_parallel,
            startends,
            n_jobs,
            debug=self._debug,
            threading=self._threading,
        )
        hcs_horizontal = hcs[:n_k]
        for hc in hcs_horizontal:
            hc.snap_to_grid(ss)
        hcs_vertical = hcs[n_k:]
        for hc in hcs_vertical:
            hc.snap_to_grid(ks[0] - ks)

        def _splice_hcs(s_index, k_index):
            # the horizontal hierarchical clustering
            hor_hc = hcs_horizontal[k_index]
            # keep only things that happened before s_index
            hor_heights = hor_hc._heights.copy()
            hor_heights[hor_heights >= s_index] = s_index + len(ks) - k_index + 1
            hor_merges = hor_hc._merges[hor_hc._merges_heights < s_index]
            hor_merges_heights = hor_hc._merges_heights[
                hor_hc._merges_heights < s_index
            ]
            hor_end = s_index - 1
            hor_start = hor_hc._start

            # the vertical hierarchical clustering
            ver_hc = hcs_vertical[s_index]
            # push all things that happened before k_index there, and index starting from s_index
            ver_heights = s_index + np.maximum(k_index, ver_hc._heights) - k_index
            ver_merges_heights = (
                s_index + np.maximum(k_index, ver_hc._merges_heights) - k_index
            )
            # same merges in same order
            ver_merges = ver_hc._merges

            ver_start = s_index
            ver_end = s_index + ver_hc._end - k_index + 1

            heights = np.minimum(hor_heights, ver_heights)
            if len(hor_merges) == 0 and len(ver_merges) == 0:
                merges = np.array([], dtype=int)
            else:
                if len(hor_merges) == 0:
                    hor_merges.reshape([0, 2])
                if len(ver_merges) == 0:
                    ver_merges.reshape([0, 2])
                merges = np.concatenate((hor_merges, ver_merges))
            merges_heights = np.concatenate((hor_merges_heights, ver_merges_heights))
            start = hor_start
            end = ver_end

            return _HierarchicalClustering(heights, merges, merges_heights, start, end)

        def _pd_spliced_hc(s_index_k_index):
            s_index, k_index = s_index_k_index
            return _splice_hcs(s_index, k_index).persistence_diagram(reduced=reduced)

        indices = [
            [s_index, k_index] for s_index in range(n_s) for k_index in range(n_k)
        ]
        pds = parallel_computation(
            _pd_spliced_hc,
            indices,
            n_jobs,
            debug=self._debug,
            threading=self._threading,
        )
        pds = [[indices[i][0], indices[i][1], pds[i]] for i in range(len(indices))]

        ri = np.zeros((n_s, n_k, n_s, n_k), dtype=int)

        for s_index, k_index, pd in pds:
            for bar in pd:
                b, d = bar
                b, d = int(b), int(d)
                # this if may be unnecessary
                if b <= s_index and d >= s_index:
                    for i in range(b, s_index + 1):
                        for j in range(s_index, d):
                            ri[i, k_index, s_index, j - s_index + k_index] += 1

        ri = ri[:-1, :, :, :][:, :-1, :, :][:, :, :-1, :][:, :, :, :-1]
        return ri

    def rank_invariant_on_regular_grid(
        self, min_s, max_s, max_k, min_k, granularity, reduced=False, n_jobs=1
    ):
        if min_k >= max_k:
            raise ValueError("min_k must be smaller than max_k.")
        if min_s >= max_s:
            raise ValueError("min_s must be smaller than max_s.")
        if max_k > self._mpspace.max_fitted_density():
            max_k = min(max_k, self._mpspace.max_fitted_density())
            warnings.warn(
                "Not enough neighbors to compute chosen max density threshold, using "
                + str(max_k)
                + " instead. If needed, re-initialize the Persistable instance with a larger n_neighbors."
            )
        if min_k >= max_k:
            min_k = max_k / 2
            warnings.warn(
                "max density threshold too large, using " + str(min_k) + " instead."
            )

        ss = np.linspace(min_s, max_s, granularity)
        ks = np.linspace(min_k, max_k, granularity)[::-1]
        ri = self._rank_invariant(ss, ks, n_jobs=n_jobs, reduced=reduced)
        # need to cast explicitly to int64 for windows compatibility
        rdr = rank_decomposition_2d_rectangles(np.array(ri, dtype=np.int64))
        return ss, ks, ri, rdr, rank_decomposition_2d_rectangles_to_hooks(rdr)

    def _hilbert_function(self, ss, ks, reduced=False, n_jobs=1):
        n_s = len(ss)
        n_k = len(ks)
        ss = list(ss)
        # go on one more step to compute the Hilbert function at the last point
        ss.append(ss[-1] + _TOL)
        startends = [[[ss[0], k], [ss[-1], k]] for k in ks]
        pds = self.lambda_linkage_vineyard(startends, reduced=reduced, n_jobs=n_jobs)
        hf = np.zeros((n_s, n_k), dtype=int)
        for i, pd in enumerate(pds):
            for bar in pd:
                b, d = bar
                start = np.searchsorted(ss[:-1], b)
                end = np.searchsorted(ss[:-1], d)
                hf[start:end, i] += 1
        return hf

    def hilbert_function_on_regular_grid(
        self,
        min_s,
        max_s,
        max_k,
        min_k,
        granularity,
        reduced=False,
        n_jobs=1,
    ):
        if min_k >= max_k:
            raise ValueError("min_k must be smaller than max_k.")
        if min_s >= max_s:
            raise ValueError("min_s must be smaller than max_s.")
        if max_k > self._mpspace.max_fitted_density():
            max_k = min(max_k, self._mpspace.max_fitted_density())
            warnings.warn(
                "Not enough neighbors to compute chosen max density threshold, using "
                + str(max_k)
                + " instead. If needed, re-initialize the Persistable instance with a larger n_neighbors."
            )
        if min_k >= max_k:
            min_k = max_k / 2
            warnings.warn(
                "max density threshold too large, using " + str(min_k) + " instead."
            )

        ss = np.linspace(min_s, max_s, granularity)
        ks = np.linspace(min_k, max_k, granularity)[::-1]
        hf = self._hilbert_function(ss, ks, reduced=reduced, n_jobs=n_jobs)
        return ss, ks, hf, signed_betti(hf)


class _MetricSpace:

    _MAX_DIM_USE_BORUVKA = 60

    def __init__(
        self, X, metric, leaf_size=40, threading=False, debug=False, n_jobs=1, **kwargs
    ):
        # save extra arguments for metric
        self._kwargs = kwargs

        self._threading = threading
        self._n_jobs = n_jobs
        self._debug = debug

        # default values before fitting
        self._fitted_nn = False
        self._nn_distance = None
        self._nn_indices = None
        self._n_neighbors = None
        self._maxs = None
        self._size = None
        self._dimension = None
        self._points = None
        self._nn_tree = None
        self._boruvka_tree = None
        self._dist_mat = None
        self._dist_metric = None

        self._fit_metric(X, metric, leaf_size, **kwargs)

    def _fit_metric(self, X, metric, leaf_size, **kwargs):
        # save point cloud
        self._size = X.shape[0]
        self._dimension = X.shape[1]
        if metric != "precomputed":
            self._points = X
        else:
            self._points = np.array(range(self._size))

        # save metric and spatial tree
        self._metric = metric
        self._leaf_size = leaf_size
        if metric in kdtree_valid_metrics + balltree_valid_metrics:
            leaf_size_boruvka = 3 if self._leaf_size < 3 else self._leaf_size // 3
            if metric in kdtree_valid_metrics:
                self._nn_tree = KDTree(
                    X, metric=metric, leaf_size=self._leaf_size, **kwargs
                )
                self._boruvka_tree = KDTree(
                    X, metric=metric, leaf_size=leaf_size_boruvka, **kwargs
                )
            elif metric in balltree_valid_metrics:
                self._nn_tree = BallTree(
                    X, metric=metric, leaf_size=self._leaf_size, **kwargs
                )
                self._boruvka_tree = BallTree(
                    X, metric=metric, leaf_size=leaf_size_boruvka, **kwargs
                )

            self._dist_metric = DistanceMetric.get_metric(self._metric, **self._kwargs)
        elif metric == "precomputed":
            self._dist_mat = np.array(X)
            self._dist_metric = self._dist_mat
        else:
            raise ValueError("Metric given is not supported.")

    def _fit_nn(self, n_neighbors):
        self._n_neighbors = n_neighbors
        if self._metric in balltree_valid_metrics + kdtree_valid_metrics:

            def query_neighbors(points):
                return self._nn_tree.query(
                    points,
                    self._n_neighbors,
                    return_distance=True,
                    sort_results=True,
                    dualtree=True,
                    breadth_first=True,
                )

            # if we don't have too many points
            if self.size() <= _MANY_POINTS or self._n_jobs == 1:
                _nn_distance, neighbors = query_neighbors(self._points)
            else:
                delta = self.size() // self._n_jobs
                datasets = []
                for i in range(self._n_jobs):
                    if i == self._n_jobs - 1:
                        datasets.append(self._points[i * delta :])
                    else:
                        datasets.append(self._points[i * delta : (i + 1) * delta])
                nn_data = parallel_computation(
                    query_neighbors, datasets, n_jobs=self._n_jobs, debug=self._debug
                )

                _nn_distance = np.vstack([x[0] for x in nn_data])
                neighbors = np.vstack([x[1] for x in nn_data])

            maxs_given_by_n_neighbors = np.min(_nn_distance[:, -1])
            self._maxs = maxs_given_by_n_neighbors
        else:
            self._n_neighbors = self._size
            self._maxs = 0
            neighbors = np.argsort(self._dist_mat)
            _nn_distance = self._dist_mat[
                np.arange(len(self._dist_mat)), neighbors.transpose()
            ].transpose()
        self._fitted_nn = True

        self._nn_indices = np.array(neighbors, dtype=np.int_)
        self._nn_distance = np.array(_nn_distance)

    ####def distance(self, i,j):
    ####    if self._metric in kdtree_valid_metrics + balltree_valid_metrics:
    ####        return self._dist_metric.dist(self._points[i], self._points[j], self._dimension)
    ####    else:
    ####        return self._dist_mat[i,j]

    def size(self):
        return self._size

    def generalized_single_linkage(self, core_distances):
        if self._metric in kdtree_valid_metrics:
            if self._dimension > self._MAX_DIM_USE_BORUVKA:
                X = self._points
                if not X.flags["C_CONTIGUOUS"]:
                    X = np.array(X, dtype=np.double, order="C")
                sl = mst_linkage_core_vector(X, core_distances, self._dist_metric)
            else:
                sl = KDTreeBoruvkaAlgorithm(
                    self._boruvka_tree,
                    core_distances,
                    self._nn_indices,
                    metric=self._metric,
                    **self._kwargs
                ).spanning_tree()
        elif self._metric in balltree_valid_metrics:
            if self._dimension > self._MAX_DIM_USE_BORUVKA:
                X = self._points
                if not X.flags["C_CONTIGUOUS"]:
                    X = np.array(X, dtype=np.double, order="C")
                sl = mst_linkage_core_vector(X, core_distances, self._dist_metric)
            else:
                sl = BallTreeBoruvkaAlgorithm(
                    self._boruvka_tree,
                    core_distances,
                    self._nn_indices,
                    metric=self._metric,
                    **self._kwargs
                ).spanning_tree()
        else:
            sl = stepwise_dendrogram_with_core_distances(
                self.size(), self._dist_mat, core_distances
            )
        merges = sl[:, 0:2].astype(int)
        merges_heights = sl[:, 2]

        return _HierarchicalClustering(
            core_distances, merges, merges_heights, -np.inf, np.inf
        )

    def hierarchical_clustering_filtered_rips_graph(self, k_births, rips_radius):
        shift = min(k_births) + 1
        # must shift to strictly positive births (sparse matrix mst routine treats
        # edges with zero and very small weight as not there (i.e., as having infinite weight))
        k_births = k_births + shift

        # metric tree case
        if self._metric in kdtree_valid_metrics + balltree_valid_metrics:
            s_neighbors = self._nn_tree.query_radius(self._points, rips_radius)
        # dense distance matrix case
        elif self._metric == "precomputed":
            s_neighbors = []
            for i in range(self.size()):
                s_neighbors.append(np.argwhere(self._dist_mat[i] <= rips_radius)[:, 0])
        else:
            raise ValueError("Metric given is not supported.")

        edges = []
        entries = []
        for i in range(self.size()):
            for j in s_neighbors[i]:
                if j > i:
                    edges.append([i, j])
                    entries.append(max(k_births[i], k_births[j]))
        matrix_entries = np.array(entries)
        edges = np.array(edges, dtype=int)
        if len(edges) > 0:
            graph = csr_matrix(
                (matrix_entries, (edges[:, 0], edges[:, 1])), (self.size(), self.size())
            )

            mst = sparse_matrix_minimum_spanning_tree(graph)
            Is, Js = mst.nonzero()
            # we now undo the shift
            vals = np.array(mst[Is, Js])[0] - shift
            sort_indices = np.argsort(vals)
            Is = Is[sort_indices]
            Js = Js[sort_indices]
            vals = vals[sort_indices]
            merges = np.zeros((vals.shape[0], 2), dtype=int)
            merges[:, 0] = Is
            merges[:, 1] = Js
            merges_heights = vals
        else:
            merges = np.array([], dtype=int)
            merges_heights = np.array([])

        # undo the shift
        core_scales = k_births - shift

        return _HierarchicalClustering(
            core_scales, merges, merges_heights, -np.inf, np.inf
        )

    def close_subsample(self, subsample_size, seed=0, euclidean=False):
        """ Returns a pair of arrays with the first array containing the indices \
            of a subsample of the given size that is close in the Hausdorff distance \
            and the second array containing the subsequent covering radii """

        if euclidean:
            return self._close_subsample_euclidean(subsample_size)

        np.random.seed(seed)
        random_start = np.random.randint(0, self.size())

        if self._metric in kdtree_valid_metrics + balltree_valid_metrics:
            X = self._points
            if not X.flags["C_CONTIGUOUS"]:
                X = np.array(X, dtype=np.double, order="C")
            return close_subsample_fast_metric(
                subsample_size, X, self._dist_metric, random_start=random_start
            )
        elif self._metric == "precomputed":
            dist_mat = self._dist_mat
            if not dist_mat.flags["C_CONTIGUOUS"]:
                dist_mat = np.array(dist_mat, dtype=np.double, order="C")
            return close_subsample_distance_matrix(
                subsample_size, dist_mat, random_start=random_start
            )
        else:
            raise ValueError("Metric given is not supported.")

    def _close_subsample_euclidean(self, subsample_size, num_points_tolerance=100):
        X = self._points

        lower_bound = 0.0

        upper_bound = 1.0
        while True:
            W = (upper_bound * X).astype(int)
            count = np.unique(W, axis=0).shape[0]

            if count < subsample_size:
                upper_bound *= 2
            else:
                break

        i = 0
        while True:
            epsilon = (lower_bound + upper_bound) / 2
            i += 1

            W = (epsilon * X).astype(int)
            count = np.unique(W, axis=0).shape[0]

            if count > subsample_size + num_points_tolerance:
                upper_bound = epsilon
            elif count < subsample_size - num_points_tolerance:
                lower_bound = epsilon
            else:
                break
            if np.abs(lower_bound - upper_bound) < _TOL:
                break

        W = (epsilon * X).astype(int)
        _, subsample_indices, subsample_representatives = np.unique(
            W, axis=0, return_index=True, return_inverse=True
        )
        return subsample_indices, subsample_representatives


# TODO:
# rips bifiltration should take a persistent metric space as input.
# examples of persistent metric spaces are the kernel filtration induced by a
# metric probability space, as well as a metric space together with a function
# class _PersistentMetricSpace:
#
#
# class _FilteredMetricSpace(_MetricSpace):
#    def __init__(self, X, metric, filter_function, **kwargs):
#        _MetricSpace.__init__(self, X, metric, **kwargs)
#        self._filter_function = filter_function


class _MetricProbabilitySpace(_MetricSpace):
    """Implements a finite metric probability space that can compute its \
       kernel density estimates """

    def __init__(
        self,
        X,
        metric,
        measure,
        n_neighbors,
        threading=False,
        debug=False,
        n_jobs=1,
        **kwargs
    ):
        _MetricSpace.__init__(
            self, X, metric, threading=threading, debug=debug, n_jobs=n_jobs, **kwargs
        )

        # fit metric space nearest neighbors
        self._fit_nn(n_neighbors)

        # default values before fitting
        self._fitted_density_estimates = False
        self._kernel_estimate = None
        self._measure = None
        self._maxk = None

        self._fit(measure)

    def _fit(self, measure):

        # save measure
        self._measure = measure

        # fit density estimate
        self._fitted_density_estimates = True
        self._kernel_estimate = np.cumsum(self._measure[self._nn_indices], axis=1)

        # set the max k for which we have enough neighbors
        self._maxk = self._n_neighbors / self._size

    def density_estimate(self, point_index, radius, max_density=1):
        """ Given a list of point indices and a radius, return the (unnormalized) \
            kernel density estimate at those points and at that radius """
        density_estimates = []
        out_of_range = False
        for p in point_index:
            if self._kernel_estimate[p, -1] < max_density:
                out_of_range = True
            neighbor_idx = np.searchsorted(self._nn_distance[p], radius, side="right")
            density_estimates.append(self._kernel_estimate[p, neighbor_idx - 1])
        if out_of_range:
            warnings.warn("Don't have enough neighbors to properly compute core scale.")
        return np.array(density_estimates)

    def kernel_estimate(self):
        return self._kernel_estimate

    def nn_distance(self):
        return self._nn_distance

    def max_fitted_density(self):
        return self._maxk


class _HierarchicalClustering:
    """Implements a covariant hierarchical clustering"""

    def __init__(self, heights, merges, merges_heights, start, end):
        # assumes heights and merges_heights are between start and end
        self._merges = np.array(merges, dtype=int)
        self._merges_heights = np.array(merges_heights, dtype=float)
        self._heights = np.array(heights, dtype=float)
        self._start = start
        self._end = end
        # persistence_diagram_h0 will fail if it receives an empty array
        if self._merges.shape[0] == 0:
            # we make the first (zeroth) element merge with itself as soon as it is born
            self._merges = np.array([[0, 0]], dtype=int)
            self._merges_heights = np.array([self._heights[0]], dtype=float)

    def merges_heights(self):
        return self._merges_heights

    def snap_to_grid(self, grid):
        def _snap_array(grid, arr):
            res = np.zeros(arr.shape[0], dtype=int)
            # assumes grid and arr are ordered smallest to largest
            res[arr <= grid[0]] = 0
            for i in range(len(grid) - 1):
                res[(arr <= grid[i + 1]) & (arr > grid[i])] = i + 1
            res[arr > grid[-1]] = len(grid) - 1
            return res

        self._merges_heights = _snap_array(grid, self._merges_heights)
        self._heights = _snap_array(grid, self._heights)
        self._start, self._end = _snap_array(grid, np.array([self._start, self._end]))

    def clip(self, start, end):
        assert start <= end

        self._heights = np.minimum(self._heights, end)
        self._heights = np.maximum(self._heights, start)
        self._merges_heights = np.minimum(self._merges_heights, end)
        self._merges_heights = np.maximum(self._merges_heights, start)
        self._start = start
        self._end = end

    def cut(self, cut_height):
        end = min(self._end, cut_height)
        heights = self._heights
        merges = self._merges
        merges_heights = self._merges_heights
        n_points = heights.shape[0]
        n_merges = merges.shape[0]
        # this orders the point by appearance
        appearances = np.argsort(heights)
        # contains the current clusters
        uf = DisjointSet()
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
                and heights[appearances[hind]] <= end
            ):
                # add all points that are born as new clusters
                uf.add(appearances[hind])
                hind += 1
                if hind == n_points:
                    current_appearence_height = end
                else:
                    current_appearence_height = heights[appearances[hind]]
            # while there is no cluster being born
            while (
                mind < n_merges
                and merges_heights[mind] < current_appearence_height
                and merges_heights[mind] <= end
            ):
                xy = merges[mind]
                x, y = xy
                uf.merge(x, y)
                mind += 1
                if mind == n_merges:
                    current_merge_height = end
                else:
                    current_merge_height = merges_heights[mind]
            if (hind == n_points or heights[appearances[hind]] >= end) and (
                mind == n_merges or merges_heights[mind] >= end
            ):
                break
        # contains the flat clusters
        clusters = []
        # go through all clusters
        for c in uf.subsets():
            c = list(c)
            clusters.append(list(c))
        current_cluster = 0
        res = np.full(n_points, -1)
        for cl in clusters:
            for x in cl:
                if x < n_points:
                    res[x] = current_cluster
            current_cluster += 1
        return res

    def persistence_based_flattening(
        self, threshold, flattening_mode, keep_low_persistence_clusters
    ):
        if flattening_mode=="conservative":
            return self._conservative_persistence_based_flattening(threshold)
        else:
            return self._tomato_style_persistence_based_flattening(
                threshold, keep_low_persistence_clusters
            )

    def _tomato_style_persistence_based_flattening(
        self, threshold, keep_low_persistence_clusters
    ):
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
        # contains the birth time of clusters
        clusters_birth = np.full(len(heights), -np.inf)
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
                bx = clusters_birth[rx]
                by = clusters_birth[ry]
                # if one of them has not lived more than the threshold, merge them
                # otherwise, don't
                if (
                    bx + threshold > merges_heights[mind]
                    or by + threshold > merges_heights[mind]
                ):
                    uf.merge(x, y)
                    rxy = uf.__getitem__(x)
                    clusters_birth[rxy] = min(bx, by)
                mind += 1
                if mind == n_merges:
                    current_merge_height = end
                else:
                    current_merge_height = merges_heights[mind]
            if (hind == n_points or heights[appearances[hind]] >= end) and (
                mind == n_merges or merges_heights[mind] >= end
            ):
                break
        # contains the flat clusters
        clusters = []
        # go through all clusters
        for c in uf.subsets():
            c = list(c)
            rc = uf.__getitem__(c[0])
            if (clusters_birth[rc] + threshold <= end) or keep_low_persistence_clusters:
                clusters.append(list(c))
        current_cluster = 0
        res = np.full(n_points, -1)
        for cl in clusters:
            for x in cl:
                if x < n_points:
                    res[x] = current_cluster
            current_cluster += 1
        return res

    def _conservative_persistence_based_flattening(self, threshold):
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
                        rxy = uf.__getitem__(x)
                        clusters_died[rxy] = True
                    # otherwise, merge them
                    else:
                        # then merge them
                        del clusters_birth[rx]
                        del clusters_birth[ry]
                        uf.merge(x, y)
                        rxy = uf.__getitem__(x)
                        clusters_birth[rxy] = min(bx, by)
                # if both clusters are already dead, just merge them into a dead cluster
                elif rx in clusters_died and ry in clusters_died:
                    uf.merge(x, y)
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

    def persistence_diagram(self, reduced=False, tol=_TOL):
        pd = persistence_diagram_h0(
            self._end,
            self._heights,
            # need to cast explicitly to int64 for windows compatibility
            np.array(self._merges, dtype=np.int64),
            self._merges_heights,
        )
        pd = np.array(pd)
        if pd.shape[0] == 0:
            return np.array([])
        pd = pd[np.abs(pd[:, 0] - pd[:, 1]) > tol]
        if reduced:
            to_delete = np.argmax(pd[:, 1] - pd[:, 0])
            return np.delete(pd, to_delete, axis=0)
        return pd
