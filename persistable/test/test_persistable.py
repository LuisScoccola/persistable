# Authors: Luis Scoccola
# License: 3-clause BSD

import unittest
from persistable import Persistable
from persistable.persistable import _HierarchicalClustering, _MetricProbabilitySpace
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from sklearn import datasets
import numpy as np


class TestMetricProbabilitySpace(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self._n = 25
        self._X = np.random.random_sample((self._n, 2))
        self._ps = [2, 4, 8, np.inf]
        self._s0s = np.linspace(0.1, 2, 10)
        self._k0s = np.linspace(0.01, 0.5, 10)
        self._number_different_weights = 3
        self._different_weights = [None]
        for _ in range(self._number_different_weights):
            self._different_weights.append(np.random.random_sample(self._n))

    def test_core_distances(self):
        n = 4
        X = np.array([[0, 0], [1, 0], [1, 1], [3, 0]])
        p = Persistable(X)
        mps = p._mpspace

        s0 = 2
        k0 = 0.5
        res = np.array([1, 1, 1, 1])
        np.testing.assert_almost_equal(mps._core_distance(np.arange(n), s0, k0), res)

        s0 = 1
        k0 = 0.5
        res = np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2])
        np.testing.assert_almost_equal(mps._core_distance(np.arange(n), s0, k0), res)

        s0 = np.infty
        k0 = 1
        res = np.array([3, 2, np.sqrt(5), 3])
        np.testing.assert_almost_equal(mps._core_distance(np.arange(n), s0, k0), res)

        s0 = np.infty
        k0 = 0.5
        res = np.array([1, 1, 1, 2])
        np.testing.assert_almost_equal(mps._core_distance(np.arange(n), s0, k0), res)

        p = Persistable(X, measure=np.array([0.5, 0.5, 0.5, 0.5]))
        mps = p._mpspace
        s0 = np.infty
        k0 = 0.6
        res = np.array([1, 1, 1, 2])
        np.testing.assert_almost_equal(mps._core_distance(np.arange(n), s0, k0), res)

    def test_same_core_distances(self):
        for w in self._different_weights:
            for p in self._ps:
                p1 = Persistable(self._X, measure=w, p=p)
                mps1 = p1._mpspace
                p2 = Persistable(
                    distance_matrix(self._X, self._X, p=p),
                    metric="precomputed",
                    measure=w,
                )
                mps2 = p2._mpspace
                for s0 in self._s0s:
                    for k0 in self._k0s:
                        np.testing.assert_almost_equal(
                            mps1._core_distance(np.arange(self._n), s0, k0),
                            mps2._core_distance(np.arange(self._n), s0, k0),
                        )

    def test_same_hierarchy(self):
        # test Boruvka, Prim, and dense Prim
        for w in self._different_weights:
            V = np.ones(self._X.shape[1])
            # will use BallTree and Boruvka
            p1 = Persistable(self._X, metric="seuclidean", measure=w, V=V)
            mps1 = p1._mpspace
            # will use dense MST
            p2 = Persistable(
                cdist(self._X, self._X, metric="seuclidean", V=V),
                metric="precomputed",
                measure=w,
            )
            mps2 = p2._mpspace
            num_components = 1000
            big_X = np.zeros((self._X.shape[0], num_components))
            big_X[:, : self._X.shape[1]] = self._X
            # will use BallTree and Prim
            V2 = np.ones(big_X.shape[1])
            p3 = Persistable(big_X, metric="seuclidean", measure=w, V=V2)
            mps3 = p3._mpspace
            for s0 in self._s0s:
                for k0 in self._k0s:
                    hc1 = mps1.lambda_linkage([0, k0], [s0, 0])
                    hc2 = mps2.lambda_linkage([0, k0], [s0, 0])
                    hc3 = mps3.lambda_linkage([0, k0], [s0, 0])
                    np.testing.assert_almost_equal(
                        hc1._merges_heights, hc2._merges_heights
                    )
                    np.testing.assert_almost_equal(
                        hc2._merges_heights, hc3._merges_heights
                    )
            for p in self._ps:
                # will use KDTree and Boruvka
                p1 = Persistable(self._X, measure=w, p=p)
                mps1 = p1._mpspace
                # will use dense MST
                p2 = Persistable(
                    distance_matrix(self._X, self._X, p=p),
                    metric="precomputed",
                    measure=w,
                )
                mps2 = p2._mpspace
                num_components = 1000
                big_X = np.zeros((self._X.shape[0], num_components))
                big_X[:, : self._X.shape[1]] = self._X
                # will use KDTree and Prim
                p3 = Persistable(big_X, measure=w, p=p)
                mps3 = p3._mpspace
                for s0 in self._s0s:
                    for k0 in self._k0s:
                        hc1 = mps1.lambda_linkage([0, k0], [s0, 0])
                        hc2 = mps2.lambda_linkage([0, k0], [s0, 0])
                        hc3 = mps3.lambda_linkage([0, k0], [s0, 0])
                        np.testing.assert_almost_equal(
                            hc1._merges_heights, hc2._merges_heights
                        )
                        np.testing.assert_almost_equal(
                            hc2._merges_heights, hc3._merges_heights
                        )

    def test_hilbert_function(self):
        X = np.array([[0, 0], [1, 0], [1, 1], [3, 0]])
        p = Persistable(X)
        mps = p._mpspace

        ss = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        ks = [0, 1 / 4 + 0.01, 1 / 2, 3 / 4, 1, 1.1]

        res = np.array(
            [
                [4, 4, 2, 2, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ]
        )
        np.testing.assert_almost_equal(mps.hilbert_function(ks, ss, n_jobs=4), res)


class TestHierarchicalClustering(unittest.TestCase):
    def clustering_matrix(self, c):
        mat = np.full((c.shape[0], c.shape[0]), -1)
        for i in range(c.shape[0]):
            for j in range(c.shape[0]):
                if c[i] == c[j] and c[i] != -1:
                    mat[i, j] = 0
        return mat

    def test_persistence_diagram(self):
        heights = np.array([0, 1, 3, 2])
        merges = np.array([[0, 1], [2, 4]])
        merges_heights = np.array([2, 6])
        end = 10
        hc = _HierarchicalClustering(heights, merges, merges_heights, 0, end)
        pd = hc.persistence_diagram()
        res = np.array([[0, 10], [1, 2], [3, 6], [2, 10]])
        np.testing.assert_array_equal(
            pd[np.lexsort(pd.T[::-1])], res[np.lexsort(res.T[::-1])]
        )

        heights = np.array([0, 1, 3, 2, 4])
        merges = np.array([[0, 1], [3, 5], [2, 6]])
        merges_heights = np.array([2, 4, 6])
        end = 10
        hc = _HierarchicalClustering(heights, merges, merges_heights, 0, end)
        pd = hc.persistence_diagram()
        res = np.array([[0, 10], [1, 2], [2, 4], [3, 6], [4, 10]])
        np.testing.assert_array_equal(
            pd[np.lexsort(pd.T[::-1])], res[np.lexsort(res.T[::-1])]
        )

        heights = np.array([0.5, 1, 2, 0])
        merges = np.array([[0, 1], [2, 4], [3, 5]])
        merges_heights = np.array([2, 4, 6])
        end = 10
        hc = _HierarchicalClustering(heights, merges, merges_heights, 0, end)
        pd = hc.persistence_diagram()
        res = np.array([[0.5, 6], [1, 2], [2, 4], [0, 10]])
        np.testing.assert_array_equal(
            pd[np.lexsort(pd.T[::-1])], res[np.lexsort(res.T[::-1])]
        )

        heights = np.array([0, 1])
        merges = np.array([[0, 1]])
        merges_heights = np.array([1])
        end = 10
        hc = _HierarchicalClustering(heights, merges, merges_heights, 0, end)
        pd = hc.persistence_diagram()
        res = np.array([[0, 10]])
        np.testing.assert_array_equal(
            pd[np.lexsort(pd.T[::-1])], res[np.lexsort(res.T[::-1])]
        )

    def test_flattening(self):
        heights = np.array([0, 1, 3, 8])
        merges = np.array([[0, 1], [2, 4]])
        merges_heights = np.array([2, 6])
        end = 10
        hc = _HierarchicalClustering(heights, merges, merges_heights, 0, end)
        c = hc.persistence_based_flattening(0.5)
        res = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(
            self.clustering_matrix(c), self.clustering_matrix(res)
        )

        c = hc.persistence_based_flattening(1.5)
        res = np.array([0, 0, 1, 2])
        np.testing.assert_array_equal(
            self.clustering_matrix(c), self.clustering_matrix(res)
        )


class TestPersistable(unittest.TestCase):
    def test_number_clusters(self):
        n_datapoints = 1000
        n_true_points = int(n_datapoints * 0.7)
        X, _ = datasets.make_blobs(
            n_samples=n_true_points,
            centers=6,
            cluster_std=[0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            random_state=0,
        )
        np.random.seed(0)
        n_noise = n_datapoints - n_true_points
        noise = (np.random.random_sample((n_noise, 2)) - 0.4) * 4
        X = np.vstack((X, noise))

        p = Persistable(X)
        k0 = 0.05
        for s0 in np.linspace(0.1, 0.5, 5):
            for i in list(range(2, 5)):
                c = p.cluster(n_clusters=i, start=[0, k0], end=[s0, 0])
                self.assertEqual(len(set(c[c >= 0])), i)
                c = p.cluster(
                    n_clusters=i,
                    start=[0, k0],
                    end=[s0, 0],
                    extend_clustering_by_hill_climbing=True,
                )
                self.assertEqual(len(set(c[c >= 0])), i)

    def test_number_clusters_quick_cluster(self):
        X, _ = datasets.make_blobs(
            n_samples=1000, centers=3, cluster_std=[0.05, 0.06, 0.07], random_state=1
        )
        p = Persistable(X)
        c = p.quick_cluster()
        self.assertEqual(len(set(c[c >= 0])), 3)
        c = p.quick_cluster(extend_clustering_by_hill_climbing=True)
        self.assertEqual(len(set(c[c >= 0])), 3)

        X, _ = datasets.make_blobs(n_samples=1000, centers=4, random_state=2)
        p = Persistable(X)
        c = p.quick_cluster(n_neighbors=50)
        self.assertEqual(len(set(c[c >= 0])), 4)
        c = p.quick_cluster(n_neighbors=50, extend_clustering_by_hill_climbing=True)
        self.assertEqual(len(set(c[c >= 0])), 4)

        X, _ = datasets.make_blobs(n_samples=1000, centers=5, random_state=3)
        p = Persistable(X)
        c = p.quick_cluster()
        self.assertEqual(len(set(c[c >= 0])), 5)
        c = p.quick_cluster(extend_clustering_by_hill_climbing=True)
        self.assertEqual(len(set(c[c >= 0])), 5)

    def test_hilbert_function(self):
        X = np.array([[0, 0], [1, 0], [1, 1], [3, 0]])
        p = Persistable(X)

        res_ss = [0.0, 0.5625, 1.125, 1.6875, 2.25, 2.8125, 3.375, 3.9375, 4.5]
        res_ks = [
            0.0,
            0.140625,
            0.28125,
            0.421875,
            0.5625,
            0.703125,
            0.84375,
            0.984375,
            1.125,
        ]

        res = np.array(
            [
                [4, 4, 2, 2, 1, 1, 1, 1],
                [4, 4, 2, 2, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ]
        )
        ss, ks, hs = p._compute_hilbert_function(0, 1, 0, 4, granularity=8)

        np.testing.assert_almost_equal(ss, np.array(res_ss))
        np.testing.assert_almost_equal(ks, np.array(res_ks))
        np.testing.assert_almost_equal(hs, res)

    def test_prominence_vineyard(self):
        X = np.array([[0, 0], [1, 0], [1, 1], [3, 0]])
        p = Persistable(X)

        start_end1 = [(0, 0.1), (10, 0)]
        start_end2 = [(0, 1), (10, 0.9)]
        vineyard = p._compute_vineyard(start_end1, start_end2, n_parameters=4)

        vines = vineyard._vineyard_to_vines()
        res_vines = [
            (np.array([0, 1, 2, 3]), np.array([10.0, 9.0, 9.0, 8.0])),
            (np.array([0, 1, 2, 3]), np.array([2.0, 0.0, 0.0, 0.0])),
            (np.array([0, 1, 2, 3]), np.array([1.0, 0.0, 0.0, 0.0])),
            (np.array([0, 1, 2, 3]), np.array([1.0, 0.0, 0.0, 0.0])),
        ]

        res_vine_parts = [
            [(np.array([0, 10.0, 9.0, 9.0, 8.0]), np.array([3, 0, 1, 2, 3]))],
            [(np.array([0, 2.0, 0.0]), np.array([3, 0, 1]))],
            [(np.array([0, 1.0, 0.0]), np.array([3, 0, 1]))],
            [(np.array([0, 1.0, 0.0]), np.array([3, 0, 1]))],
        ]

        for tv, res_tv, res_vp in zip(vines, res_vines, res_vine_parts):
            t, v = tv
            res_t, res_v = res_tv
            np.testing.assert_almost_equal(t, res_t)
            np.testing.assert_almost_equal(v, res_v)
            np.testing.assert_almost_equal(vineyard._vine_parts(v), res_vp)


if __name__ == "__main__":
    unittest.main()
