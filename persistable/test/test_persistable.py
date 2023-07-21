# Authors: Luis Scoccola
# License: 3-clause BSD

import unittest
from persistable import Persistable
from persistable.persistable import _HierarchicalClustering, _MetricSpace
from persistable.signed_betti_numbers import signed_betti
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from sklearn import datasets
from sklearn.datasets import make_blobs
import numpy as np


class TestDegreeRipsBifiltration(unittest.TestCase):
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

    def test_k_can_be_one(self):
        """Check that k can be 1 when intialized with all neighbors"""
        n = 2000
        X = np.random.random_sample((n, 2))

        p = Persistable(X, n_neighbors="all")
        bf = p._bifiltration
        s0 = np.infty
        k0 = 1
        bf._core_distance(np.arange(n), s0, k0)

    def test_core_distances(self):
        """Check that the _core_distance method returns the correct answer"""
        n = 4
        X = np.array([[0, 0], [1, 0], [1, 1], [3, 0]])
        p = Persistable(X)
        bf = p._bifiltration

        s0 = 2
        k0 = 0.5
        res = np.array([1, 1, 1, 1])
        np.testing.assert_almost_equal(bf._core_distance(np.arange(n), s0, k0), res)

        s0 = 1
        k0 = 0.5
        res = np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2])
        np.testing.assert_almost_equal(bf._core_distance(np.arange(n), s0, k0), res)

        s0 = np.infty
        k0 = 1
        res = np.array([3, 2, np.sqrt(5), 3])
        np.testing.assert_almost_equal(bf._core_distance(np.arange(n), s0, k0), res)

        s0 = np.infty
        k0 = 0.5
        res = np.array([1, 1, 1, 2])
        np.testing.assert_almost_equal(bf._core_distance(np.arange(n), s0, k0), res)

        p = Persistable(X, measure=np.array([0.25, 0.25, 0.25, 0.25]))
        bf = p._bifiltration
        s0 = np.infty
        k0 = 0.3
        res = np.array([1, 1, 1, 2])
        np.testing.assert_almost_equal(bf._core_distance(np.arange(n), s0, k0), res)

    def test_same_core_distances(self):
        """ Check that the _core_distance method returns the same answer \
            when using a precomputed distance and a Minkowski distance """
        for w in self._different_weights:
            for p in self._ps:
                p1 = Persistable(self._X, measure=w, p=p)
                bf1 = p1._bifiltration
                p2 = Persistable(
                    distance_matrix(self._X, self._X, p=p),
                    metric="precomputed",
                    measure=w,
                )
                bf2 = p2._bifiltration
                for s0 in self._s0s:
                    for k0 in self._k0s:
                        np.testing.assert_almost_equal(
                            bf1._core_distance(np.arange(self._n), s0, k0),
                            bf2._core_distance(np.arange(self._n), s0, k0),
                        )

    def test_same_hierarchy(self):
        """ Check that lambda_linkage returns the same answer when using precomputed \
            distance, KDTree, BallTree, and Boruvka, Prim """
        # test Boruvka, Prim, and dense Prim
        for w in self._different_weights:
            V = np.ones(self._X.shape[1])
            # will use BallTree and Boruvka
            p1 = Persistable(self._X, metric="seuclidean", measure=w, V=V)
            bf1 = p1._bifiltration
            # will use dense MST
            p2 = Persistable(
                cdist(self._X, self._X, metric="seuclidean", V=V),
                metric="precomputed",
                measure=w,
            )
            bf2 = p2._bifiltration
            num_components = 1000
            big_X = np.zeros((self._X.shape[0], num_components))
            big_X[:, : self._X.shape[1]] = self._X
            # will use BallTree and Prim
            V2 = np.ones(big_X.shape[1])
            p3 = Persistable(big_X, metric="seuclidean", measure=w, V=V2)
            bf3 = p3._bifiltration
            for s0 in self._s0s:
                for k0 in self._k0s:
                    hc1 = bf1.lambda_linkage([0, k0], [s0, 0])
                    hc2 = bf2.lambda_linkage([0, k0], [s0, 0])
                    hc3 = bf3.lambda_linkage([0, k0], [s0, 0])
                    np.testing.assert_almost_equal(
                        hc1._merges_heights, hc2._merges_heights
                    )
                    np.testing.assert_almost_equal(
                        hc2._merges_heights, hc3._merges_heights
                    )
            for p in self._ps:
                # will use KDTree and Boruvka
                p1 = Persistable(self._X, measure=w, p=p)
                bf1 = p1._bifiltration
                # will use dense MST
                p2 = Persistable(
                    distance_matrix(self._X, self._X, p=p),
                    metric="precomputed",
                    measure=w,
                )
                bf2 = p2._bifiltration
                num_components = 1000
                big_X = np.zeros((self._X.shape[0], num_components))
                big_X[:, : self._X.shape[1]] = self._X
                # will use KDTree and Prim
                p3 = Persistable(big_X, measure=w, p=p)
                bf3 = p3._bifiltration
                for s0 in self._s0s:
                    for k0 in self._k0s:
                        hc1 = bf1.lambda_linkage([0, k0], [s0, 0])
                        hc2 = bf2.lambda_linkage([0, k0], [s0, 0])
                        hc3 = bf3.lambda_linkage([0, k0], [s0, 0])
                        np.testing.assert_almost_equal(
                            hc1._merges_heights, hc2._merges_heights
                        )
                        np.testing.assert_almost_equal(
                            hc2._merges_heights, hc3._merges_heights
                        )

    def test_hilbert_function(self):
        """Check that _hilbert_function method returns a correct answer"""
        X = np.array([[0, 0], [1, 0], [1, 1], [3, 0]])
        p = Persistable(X)
        bf = p._bifiltration

        ss = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
        ks = [1, 3 / 4, 1 / 2, 1 / 4 + 0.01, 0]

        res = np.array(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [4, 4, 2, 2, 1, 1, 1, 1],
            ]
        ).T
        np.testing.assert_almost_equal(
            bf._hilbert_function(ss, ks, reduced=False, n_jobs=4), res
        )

        X = np.array([[0, 0], [1, 0], [1, 1], [3, 0]])
        p = Persistable(X)

        # res_ss = [0.0, 0.5625, 1.125, 1.6875, 2.25, 2.8125, 3.375, 3.9375, 4.5]
        res_ss = [
            0.0,
            0.5714286,
            1.1428571,
            1.7142857,
            2.2857143,
            2.8571429,
            3.4285714,
            4.0,
        ]
        res_ks = [
            1,
            0.8571429,
            0.7142857,
            0.5714286,
            0.4285714,
            0.2857143,
            0.1428571,
            0.0,
        ]

        res = np.array(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [4, 4, 2, 2, 1, 1, 1, 1],
                [4, 4, 2, 2, 1, 1, 1, 1],
            ]
        ).T
        ss, ks, hs, _ = p._bifiltration.hilbert_function_on_regular_grid(
            0, 4, 1, 0, granularity=8
        )

        np.testing.assert_almost_equal(ss, np.array(res_ss))
        np.testing.assert_almost_equal(ks, np.array(res_ks))
        np.testing.assert_almost_equal(hs, res)

    def test_vertical_slice(self):
        """ Check that the persistence diagram of lambda_linkage is correct \
            for some vertical slices """
        X = np.array(
            [
                [0, 0],
                [1, 1],
                [1, 0],
                [1, -1],
                [2, 0],
                [3, 0],
                [4, 0],
                [5, 1],
                [5, 0],
                [5, -1],
                [6, 0],
            ]
        )
        p = Persistable(X, debug=True)
        bf = p._bifiltration
        hc = bf._lambda_linkage_vertical(1, 1, 0)
        res = np.array(
            [
                0.72727273,
                0.72727273,
                0.72727273,
                0.72727273,
                0.8181818,
                0.8181818,
                0.8181818,
                0.8181818,
                0.8181818,
                0.8181818,
            ]
        )

        np.testing.assert_almost_equal(res, hc._merges_heights)

        dist_mat = distance_matrix(X, X)
        p = Persistable(dist_mat, debug=True, metric="precomputed")
        bf = p._bifiltration
        hc = bf._lambda_linkage_vertical(1, 1, 0)
        np.testing.assert_almost_equal(res, hc._merges_heights)

        np.testing.assert_almost_equal(
            np.array([[0.0, 0.1010101], [0.0, 0.1010101]]),
            bf.lambda_linkage(
                [1.777777777777778, 0.393939393939394],
                [1.777777777777778, 0.29292929292929293],
            ).persistence_diagram(),
        )
        np.testing.assert_almost_equal(
            np.array([[0.0, 0.1], [0.0, 0.1]]),
            bf.lambda_linkage([1.3, 0.4], [1.3, 0.3]).persistence_diagram(),
        )
        np.testing.assert_almost_equal(
            np.array([[0.0, 0.1], [0.0, 0.1]]),
            bf.lambda_linkage([1.5, 0.4], [1.5, 0.3]).persistence_diagram(),
        )
        np.testing.assert_almost_equal(
            np.array([[0.0, 0.1], [0.0, 0.1]]),
            bf.lambda_linkage([1.7, 0.4], [1.7, 0.3]).persistence_diagram(),
        )

    def test_rank_invariant(self):
        """Check that the rank_invariant method returns correct answers"""
        X = np.array(
            [
                [0, 0],
                [1, 1],
                [1, 0],
                [1, -1],
                [2, 0],
                [3, 0],
                [4, 0],
                [5, 1],
                [5, 0],
                [5, -1],
                [6, 0],
            ]
        )
        p = Persistable(X, debug=True)

        ks = [0.4, 0.3]
        ss = [1.3, 1.5]
        ri = p._bifiltration._rank_invariant(ss, ks, reduced=False)
        np.testing.assert_almost_equal(
            ri,
            [
                [[[2, 2], [2, 2]], [[0, 2], [0, 2]]],
                [[[0, 0], [2, 2]], [[0, 0], [0, 2]]],
            ],
        )

        ks = [0.3, 0.2]
        ss = [1.5, 2.5]
        ri = p._bifiltration._rank_invariant(ss, ks, reduced=False)
        np.testing.assert_almost_equal(
            ri,
            [
                [[[2, 1], [1, 1]], [[0, 1], [0, 1]]],
                [[[0, 0], [1, 1]], [[0, 0], [0, 1]]],
            ],
        )

        ks = [0.4, 0.3, 0.2]
        ss = [1.3, 1.5, 2.5]
        ri = p._bifiltration._rank_invariant(ss, ks, reduced=False)
        res = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for i_ in range(i, 3):
                    for j_ in range(j, 3):
                        res[i, j, i_, j_] = (
                            2 if i < 2 and j < 2 and i_ < 2 and j_ < 2 else 1
                        )
        np.testing.assert_almost_equal(ri, res)


class TestMetricSpace(unittest.TestCase):
    def test_subsampling(self):
        """ Check that subsampling with fast metric produces the same \
            subsample as subsampling with distance matrix """

    X = make_blobs(
        n_samples=1000, n_features=2, centers=3, random_state=6, cluster_std=1.5
    )[0]
    dm = distance_matrix(X, X, p=2)
    ms = _MetricSpace(dm, "precomputed")
    Y, reps = ms.close_subsample(100, seed=0)

    ms2 = _MetricSpace(X, metric="minkowski")
    Y2, reps2 = ms2.close_subsample(100, seed=0)

    np.testing.assert_array_equal(Y, Y2)
    np.testing.assert_array_equal(reps, reps2)


class TestHierarchicalClustering(unittest.TestCase):
    def clustering_matrix(self, c):
        mat = np.full((c.shape[0], c.shape[0]), -1)
        for i in range(c.shape[0]):
            for j in range(c.shape[0]):
                if c[i] == c[j] and c[i] != -1:
                    mat[i, j] = 0
        return mat

    def test_persistence_diagram(self):
        """Check that persistence_diagram method returns correct answers"""
        heights = np.array([0, 1, 3, 2])
        merges = np.array([[0, 1], [2, 0]])
        merges_heights = np.array([2, 6])
        end = 10
        hc = _HierarchicalClustering(heights, merges, merges_heights, 0, end)
        pd = hc.persistence_diagram()
        res = np.array([[0, 10], [1, 2], [3, 6], [2, 10]])
        np.testing.assert_array_equal(
            pd[np.lexsort(pd.T[::-1])], res[np.lexsort(res.T[::-1])]
        )

        heights = np.array([0, 1, 3, 2, 4])
        merges = np.array([[0, 1], [3, 0], [2, 3]])
        merges_heights = np.array([2, 4, 6])
        end = 10
        hc = _HierarchicalClustering(heights, merges, merges_heights, 0, end)
        pd = hc.persistence_diagram()
        res = np.array([[0, 10], [1, 2], [2, 4], [3, 6], [4, 10]])
        np.testing.assert_array_equal(
            pd[np.lexsort(pd.T[::-1])], res[np.lexsort(res.T[::-1])]
        )

        heights = np.array([0.5, 1, 2, 0])
        merges = np.array([[0, 1], [2, 0], [3, 0]])
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
        """Check that persistence_based_flattening method returns correct answers"""
        heights = np.array([0, 1, 3, 8])
        merges = np.array([[0, 1], [2, 0]])
        merges_heights = np.array([2, 6])
        end = 10
        hc = _HierarchicalClustering(heights, merges, merges_heights, 0, end)
        c = hc.persistence_based_flattening(
            0.5, flattening_mode="conservative", keep_low_persistence_clusters=False
        )
        res = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(
            self.clustering_matrix(c), self.clustering_matrix(res)
        )

        c = hc.persistence_based_flattening(
            1.5, flattening_mode="conservative", keep_low_persistence_clusters=False
        )
        res = np.array([0, 0, 1, 2])
        np.testing.assert_array_equal(
            self.clustering_matrix(c), self.clustering_matrix(res)
        )


class TestPersistable(unittest.TestCase):
    def test_number_clusters(self):
        """Check that cluster method returns the correct number of clusters"""
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
                c = p.cluster(
                    n_clusters=i,
                    start=[0, k0],
                    end=[s0, 0],
                    flattening_mode="conservative",
                )
                self.assertEqual(len(set(c[c >= 0])), i)
                c = p.cluster(
                    n_clusters=i,
                    start=[0, k0],
                    end=[s0, 0],
                    flattening_mode="conservative",
                )
                self.assertEqual(len(set(c[c >= 0])), i)

    def test_number_clusters_quick_cluster(self):
        """Check that quick_cluster method returns the correct number of clusters"""
        X, _ = datasets.make_blobs(
            n_samples=1000, centers=3, cluster_std=[0.05, 0.06, 0.07], random_state=1
        )
        p = Persistable(X)
        c = p.quick_cluster()
        self.assertEqual(len(set(c[c >= 0])), 3)

        X, _ = datasets.make_blobs(n_samples=1000, centers=4, random_state=2)
        p = Persistable(X)
        c = p.quick_cluster(n_neighbors=50)
        self.assertEqual(len(set(c[c >= 0])), 4)

        X, _ = datasets.make_blobs(n_samples=1000, centers=5, random_state=3)
        p = Persistable(X)
        c = p.quick_cluster()
        self.assertEqual(len(set(c[c >= 0])), 5)


class TestVineyard(unittest.TestCase):
    def test_prominence_vineyard(self):
        """Check that _vineyard_to_vines and _vine_parts methods return a correct answer"""
        X = np.array([[0, 0], [1, 0], [1, 1], [3, 0]])
        p = Persistable(X)

        start_end1 = [(0, 0.1), (10, 0)]
        start_end2 = [(0, 1), (10, 0.9)]
        vineyard = p._bifiltration.linear_vineyard(
            start_end1, start_end2, n_parameters=4
        )

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


class TestBettiNumbers(unittest.TestCase):
    def test_signed_betti(self):
        """ Check that signed_betti method returns correct answers in \
            dimensions (shape) 1, 2, 3, 4 """

        np.random.seed(0)
        N = 4

        # test 1D
        for _ in range(N):
            a = np.random.randint(10, 30)

            f = np.random.randint(0, 40, size=(a))
            sb = signed_betti(f)

            check = np.zeros(f.shape)
            for i in range(f.shape[0]):
                for i_ in range(0, i + 1):
                    check[i] += sb[i_]

            np.testing.assert_equal(check, f)

        # test 2D
        for _ in range(N):
            a = np.random.randint(10, 30)
            b = np.random.randint(10, 30)

            f = np.random.randint(0, 40, size=(a, b))
            sb = signed_betti(f)

            check = np.zeros(f.shape)
            for i in range(f.shape[0]):
                for j in range(f.shape[1]):
                    for i_ in range(0, i + 1):
                        for j_ in range(0, j + 1):
                            check[i, j] += sb[i_, j_]

            np.testing.assert_equal(check, f)

        # test 3D
        for _ in range(N):
            a = np.random.randint(5, 10)
            b = np.random.randint(5, 10)
            c = np.random.randint(5, 10)

            f = np.random.randint(0, 40, size=(a, b, c))
            sb = signed_betti(f)

            check = np.zeros(f.shape)
            for i in range(f.shape[0]):
                for j in range(f.shape[1]):
                    for k in range(f.shape[2]):
                        for i_ in range(0, i + 1):
                            for j_ in range(0, j + 1):
                                for k_ in range(0, k + 1):
                                    check[i, j, k] += sb[i_, j_, k_]

            np.testing.assert_equal(check, f)

        # test 4D
        for _ in range(N):
            a = np.random.randint(5, 10)
            b = np.random.randint(5, 10)
            c = np.random.randint(5, 10)
            d = np.random.randint(5, 10)

            f = np.random.randint(0, 40, size=(a, b, c, d))
            sb = signed_betti(f)

            check = np.zeros(f.shape)
            for i in range(f.shape[0]):
                for j in range(f.shape[1]):
                    for k in range(f.shape[2]):
                        for l in range(f.shape[3]):
                            for i_ in range(0, i + 1):
                                for j_ in range(0, j + 1):
                                    for k_ in range(0, k + 1):
                                        for l_ in range(0, l + 1):
                                            check[i, j, k, l] += sb[i_, j_, k_, l_]

            np.testing.assert_equal(check, f)


if __name__ == "__main__":
    unittest.main()
