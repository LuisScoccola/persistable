import unittest
from persistable import Persistable
from persistable.persistable import _HierarchicalClustering, _MetricProbabilitySpace
from scipy.spatial import distance_matrix
from sklearn import datasets
import numpy as np


class TestMetricProbabilitySpace(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self._n = 25
        self._X = np.random.random_sample((self._n,2))
        self._ps = [2,4,8]
        self._s0s = np.linspace(0.1,2,10)
        self._k0s = np.linspace(0.01,0.5,10)
        self._number_different_weights = 3
        self._different_weights = [None]
        for _ in range(self._number_different_weights):
            self._different_weights.append(np.random.random_sample(self._n))

    def test_same_core_distances(self):
        for w in self._different_weights:
            for p in self._ps:
                mps1 = _MetricProbabilitySpace(self._X, p=p, measure=w)
                mps2 = _MetricProbabilitySpace(distance_matrix(self._X,self._X,p=p),metric="precomputed", measure=w)
                mps1.fit()
                mps2.fit()
                for s0 in self._s0s:
                    for k0 in self._k0s:
                        np.testing.assert_almost_equal(mps1.core_distance(np.arange(self._n),s0,k0),mps2.core_distance(np.arange(self._n),s0,k0))

    def test_same_hierarchy(self):
        for w in self._different_weights:
            for p in self._ps:
                mps1 = _MetricProbabilitySpace(self._X, p=p, measure=w)
                mps2 = _MetricProbabilitySpace(distance_matrix(self._X,self._X,p=p),metric="precomputed", measure=w)
                mps1.fit()
                mps2.fit()
                for s0 in self._s0s:
                    for k0 in self._k0s:
                        hc1 = mps1.lambda_linkage(s0,k0)
                        hc2 = mps2.lambda_linkage(s0,k0)
                        #np.testing.assert_array_equal(hc1._merges, hc2._merges)
                        np.testing.assert_almost_equal(hc1._merges_heights, hc2._merges_heights)




class TestHierarchicalClustering(unittest.TestCase):

    def clustering_matrix(self,c):
        mat = np.full((c.shape[0],c.shape[0]), -1)
        for i in range(c.shape[0]):
            for j in range(c.shape[0]):
                if c[i] == c[j] and c[i] != -1:
                    mat[i,j] = 0
        return mat

    def test_persistence_diagram_simple_hierarchy(self):
        heights = np.array([0,1,3])
        merges = np.array([[0,1],[2,3]])
        merges_heights = np.array([2,6])
        end = 10
        hc = _HierarchicalClustering(heights, merges, merges_heights, end)
        pd = hc.persistence_diagram()
        res = np.array([[0,10],[1,2],[3,6]])
        np.testing.assert_almost_equal(np.sort(pd, axis=0), np.sort(res,axis=0))

    def test_flattening_simple_hierarchy(self):
        heights = np.array([0,1,3,8])
        merges = np.array([[0,1],[2,4]])
        merges_heights = np.array([2,6])
        end = 10
        hc = _HierarchicalClustering(heights, merges, merges_heights, end)
        c = hc.persistence_based_flattening(0.5)
        res = np.array([0,1,2,3])
        np.testing.assert_array_equal(self.clustering_matrix(c), self.clustering_matrix(res))

        c = hc.persistence_based_flattening(1.5)
        res = np.array([0,0,1,2])
        np.testing.assert_array_equal(self.clustering_matrix(c), self.clustering_matrix(res))

class TestPersistable(unittest.TestCase):

    def test_number_clusters(self):
        n_datapoints = 1000
        n_true_points = int(n_datapoints * 0.7)
        X, _ = datasets.make_moons(n_samples=n_true_points, noise=0.1, random_state=0)
        np.random.seed(0)
        n_noise = n_datapoints - n_true_points
        noise = (np.random.random_sample((n_noise,2)) - 0.4) * 4
        X = np.vstack((X,noise))

        p = Persistable(X)
        k0 = 0.05
        for s0 in np.linspace(0.1,0.5,5):
            for i in list(range(11,1)) :
                c = p.cluster(num_clusters = i, s0 = s0, k0 = k0)
                print(s0, k0)
                print(i, len(set(c[c>=0])))
                self.assertEqual(len(set(c[c>=0])), i)

if __name__ == "__main__":
    unittest.main()
    