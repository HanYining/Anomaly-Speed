import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
from iforest.iforest import *
import unittest
import numpy as np

class Test_iTree(unittest.TestCase):

    def test_maximum_Edge_Path(self):
        data = np.random.multivariate_normal([0,0],[[1,0],[0,3]], 1000)
        a = isolation_forest(data).iTree(node(),data,0, 8)
        self.assertTrue(self._maxEdges(a)<=8)

    def _maxEdges(self, root):
        # recursive structure to get the maximum edge path from root to leaf in
        # an isolation tree. should be smaller or equal to the the maximum edge requirement
        # when constructing the tree.
        if not root:
            return 0
        return max(self._maxEdges(root.left), self._maxEdges(root.right))

    def test_tree(self):
        data = np.random.multivariate_normal([0,0],[[1,0],[0,3]], 1000)
        a = isolation_forest(data).iTree(node(),data,0, 8)
        self._test_helper(a, data)

    def _test_helper(self, root, data):
        # use recursive testing to test all the node in a isolation tree satisfy the
        # constraints in class structure.
        self.assertTrue(isinstance(root, node))
        self.assertTrue(root.split_axis in [0, 1])
        self.assertTrue(root.split_val < max(data[:, root.split_axis]) and root.split_val > min(data[:, root.split_axis]))
        # as long as we are not at the bottom 2 levels of the isolation tree, the above conditions should
        # all be satisfied
        # so check root.left.left is not None, basically checks if root.left.right is not None as well
        # since this is a full binary tree.
        if root.left.left:
            return self._test_helper(root.left, data)
        if root.right.right:
            return self._test_helper(root.right, data)
        # since this is a full binary tree, it does not really matter
        # which direction we go, when we reach the last but one level.
        # so this contains root.right and not root.left.left and well
        if root.left and not root.left.left and not root.right.right:
            self.assertTrue(root.left.split_axis is None and root.left.split_val is None)
            self.assertTrue(root.right.split_axis is None and root.right.split_val is None and
                            root.left.right is None and root.right.left is None )

    def test_anomaly_separate(self):
        # this test is a little bit risky
        # here I assume if given a perfect data, which is a closely cluster group of normal points
        # and a small number of clearly out of cluster anomaly points
        # the corresponding anomaly scores should differ at least 0.1
        data = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 1000)
        data2 = np.random.multivariate_normal([100,100], [[2,0],[0,2]], 10)
        data = np.vstack([data, data2])
        scores = isolation_forest(data).evaluate_forest(data, 8)
        self.assertTrue(np.mean(scores[1000:]) > np.mean(scores[:1000]) + 0.1)
        self.assertTrue(all([score>0 and score<1 for score in scores]))

    def test_Ntree(self):
        # the number of trees should be equal to 100
        data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1000)
        forest = isolation_forest(data)
        self.assertTrue(len(forest) == 100)

if __name__ == '__main__':

    unittest.main()
