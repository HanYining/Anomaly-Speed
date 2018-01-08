import random
import math
import numpy as np

class node():

    def __init__(self, axis=None, val=None, left=None, right=None, cnt=None):
        """
        external node here are defined as a general node with the split_axis and split_val attribute equal to None
        :param axis: the randomly selected splitting axis  
        :param val: the randomly selected splitting value 
        :param left: the left child of the node
        :param right: the right child of the node
        :param cnt: # of data points at that node
        """
        self.split_axis = axis
        self.split_val = val
        self.left = left
        self.right = right
        self.cnt = cnt

class _isolation_iterator:
    def __init__(self, theForest):
        self._theForest = theForest
        self._curNode = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._curNode < len(self._theForest):
            entry = self._theForest[self._curNode]
            self._curNode += 1
            return entry
        else:
            raise StopIteration

class isolation_forest():

    def __init__(self, data, sub_sample_size = 256, forest_size = 100, maximum_depth = 8):
        """
        :param data: the data used to construct the isolation forest  
        :param sub_sample_size: the size of data used to build a single isolation tree
        :param forest_size: the number of isolation tree in the iForest
        :param maximum_depth: the maximum length of a isolation tree.
        """
        self._attributes = data.shape[1]
        self._sub_sample_size = sub_sample_size
        self._forest_size = forest_size
        self._forest = []

        for i in range(self._forest_size):
            sub_data = np.asarray(random.sample(data.tolist(), self._sub_sample_size))
            root = node()
            self._forest.append(self.iTree(root, sub_data, 0, maximum_depth))

    def get_forest(self):
        return self._forest

    def __len__(self):
        return len(self._forest)

    def __iter__(self):
        return _isolation_iterator(self._forest)

    def iTree(self, root, sub_data, e, l):

        if e>=l or sub_data.shape[0] <=1 or all([len(set(val))==1 for val in sub_data.T]):
            return node(cnt=sub_data.shape[0])

        splitable_axis = [ i for i,val in enumerate(sub_data.T) if max(val)>min(val)]
        random_axis =  splitable_axis[random.randint(0, len(splitable_axis)-1)]

        lower, upper = min(sub_data[:,random_axis]), max(sub_data[:, random_axis])
        random_split = random.uniform(lower, upper)

        data_lower = sub_data[sub_data[:,random_axis]<random_split]
        data_upper = sub_data[sub_data[:,random_axis]>=random_split]
        root = node(random_axis, random_split, cnt = sub_data.shape[0])
        root.left = self.iTree(root.left, data_lower, e+1, l)
        root.right = self.iTree(root.right, data_upper, e+1, l)

        return root

    def _get_path_len(self, root, data, hlim):
        # compute how many edges we need to traverse until we reach an external node.
        # external node here are defined by a node without a split_axis value.
        h=0
        while root.split_val:
           h += 1
           if h>=hlim:
               break
           if data[root.split_axis] < root.split_val:
               root = root.left
           else:
               root = root.right
        return h, root

    def evaluate_forest(self, data, hlim):
        # evaluate the dataset and estimate an anomaly score for each observation.
        # return a list of the corresponding anomaly scores.
        path_len = []
        for point in data:
            h_committee = []
            for root in self:
                h, root = self._get_path_len(root, point, hlim)
                h_committee.append(h + _c(root.cnt))
            path_len.append(h_committee)
        return _cal_anomaly(path_len, self._sub_sample_size)

def _H(x):
    return math.log(x) + 0.5772156649

def _c(x):
    if x> 2:
        path_length = 2 * _H(x - 1) - 2 * (x - 1) / x
    elif x == 2:
        path_length = 1
    else:
        path_length = 0
    return path_length

def _cal_anomaly(lengths, sub_samplesize):
    return [2**(-np.mean(val)/_c(sub_samplesize)) for val in lengths]

