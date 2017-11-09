from __future__ import division
import numpy as np
import scipy.sparse as ss


class Filter(object):
    base_file = "./ml-100k/u1.base"
    test_file = "./ml-100k/u1.test"
    base_entries = 80000
    test_entries = 20000

    def __init__(self):
        self.build_matrix()

    def build_matrix(self):
        rows = np.zeros(self.base_entries)
        columns = np.zeros(self.base_entries)
        ratings = np.zeros(self.base_entries)
        count = 0
        with open(self.base_file, "r") as f:
            for line in f:
                line = [int(word) for word in line.strip().split("\t")]
                rows[count] = line[0]
                columns[count] = line[1]
                ratings[count] = line[2]
                count += 1
        np.save("rows.npy")
        np.save("columns.npy")

if __name__ == "__main__":
    Filter()
