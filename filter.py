from __future__ import division
import numpy as np
import scipy.sparse as ss


class Filter(object):
    base_file = "./ml-100k/u1.base"
    test_file = "./ml-100k/u1.test"
    users = None
    items = None
    ratings = None
    rating_matrix = None
    base_entries = 80000
    test_entries = 20000

    def __init__(self):
        self.build_matrix()

    def build_matrix(self):
        users = np.zeros(self.base_entries)
        movies = np.zeros(self.base_entries)
        ratings = np.zeros(self.base_entries)
        count = 0
        with open(self.base_file, "r") as f:
            for line in f:
                line = [int(word) for word in line.strip().split("\t")]
                users[count] = line[0] - 1
                movies[count] = line[1] - 1
                ratings[count] = line[2]
                count += 1
        self.rating_matrix = ss.csr_matrix((ratings, (movies, users)))
        self.items = movies
        self.users = users
        self.ratings = ratings

    @classmethod
    def adjust_vector(cls, vector):
        mean = np.average(vector)
        for i in xrange(vector.shape[0]):
            if not vector[i] == 0:
                vector[i] -= mean

    def find_cosine_similarity(self, item1, item2):
        user1_vector = self.rating_matrix[item1, :].toarray()[0]
        user2_vector = self.rating_matrix[item2, :].toarray()[0]
        Filter.adjust_vector(user1_vector)
        Filter.adjust_vector(user2_vector)
        user1_vector = user1_vector/np.linalg.norm(user1_vector)
        user2_vector = user2_vector / np.linalg.norm(user2_vector)

        return user1_vector.dot(user2_vector)

    def find_top_k_neighbors(self, item, k=100):
        similarity = []
        for item_id in xrange(self.rating_matrix.shape[0]):
            if not item == item_id:
                if np.isnan(self.find_cosine_similarity(item, item_id)):
                    print "this {}".format(item_id)
                    continue
                similarity.append(self.find_cosine_similarity(item, item_id))
        similar_vectors = np.array(similarity).argsort()[::-1][0:k]
        return similar_vectors

    def baseline_estimate(self, user, item):
        return 0

    def predict_rating(self, user, item, baseline=False):
        if baseline:
            correction = self.baseline_estimate(user, item)
        else:
            correction = 0
        similar_items = self.find_top_k_neighbors(item)
        denominator = 0
        numerator = 0
        item_vector = self.rating_matrix[item, :].toarray()[0]
        user_rating = np.array([x[0] for x in
                               list(self.rating_matrix[:, user].toarray())])
        for i in similar_items:
            rating = user_rating[i]
            similarity = self.find_cosine_similarity(item, i)
            numerator += similarity * rating
            denominator += similarity
        return correction + (numerator/denominator) + np.average(item_vector)

if __name__ == "__main__":
    filterObj = Filter()
    print filterObj.predict_rating(0, 0)
