from __future__ import division
import numpy as np
import scipy.sparse as ss
from datetime import datetime


class Filter(object):
    base_file = "./ml-100k/u1.base"
    test_file = "./ml-100k/u1.test"
    users = None
    items = None
    ratings = None
    rating_matrix = None
    base_entries = 80000
    test_entries = 20000
    related_items = {}

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
                sim = self.find_cosine_similarity(item, item_id)
                if np.isnan(sim):
                    continue
                similarity.append(sim)
        similar_vectors = np.array(similarity).argsort()[::-1][0:k]
        return similar_vectors

    def baseline_estimate(self, user, item):
        return 0

    def predict_rating(self, user, item, baseline=False):
        if baseline:
            correction = self.baseline_estimate(user, item)
        else:
            correction = 0
        if item in self.related_items:
            print "there"
            similar_items = self.related_items[item]
        else:
            similar_items = self.find_top_k_neighbors(item, k=200)
            self.related_items[item] = similar_items
        denominator = 0
        numerator = 0
        item_vector = self.rating_matrix[item, :].toarray()[0]
        user_rating = np.array([x[0] for x in
                               list(self.rating_matrix[:, user].toarray())])
        self.adjust_vector(user_rating)
        for i in similar_items:
            rated = user_rating[i]
            if rated == 0:
                continue
            similarity = self.find_cosine_similarity(item, i)
            numerator += similarity * rated
            denominator += similarity
        try:
            predicted_rating = correction + (numerator/denominator) + np.average(
                item_vector)
        except Exception:
            predicted_rating = correction + np.average(item_vector)

        return predicted_rating


if __name__ == "__main__":
    filterObj = Filter()
    print datetime.now().time()
    rmse = 0
    try:
        for j in xrange(filterObj.users.shape[0]):
            print j, rmse
            user = filterObj.users[j]
            item = filterObj.items[j]
            rating = filterObj.ratings[j]
            predicted = filterObj.predict_rating(user, item)
            rmse += (predicted - rating) ** 2
    except KeyboardInterrupt:
        print datetime.now().time()
    rmse = rmse/filterObj.users.shape[0]
    print rmse ** 0.5
    print datetime.now().time()

# 20:59:46.446547
# rmse = 1.04799110545
