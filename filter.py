from __future__ import division
import time
import numpy as np
import scipy.sparse as ss
from datetime import datetime
import shelve
from contextlib import closing


class Filter(object):
    base_file = "./ml-100k/u1.base"
    test_file = "./ml-100k/u1.test"
    users = None
    items = None
    ratings = None
    rating_matrix = None
    base_entries = 80000
    test_entries = 20000
    related_items = "items_dict.db"
    mean = 0

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
                self.mean += ratings[count]
                count += 1
        self.rating_matrix = ss.csr_matrix((ratings, (movies, users)))
        self.items = movies
        self.users = users
        self.ratings = ratings
        self.mean = self.mean/count

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
        similarities = {}
        for vector in similar_vectors:
            similarities[vector] = similarity[vector]
        return similarities

    def average(self, vector):
        sum = 0
        count = 0
        for i in vector:
            if not i == 0:
                count += 1
                sum += i

        try:
            ave = sum/count
        except ZeroDivisionError:
            ave = 0
        return ave

    def baseline_estimate(self, user, item):
        item_vector = self.rating_matrix[item, :].toarray()[0]
        item_mean = self.average(item_vector)
        user_rating = np.array([x[0] for x in
                                list(self.rating_matrix[:, user].toarray())])
        user_mean = self.average(user_rating)
        return self.mean + (user_mean - self.mean) + (item_mean - self.mean)

    def predict_rating(self, user, item, baseline=False):
        if baseline:
            correction = self.baseline_estimate(user, item)
        else:
            correction = 0
        related = shelve.open(self.related_items)
        similar_items = related[str(int(item))]
        denominator = 0
        numerator = 0
        user_rating = np.array([x[0] for x in
                               list(self.rating_matrix[:, user].toarray())])
        for i in similar_items:
            rated = user_rating[i]
            if rated == 0:
                continue
            similarity = similar_items[i]
            if baseline:
                rated = rated - self.baseline_estimate(user, i)
            numerator += similarity * rated
            denominator += similarity
        try:
            predicted_rating = correction + (numerator/denominator)
        except Exception:
            predicted_rating = correction

        return predicted_rating

    def save_similarities(self):
        print datetime.now().time()  # 10:20:46.813193
        with closing(shelve.open(self.related_items)) as db:
            for j in xrange(1682):
                print j
                db[str(j)] = self.find_top_k_neighbors(j)
        print datetime.now().time()

    def rmse(self, test_movies, test_users, test_ratings):
        rmse_default = 0
        rmse_baseline = 0
        count = 0
        try:
            for j in xrange(test_ratings.shape[0]):
                print j
                count += 1
                user = test_users[j]
                item = test_movies[j]
                rating = test_ratings[j]
                predicted = self.predict_rating(user, item)
                predicted_b = self.predict_rating(user, item, baseline=True)
                rmse_default += (predicted - rating) ** 2
                rmse_baseline += (predicted_b - rating) ** 2
        except KeyboardInterrupt:
            print "interrupted, exiting"
        rmse_default = rmse_default / count
        rmse_baseline = rmse_baseline / count
        return rmse_default ** 0.5, rmse_baseline ** 0.5

if __name__ == "__main__":
    filterObj = Filter()
    test_movies = np.zeros(20000)
    test_users = np.zeros(20000)
    test_ratings = np.zeros(20000)
    count = 0
    with open(filterObj.test_file, "r") as f:
        for line in f:
            line = [int(word) for word in line.strip().split("\t")]
            test_users[count] = line[0] - 1
            test_movies[count] = line[1] - 1
            test_ratings[count] = line[2]
            count += 1
    start_time = time.time()
    default_rmse, baseline_rmse = \
        filterObj.rmse(test_movies, test_users, test_ratings)

    print default_rmse, baseline_rmse
    print "time taken is {}".format(time.time() - start_time)
# rmse = 2.121654, rmse_baseline = 0.964939487747
