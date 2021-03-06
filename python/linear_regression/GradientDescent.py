import numpy as np
import utils as utils


class GradientDescent:
    def __init__(self, alpha, training_features, training_values, min_diff=0.0001, debug=False):
        self.alpha = alpha
        self.debug = debug
        self.min_diff = min_diff
        self.first_theta = None
        if min_diff is None:
            self.min_diff = 0

        if debug:
            print("Min diff: ", self.min_diff)

        self.training_features = training_features
        self.training_values = training_values
        self.theta_values = [1] * (len(self.training_features[0])+1)
        self.T = np.matrix.transpose(self.training_features)
        self.T = self.T.astype(float)
        self.feature_scaling_coeficient = None
        if not all(number == 1 for number in self.T[0]):
            v = [1] * (len(self.T[0]))
            self.T = np.insert(self.T, 0, np.array(v), axis=0)

    def feature_scaling(self):
        self.T, self.feature_scaling_coeficient = utils.feature_scaling(self.T)
        self.training_features = np.matrix.transpose(self.T)

    def lrm_descent(self):
        self.theta_values = [1] * (len(self.training_features[0]))
        X = self.training_features
        if not all(np.transpose(X)[0]) == 1:
            X = np.transpose(np.insert(np.transpose(X), 0, [1]*(len(self.training_values)), axis=0))
        cj = utils.cost_function(self.theta_values, X, self.training_values)
        J = 999999999999999999999
        while abs(J-cj) > self.min_diff:
            J = cj
            self.theta_values = utils.minimized_theta(self.theta_values, X, self.alpha, self.training_values)
            cj = utils.cost_function(self.theta_values, X, self.training_values)
        return J

    def use_function(self, vector):
        vector = np.insert(vector, 0, 1)
        new_vector = []
        if self.feature_scaling_coeficient is not None:
            for i in range(len(vector)):
                new_vector.append(utils.feature_scaling_corrector(vector[i], self.feature_scaling_coeficient[i]))
        else:
            new_vector = vector
        print(new_vector)
        return utils.hypothesis(self.theta_values, new_vector)



'''features = np.array(
    [
        [5, 1500, 2],
        [3, 5200, 3],
        [3, 7001, 6]
    ])
'''
features = np.array(
    [
        [1],[2],[3],[4],[5],[6],[7],[8],[9],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23]
    ]
)
#y = np.array([2, 3, 4]).astype(float)
y = np.array([610, 575, 574, 781,617,503,696,800,886,602,968,842,1006,1319,1302,1435,1459,1439,1380,1374,1636,1766]).astype(float)
print len(y)
print len(features)
alpha = 0.01
gd = GradientDescent(alpha, features, y, min_diff=0.000000000000000000001, debug=True)
gd.feature_scaling()
print("Final Cost =", gd.lrm_descent())
'''print("Predicted result for [5, 1500, 2] =", gd.use_function([5, 1500, 2]), "True result is 2. The difference is =",
      2 - gd.use_function([5, 1500, 2]))
print("Predicted result for [3, 5200, 3] =", gd.use_function([3, 5200, 3]), "True result is 3. The difference is =",
      3 - gd.use_function([3, 5200, 3]))
print("Predicted result for [3, 7001, 6] =", gd.use_function([3, 7001, 6]), "True result is 4. The difference is =",
      4 - gd.use_function([3, 7001, 6]))'''
ne_theta = utils.normal_equation(features, y)
