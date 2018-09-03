import numpy as np
import utils as utils


class GradientDescent():
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
        self.theta_values = [1] * (len(self.training_values)+1)
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
        new_theta = self.theta_values
        # diff = abs(sum(new_theta)-sum(self.theta_values))
        diff = self.min_diff+1
        while diff > self.min_diff:
            self.theta_values = new_theta.copy()
            new_theta, cost_diff = self.cost_function()
            if self.first_theta is None:
                self.first_theta = new_theta.copy()
            # diff = abs(sum(new_theta) - sum(self.theta_values))
            diff = abs(sum(cost_diff))
            if self.debug:
                print(new_theta)
                print("Difference", diff)
        self.theta_values = new_theta
        return self.theta_values

    def cost_function(self):
        new_theta = []
        j_diff = []
        for j in range(len(self.theta_values)):
            theta = utils.cost_function(self.theta_values, j, self.training_features, self.alpha, self.training_values)
            new_theta.append(theta)
            j_diff.append(abs(theta-self.theta_values[j]))
        return new_theta, j_diff

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



features = np.array(
    [
        [5, 1500, 2],
        [3, 5200, 3],
        [3, 7001, 6]
    ])

y = np.array([2, 3, 4]).astype(float)
alpha = 0.1
gd = GradientDescent(alpha, features, y, min_diff=0, debug=True)
gd.lrm_descent()
ne_theta = utils.normal_equation(features, y)
