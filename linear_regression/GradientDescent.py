import numpy as np
import utils as utils


class GradientDescent():
    def __init__(self, alpha, training_features, training_values, min_diff=0.0001, debug=False):
        self.alpha = alpha
        self.debug = debug
        self.min_diff = min_diff
        if min_diff is None:
            self.min_diff = 0

        if debug:
            print("Min diff: ", self.min_diff)

        self.training_features = training_features
        self.training_values = training_values
        self.theta_values = [1] * (len(self.training_values)+1)
        self.T = np.matrix.transpose(self.training_features)
        if not all(number == 1 for number in self.T[0]):
            v = [1] * (len(self.T[0]))
            self.T = np.insert(self.T, 0, np.array(v), axis=0)
            self.training_features = np.matrix.transpose(self.T)

    def lrm_descent(self):
        new_theta, cost_diff = self.cost_function()
        #diff = abs(sum(new_theta)-sum(self.theta_values))
        diff = abs(sum(cost_diff))
        while diff > self.min_diff:
            self.theta_values = new_theta.copy()
            new_theta, cost_diff = self.cost_function()
            #diff = abs(sum(new_theta) - sum(self.theta_values))
            diff = abs(sum(cost_diff))
            if self.debug:
                print(new_theta)
                print("Cost differences", cost_diff)

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
        return utils.hypothesis(self.theta_values, vector)



features = np.array([[5, 1, 2], [3, 5, 4], [3, 7, 6]])
y = [2, 3, 4]
alpha = 0.01
gd = GradientDescent(alpha, features, y, 0, debug=True)
gd.lrm_descent()
ne_theta = utils.normal_equation(features, y)
