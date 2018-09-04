from __future__ import division
import numpy


'''
    Solves the derivated cost_function of giving training set.
    The features of the training set must be minimized and normalized with
    first column of matrix to 1 as xi0 = 1
    :arg theta_vector theta values of hypothesis and cost function
    :arg j current j iteration (of theta values)
    :arg features var values of our training set
    :arg alpha learning rate
    :arg y_vector solutions of our training set
'''
def cost_function(theta_vector, j, features, alpha, y_vector):
    theta_j = theta_vector[j]
    m = len(features)
    value = 0
    for i in range(m):
        h = hypothesis(theta_vector, features[i])
        x = features[i][j]
        y = y_vector[i]
        value += ((h-y)*x)
    return theta_j-alpha*(value/m)


'''
    hypothesis for linear regression multivariate. h(theta) = theta0 + theta1*x1 + ... + thetaN*xN
    As we have x0 = 1 (always) we have two vectors of same dimension (n+1)x1. If transpose theta vector
    we can obtain hypothesis value multiplying two vectors (as matrix operation). Because we have two matrix
    of dimensions 1x(n+1) and (n+1)x1 we obtain a matrix with dimension 1x1, in other words, a scalar (hypothesis
    value for h(theta_i).
'''
def hypothesis(theta_values, feature):
    return numpy.matmul(numpy.transpose(theta_values), feature)


def normal_equation(features, y_vector):
    x = features
    xt = numpy.matrix.transpose(x)
    xt_x = numpy.matmul(xt, x)
    inverse = numpy.linalg.inv(xt_x)
    return numpy.matmul(numpy.matmul(inverse, xt), y_vector)


def feature_scaling(t):
    t = t.astype(float)
    scaling_coef = []
    for feature in t:
        max_val = feature.max()
        min_val = feature.min()
        avg_val = sum(feature)/float(len(feature))
        standard_dev = max_val-min_val
        print("Feature =", feature, "max_val =", max_val, "min_val =", min_val, "avg_val =", avg_val)

        for i in range(len(feature)):
            if standard_dev == 0:
                feature[i] = float(feature[i])/float(max_val)
            else:
                feature[i] = (float(feature[i])-avg_val)/float(standard_dev)

        scaling_coef.append([avg_val, standard_dev, max_val])
    return [t, scaling_coef]


def feature_scaling_corrector(feature, corrector):
    if corrector[1] == 0:
        return float(feature)/float(corrector[2])
    return (float(feature)-corrector[0])/float(corrector[1])

