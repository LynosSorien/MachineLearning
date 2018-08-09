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


def hypothesis(theta_values, feature):
    h = 0
    for i in range(len(theta_values)):
        theta = theta_values[i]
        f = feature[i]
        h += theta * f

    return h


def normal_equation(features, y_vector):
    x = features
    xt = numpy.matrix.transpose(x)
    xt_x = numpy.matmul(xt, x)
    inverse = numpy.linalg.inv(xt_x)
    return numpy.matmul(numpy.matmul(inverse, xt), y_vector)

