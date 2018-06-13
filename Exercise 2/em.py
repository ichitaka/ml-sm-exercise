filepath = './Exercise 2/dataSets/gmm.txt'

import gauss_MLE
import numpy as np
print('')
print('########     EM - ALGORITHM      ########')
print('')

data = gauss_MLE.read_file(filepath)


print('Mean of all Data:')
mean = np.mean(data, 0)
print(mean)

print('Maximum of Data:')
maximum = np.max(data, 0)
print(maximum)

print('Minimum of Data:')
minimum = np.min(data, 0)
print(minimum)


# currently using abs with det because of value initialization
def multivariate_normal(x, mu, cov):
    x = np.matrix(x)
    cov = np.matrix(cov)
    det_cov = np.abs(np.linalg.det(cov))
    und_sqr = (2*np.pi)**2*det_cov
    under = np.sqrt(und_sqr)
    first_half = 1/under
    second_half = np.exp(-1/2*(x-mu)*np.linalg.inv(cov)*np.transpose((x-mu)))
    return first_half*second_half

def m_step(data_list, posteriors, parameter_list):
    for idx, parameters in enumerate(parameter_list):
        sum_mu = 0
        soft_count = 0
        sum_cov = 0
        for i, x in enumerate(data_list):
            sum_mu += posteriors[i][idx]*x
            soft_count += posteriors[i][idx]
        parameters[0] = soft_count/len(data_list)
        parameters[1] = (1/soft_count)*sum_mu
        for i, x in enumerate(data_list):
            dist = np.matrix(x-parameters[1])
            dist_2 = np.matmul(np.transpose(dist), dist)
            sum_cov += np.multiply(posteriors[i][idx], dist_2)
        parameters[2] = np.multiply((1/soft_count),sum_cov)
    return parameter_list

def e_step(data_list, posteriors, parameter_list):
    for i, x in enumerate(data_list):
        norm = 0
        for param in parameter_list:
            mul_norm = multivariate_normal(x, param[1], param[2])
            norm += param[0]*mul_norm
        for idx, parameters in enumerate(parameter_list):
            posteriors[i][idx] = (parameters[0]*multivariate_normal(x, parameters[1], parameters[2]))/norm
    return posteriors

def init_values():
    output = []
    for i in range(0,4):
        temp = [0,0,0]
        for j in range(0,3):
            if j == 1:
                temp[j] = np.random.rand(1,2)
                continue
            if j == 2:
                temp[j] = np.random.rand(2,2)
                continue
            temp[j] = np.random.uniform(0,1)
        output.append(temp)
    return output
                
def likelihood(data_list, parameter_list):
    sum_data = 0
    for data in data_list:
        sum_dists = 0
        for parameters in parameter_list:
            sum_dists += parameters[0]*multivariate_normal(data, parameters[1], parameters[2])
        sum_data += np.log(sum_dists)
    return sum_data

def em_loop(data_list, it_amount):
    iterations = 0
    posteriors = []
    likelihoods = []
    for x in data_list:
        posteriors.append([0,0,0,0])
    # * index 0 is pi
    # * index 1 is mu
    # * index 2 is cov
    parameter_list = init_values()
    notfinished = True
    while notfinished:
        old_parameters = parameter_list
        new_posteriors = e_step(data_list, posteriors, parameter_list)
        posteriors = new_posteriors
        parameter_list = m_step(data_list, new_posteriors, parameter_list)
        sum1 = old_parameters[0][0]
        sum2 = parameter_list[0][0]
        diff = sum1 - sum2
        likelihoods.append(likelihood(data_list, parameter_list))
        iterations += 1
        if iterations == it_amount:
            notfinished = False
    return parameter_list, posteriors, likelihoods

parameters, posteriors, likelihoods = em_loop(data, 30)

likelihoods = np.reshape(likelihoods, [30,-1])

import matplotlib.pyplot as plt

def estimated_dist(x_list, posteriors=posteriors, parameters=parameters):
    output = []
    value = 0
    for x in x_list:
        for param in parameters:
            value += param[0]*multivariate_normal(x, param[1], param[2])
        output.append(value)
        value = 0
    return output

X = np.arange(-2, 5, 0.1)
Y = np.arange(-2, 5, 0.1)

#21 per x coordinate
xy = np.mgrid[-2:5:0.1, -2:5:0.1].reshape(2, -1).T

Z = estimated_dist(xy)
Z = np.reshape(Z, [70, 70])

plt.figure()
plt.contour(X, Y, Z)
plt.title('EM Mixed Model - Iterations:1')

itera = range(30)

plt.figure()
plt.plot(itera, likelihoods)
plt.title('Likelihoods EM 1-30 Iterations')
plt.show()
