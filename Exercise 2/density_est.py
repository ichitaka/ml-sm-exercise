train_path = './Exercise 2/dataSets/nonParamTrain.txt'
test_path = './Exercise 2/dataSets/nonParamTest.txt'

def read_file(filepath):
    output = []
    file_list = [line.rstrip('\n') for line in open(filepath)]
    for line in file_list:
        output.append(float(line.replace(" ", "")))
    return output

train_file = read_file(train_path)
test_file = read_file(test_path)

import matplotlib.pyplot as plt
import numpy as np

# bin_width = 0.02
# plt.figure()
# plt.hist(train_file, bins=np.arange(min(train_file), max(train_file) + bin_width, bin_width))
# plt.title('Bin-width 0.02')

# bin_width = 0.5
# plt.figure()
# plt.hist(train_file, bins=np.arange(min(train_file), max(train_file) + bin_width, bin_width))
# plt.title('Bin-width 0.5')

# bin_width = 2
# plt.figure()
# plt.hist(train_file, bins=np.arange(min(train_file), max(train_file) + bin_width, bin_width))
# plt.title('Bin-width 2')

def gaussian(x, var):
    variation = (np.abs(x)/var)**2
    b_exp = (-1/2)*variation
    exp = np.exp(b_exp)
    return exp

def kernel_dens_est(x, data_list, var=0.03):
    summation = []
    for data in data_list:
        summation.append(gaussian(data-x, var))
    return 1/(var*np.sqrt(2*np.pi)*len(data_list))*np.sum(summation)

# ################ * PLOTTING * ################
def plot_helper_1(liste, sig):
    output = []
    for l in liste:
        output.append(kernel_dens_est(l, train_file, sig))
    return output

def likelihood_kde(data_list, var):
    sum_data = 0
    for data in data_list:
        sum_data += np.log(kernel_dens_est(data, data_list, var))
    return sum_data

r = np.arange(-4, 8, 0.1)
plt.figure()
plt.plot(r, plot_helper_1(r, 0.03))
plt.plot(r, plot_helper_1(r, 0.2))
plt.plot(r, plot_helper_1(r, 0.8))
plt.title('KDE - all estimates')

#train_file = np.sort(train_file)

#print('Mean of Training Data:')
#print(np.mean(np.matrix(train_file)))

k1 = 2
k2 = 8
k3 = 35

def distance(data1, data2):
    return np.abs(data1-data2)

def knn(x, training, k):
    V = k_smallest_dist(x, training, k)
    N = len(training)
    return k/(N*V)

# * returns the k-smallest distance in a set of distances
#   parameters:
#       -numbers    -list of data
#       -k          -kth distance to return
#       -x          -where to measure the distance from
def k_smallest_dist(x, numbers, k=k1):
    distances = []
    for num in numbers:
        distances.append(distance(num, x))
    distances.sort()
    return distances[k-1]

def plot_helper_2(liste, k):
    output = []
    for num in liste:
        output.append(knn(num, train_file, k))
    return output

plt.figure()

knn_r = np.arange(-4,8,0.1)
plt.plot(knn_r, plot_helper_2(knn_r, k1))
plt.plot(knn_r, plot_helper_2(knn_r, k2))
plt.plot(knn_r, plot_helper_2(knn_r, k3))
plt.title('KNN - all estimates')
plt.show()

def log_kde(data, var):
    summation = 0
    for d in data:
        summation += np.log(kernel_dens_est(d, train_file, var))
    return summation

def log_knn(data, k):
    summation = 0
    for d in data:
        summation += np.log(knn(d, train_file, k))
    return summation

print('kdes:')
print(log_kde(test_file, 0.03))
print(log_kde(test_file, 0.2))
print(log_kde(test_file, 0.8))

print(log_kde(train_file, 0.03))
print(log_kde(train_file, 0.2))
print(log_kde(train_file, 0.8))

print('knns:')
print(log_knn(test_file, 2))
print(log_knn(test_file, 8))
print(log_knn(test_file, 35))

print(log_knn(train_file, 2))
print(log_knn(train_file, 8))
print(log_knn(train_file, 35))