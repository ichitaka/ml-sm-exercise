filepath_1 = './Exercise 2/dataSets/densEst1.txt'
filepath_2 = './Exercise 2/dataSets/densEst2.txt'

def read_file(filepath):
    clean_list = []
    file_list = [line.rstrip('\n') for line in open(filepath)]
    for line in file_list:
        number_1 = ''
        number_2 = ''
        first_number = False
        for symbol in line:
            if symbol == ' ':
                if first_number == False and len(number_1)>0:
                    first_number = True
                continue
            elif not first_number:
                number_1 += symbol
                continue
            elif not symbol == '\n':
                number_2 += symbol
                continue
        clean_list.append([float(number_1), float(number_2)])
    return clean_list

import numpy as np

file1 = np.array(read_file(filepath_1))
file2 = np.array(read_file(filepath_2))

len1 = len(file1)
len2 = len(file2)

print('Class probability of C1:')
prior1 = len1/(len1+len2)
print(prior1)
print('Class probability of C2:')
prior2 = len2/(len1+len2)
print(prior2)

print('Mean of Class1:')
mean_1 = np.mean(file1, 0)
print(mean_1)
print('Mean of Class2:')
mean_2 = np.mean(file2, 0)
print(mean_2)

def covariance(number_list, mean):
    size = len(number_list)
    multiplications = []
    for vector in number_list:
        a = [x - mean for x in vector]
        multiplications.append(np.matmul(np.transpose(np.matrix(a)), np.matrix(a)))
    summation = np.sum(multiplications, 0)
    output = summation * 1/size
    return output

print('Covariance of Class1:')
cov_1 = covariance(file1, mean_1)
print(cov_1)
print('Covariance of Class2:')
cov_2 = covariance(file2, mean_2)
print(cov_2)

# not sure if correct, but seems so
def multivariate_normal(x, mu, cov):
    x = np.matrix(x)
    cov = np.matrix(cov)
    return (1/(np.sqrt((2*np.pi)**2*np.linalg.det(cov))))*np.exp(-1/2*(x-mu)*np.linalg.inv(cov)*np.transpose((x-mu)))

def expected_value(data_list, distribution, mu, cov):
    values = [np.multiply(x,distribution(x, mu, cov)) for x in data_list]
    return np.multiply(np.sum(values,0), 1/len(data_list))

# this is wrong, don't sum over data since more data means a higher expected value
print('Expected Value of Class1:')
print(expected_value(file1, multivariate_normal, mean_1, cov_1))
print('Expected Value of Class2:')
print(expected_value(file2, multivariate_normal, mean_2, cov_2))

print("Biased Estimate of Covariances:")

def handmade_cov_biased(mu, datalist):
    new_data_list = []
    new_sum_0 = 0
    new_sum_1 = 0
    new_sum_2 = 0
    new_sum_3 = 0
    for data in datalist:
        new_data = [data[0]-mean_1[0], data[1]-mean_1[1]]
        new_data = [[new_data[0]**2, new_data[0]*new_data[1]],[new_data[0]*new_data[1], new_data[1]**2]]
        new_data_list.append(new_data)
    for data in new_data_list:
        new_sum_0 += data[0][0]
        new_sum_1 += data[0][1]
        new_sum_2 += data[1][0]
        new_sum_3 += data[1][1]
    new_sum_0 = new_sum_0 * 1/len(datalist)
    new_sum_1 = new_sum_1 * 1/len(datalist)
    new_sum_2 = new_sum_2 * 1/len(datalist)
    new_sum_3 = new_sum_3 * 1/len(datalist)
    return [[new_sum_0, new_sum_1],[new_sum_2, new_sum_3]]

biased_cov_1 = handmade_cov_biased(mean_1, file1)
biased_cov_2 = handmade_cov_biased(mean_2, file2)
print('class 1:' +  str(biased_cov_1))
print('class 2:' + str(biased_cov_2))

def handmade_mean(datalist):
    new_sum = [0,0]
    for x in datalist:
        new_sum = [new_sum[0]+x[0], new_sum[1]+x[1]]
    return [1/len(datalist)*new_sum[0],1/len(datalist)*new_sum[1]]
    
print('biased/unbiased estimate of mean:')
print('class1: ' + str(handmade_mean(file1)))
print('class2: ' + str(handmade_mean(file2)))

print('unbiased estimate of cov:')

def handmade_cov_unbiased(mu, datalist):
    new_data_list = []
    new_sum_0 = 0
    new_sum_1 = 0
    new_sum_2 = 0
    new_sum_3 = 0
    for data in datalist:
        new_data = [data[0]-mean_1[0], data[1]-mean_1[1]]
        new_data = [[new_data[0]**2, new_data[0]*new_data[1]],[new_data[0]*new_data[1], new_data[1]**2]]
        new_data_list.append(new_data)
    for data in new_data_list:
        new_sum_0 += data[0][0]
        new_sum_1 += data[0][1]
        new_sum_2 += data[1][0]
        new_sum_3 += data[1][1]
    new_sum_0 = new_sum_0 * 1/(len(datalist)-1)
    new_sum_1 = new_sum_1 * 1/(len(datalist)-1)
    new_sum_2 = new_sum_2 * 1/(len(datalist)-1)
    new_sum_3 = new_sum_3 * 1/(len(datalist)-1)
    return [[new_sum_0, new_sum_1],[new_sum_2, new_sum_3]]

print('class1: ' + str(handmade_cov_unbiased(mean_1, file1)))
print('class2: ' + str(handmade_cov_unbiased(mean_2, file2)))

import matplotlib.pyplot as plt

X = np.arange(-10, 7.5, 0.1)
Y = np.arange(-10, 7.5, 0.1)

xy = np.mgrid[-10:7.5:0.1, -10:7.5:0.1].reshape(2, -1).T

def plothelper(grid, mu, cov):
    output = []
    for x in grid:
        output.append(multivariate_normal(x, mu, cov))
    return output

Z1 = plothelper(xy, mean_1, biased_cov_1)
Z1 = np.reshape(Z1, [175, 175])

plt.figure()
CS = plt.contour(X, Y, Z1)

plt.scatter(file1[:,0], file1[:,1], s=5)

Z2 = plothelper(xy, mean_2, biased_cov_2)
Z2 = np.reshape(Z2, [175, 175])
CS = plt.contour(X, Y, Z2)

plt.scatter(file2[:,0], file2[:,1], s=5)
plt.axis([-10, 7.5, -10, 7.5])
plt.clabel(CS, inline=1, fontsize=10)
plt.title('data + ML estimated dist')


Z1 = np.multiply(Z1, prior1)
Z2 = np.multiply(Z2, prior2)

plt.figure()
plt.contour(X, Y, Z1)
plt.contour(X,Y,Z2)
plt.title('posteriors')
plt.show()

# def decision_boundary(grid, dist1, dist2):
#     output = []
#     for idx, x in enumerate(grid):
#         if dist1[idx] > dist2[idx]

