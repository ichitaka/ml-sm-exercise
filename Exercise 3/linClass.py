#%%% #################### * CELL * ####################

import numpy as np
import os

cwd = os.getcwd()
filepath = cwd + '/Exercise 3/ldaData.txt'

def read_file(path):
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

data = np.array(read_file(filepath))

class1 = data[:50]
class2 = data[50:50+43]
class3 = data[50+43:]

targets = np.append(np.append(np.repeat(1, 50), np.repeat(2, 43)), np.repeat(3,44))

augmented_data = []
for x in data:
    augmented_data.append([1, x[0], x[1]])
augmented_data = np.array(augmented_data)

def gaussian(x, mu, cov):
    x = np.matrix(x)
    cov = np.matrix(cov)
    return (1/(np.sqrt((2*np.pi)**2*np.linalg.det(cov))))*np.exp(-1/2*(x-mu)*np.linalg.inv(cov)*np.transpose((x-mu)))

weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(augmented_data.T, augmented_data)),augmented_data.T), targets)

#%%% #################### * CELL * ####################
mean1 = np.mean(class1, axis=0)
mean2 = np.mean(class2,axis=0)
mean3 = np.mean(class3,axis=0)
means = [mean1, mean2, mean3]
cov1 = np.cov(class1, rowvar=False)
cov2 = np.cov(class2, rowvar=False)
cov3 = np.cov(class3, rowvar=False)
covs = [cov1,cov2,cov3]


prior1 = len(class1)/len(data)
prior2 = len(class2)/len(data)
prior3 = len(class3)/len(data)
priors = [prior1, prior2, prior3]




#%%% #################### * CELL * ####################

c = np.subtract(np.multiply(1/2,
                            np.matmul(np.matmul(mean1.T,
                                                    np.linalg.inv(cov1)),
                                        mean1)),
                np.multiply(1/2,np.matmul(np.matmul(mean2.T, np.linalg.inv(cov2)), mean2)))

def classify(x):
    y = []
    for i in range(3):
        y.append(priors[i]*gaussian(x, means[i], covs[i]))
    return np.argmax(y)

for x in data:
    print(classify(x))

#%%% #################### * CELL * ####################
def augment_data(list):
    augmented_data = []
    for x in list:
        augmented_data.append([1, x[0], x[1]])
    return np.array(augmented_data)


target1 = np.append(np.repeat(1, len(class1)), np.repeat(-1, len(class2)+len(class3)))
target2 = np.append(np.append(np.repeat(-1, len(class1)), np.repeat(1, len(class2))), np.repeat(-1, len(class3)))
target3 = np.append(np.repeat(-1, len(class1)+len(class2)), np.repeat(1, len(class3)))

weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(augmented_data.T, augmented_data)),augmented_data.T), targets)

weights1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(augmented_data.T, augmented_data)),augmented_data.T), target1)
weights2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(augmented_data.T, augmented_data)),augmented_data.T), target2)
weights3 = np.matmul(np.matmul(np.linalg.inv(np.matmul(augmented_data.T, augmented_data)),augmented_data.T), target3)

weights_full = [weights1,weights2,weights3]

def classify_2(x):
    y = []
    for i in range(3):
        y.append(np.matmul(weights_full[i], [1,x[0],x[1]]))
    return np.argmax(y)

classified = [[],[],[]]
for x in data:
    classified[classify_2(x)].append(x)
classified[0] = np.array(classified[0])
classified[1] = np.array(classified[1])
classified[2] = np.array(classified[2])

#%%% #################### * CELL * ####################

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(class1[:,0], class1[:,1])
plt.scatter(class2[:,0], class2[:,1])
plt.scatter(class3[:,0], class3[:,1])

plt.figure()
plt.scatter(classified[0][:,0], classified[0][:,1])
plt.scatter(classified[1][:,0], classified[1][:,1])
plt.scatter(classified[2][:,0], classified[2][:,1])

wrong = 0
for idx, x in enumerate(data):
    if idx < 50:
        if classify_2(x) != 0:
            wrong += 1
        continue
    if idx < 50+43:
        if classify_2(x) != 1:
            wrong += 1
        continue
    if idx >= 50+43:
        if classify_2(x) != 2:
            wrong += 1
        continue