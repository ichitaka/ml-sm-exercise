#%%% #################### * CELL * ####################
import os
import numpy as np
import matplotlib.pyplot as plt


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

cwd = os.getcwd()
data = read_file(cwd + '/Exercise 3/linRegData.txt')

training = np.array(data[:20])
test = np.array(data[20:])

print(training)

input_data, output_data = training[:, 0], np.matrix(training[:, 1])
print(input_data)

training_ = np.matrix(np.repeat(1,20))
print(training_)
#%% #################### * CELL * ####################
def root_mean_square(poly, target, deg):
    linrange = target[:,0]
    targets = target[:,1]
    pred = []
    pred = prediction(poly, linrange, deg)
    error = []
    square_sum_weights = np.sum(poly[0][:1]**2)
    l = 0.000001
    for idx, x in enumerate(pred):
        error.append((pred[idx]-targets[idx])**2)
    return np.sqrt(np.sum(error)*1/len(targets))+l*square_sum_weights


polyrange = range(1,22)
errors_test = []
errors_train = []

def get_x(x, deg):
    try:
        output = np.matrix(np.repeat(1, len(x)))
    except:
        output = np.matrix(1)
    for y in range(deg):
        curr_column = []
        try:
            for element in x:
                curr_column.append(element**(y+1))
        except:
            curr_column.append(x**(y+1))
        output = np.append(output, np.matrix(curr_column), axis=0)
    return np.transpose(output)

def prediction(weights, x, deg):
    X = get_x(x, deg)
    return np.matmul(X, weights)

def get_weights(x,y,deg):
    curr_x = get_x(x,deg)
    Y = np.transpose(np.matrix(y))
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(curr_x),curr_x)),np.transpose(curr_x)),Y)

for x in polyrange:
    weights_LSE = get_weights(input_data, output_data, x)
    err_test = root_mean_square(weights_LSE, test, x)
    err_train = root_mean_square(weights_LSE, training, x)
    errors_test.append(err_test)
    errors_train.append(err_train)
    print(str(x) + ': test - ' + str(err_test) + ' train - ' + str(err_train)+ '    shape:' + str(np.shape(weights_LSE)))
print('degree of min loss: ' + str(np.argmin(errors_test)+1))

plt.figure()
plt.plot(polyrange, errors_test, color='blue')
plt.plot(polyrange, errors_train, color='red')
plt.show()

#%% #################### * CELL * ####################

def plot_helper(ran):
    output = []
    best_weights = get_weights(input_data, output_data, 9)
    for x in ran:
        output.append(prediction(best_weights, x, 9))
    return np.reshape(output, np.shape(output)[0])

scatter_data = np.array(data)

plot_range = np.arange(0, 2, 0.01)

plt.figure()
plt.scatter(scatter_data[:,0],scatter_data[:,1])
plt.plot(plot_range, plot_helper(plot_range))

plt.show()

#%% #################### * CELL * ####################
    #################### * 1 b  * ####################
def gaussian(x, mu, sig):
    #return (1/np.sqrt(2*np.pi*sig**2))*np.exp((-1*(x-mu)**2)/(2*(sig**2)))
    return np.exp((-1*(x-mu)**2)/(2*(sig**2)))

all_mu = np.arange(0,2,(2/20))
var = 0.02

def calc_y(x):
    y = 0
    for mu in all_mu:
        y = y + gaussian(x, mu, var)
    return y

print(calc_y(2))

def calc_features(x):
    all_features = []
    for data in x:
        feature = []
        for mu in all_mu:
            feature.append(gaussian(data, mu, var))
        all_features.append(feature)
    return all_features

# might need further plots. btw no weights were learned.
# at x = 2 the condition of sum(g(x)) = 1 is not satisfied. 
plt.matshow(calc_features(input_data))