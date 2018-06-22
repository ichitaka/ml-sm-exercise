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
    best_weights = get_weights(input_data, output_data, 11)
    for x in ran:
        output.append(prediction(best_weights, x, 11))
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

print(calc_y(0.5))

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

#%% #################### * CELL * ####################
    #################### * 1 c  * ####################

input_data, output_data = training[:, 0], training[:, 1]
input_data, output_data = scatter_data[:,0] ,scatter_data[:,1]

# ? This function might need a transpose
def get_phi(x, num_features):
    mu_range = np.arange(0,2,(2/num_features))
    phi = []
    for data in x:
        features = []
        features.append(1)
        for mu in mu_range:
            features.append(gaussian(data, mu, var))
        phi.append(features)
    return np.matrix(phi).T

def get_weights_2(x,y):
    Y = np.transpose(np.matrix(y))
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x
                                                    ,np.transpose(x))),
                                x),
                    Y)
number_of_features = 20

# ! weights look like real data in a plot
# ! using all data
plt.figure()
plt.scatter(input_data, output_data)
plt.plot(get_weights_2(get_phi(input_data,number_of_features), output_data))


def single_phi(x, num_features):
    mu_range = np.arange(0,2,(2/num_features))
    phi = []
    phi.append(1)
    for mu in mu_range:
        phi.append(gaussian(x, mu, var))
    return np.matrix(phi).T

def plot_helper_2(r, num_feat):
    output = []
    weights = get_weights_2(get_phi(input_data, num_feat), output_data)
    for x in r:
        value = np.matmul(np.transpose(weights), single_phi(x, num_feat))
        output.append(value.tolist()[0][0])
    return output

print_1 = plot_helper_2(plot_range, number_of_features)

plt.figure()
plt.scatter(input_data, output_data)

plt.plot(plot_range, print_1)

def root_mean_square_gauss(weights, target, num_feat):
    linrange = target[:,0]
    targets = target[:,1]
    pred = []
    pred = plot_helper_2(linrange, num_feat)
    error = []
    square_sum_weights = np.sum(weights[0][:1]**2)
    l = 0.000001
    for idx, x in enumerate(pred):
        error.append((pred[idx]-targets[idx])**2)
    return np.sqrt(np.sum(error)*1/len(targets))+l*square_sum_weights

errors = []
# ! currently done on all data
for num_feat in range(15,41):
    phi = get_phi(input_data, num_feat)
    weights = get_weights_2(phi,output_data)
    err_all = root_mean_square_gauss(weights, scatter_data, num_feat)
    errors.append(err_all)

    print(str(num_feat) + ': ' + str(err_all) + '    shape:' + str(np.shape(weights)))

print('degree of min loss: ' + str(np.argmin(errors)+15))

plt.figure()
plt.plot(errors)

#%% #################### * CELL * ####################
    #################### * 1 d  * ####################

REGULIZER = 0.000001
BETA = 1/0.0025
#ALPHA = 0.0004
ALPHA = REGULIZER*BETA
POLY_DEGREE = 11
NUM_DATA_POINTS = 15

def get_weights_MAP(data, deg, target):
    design_matrix = get_design_matrix(data, deg)
    return np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(design_matrix.T,
                                                            design_matrix),
                                                    np.multiply(REGULIZER,
                                                    np.identity(np.shape(design_matrix)[1])))),
                            design_matrix.T),
                    target)

def get_features(x, deg):
    output = []
    for exp in range(deg+1):
        output.append(x**exp)
    return np.array(output)

def get_design_matrix(data, deg):
    output = []
    for x in data:
        output.append(get_features(x, deg))
    return np.matrix(output)

def plot_helper_3(data, weights, deg):
    output = []
    for x in data:
        output.append(np.matmul(weights, get_features(x, deg)))
    return output

def plot_helper_4(training_data, data, deg):
    output = []
    design_matrix = get_design_matrix(training_data, POLY_DEGREE)
    s_n = np.linalg.inv(np.add(np.multiply(ALPHA, np.identity(np.shape(design_matrix)[1])), np.multiply(BETA, np.matmul(design_matrix.T, design_matrix))))
    for x in data:
        curr_features = get_features(x, deg)
        output.append(1/BETA+np.matmul(np.matmul(curr_features.T, s_n), curr_features))
    return output

input_data, output_data = scatter_data[:NUM_DATA_POINTS, 0], scatter_data[:NUM_DATA_POINTS, 1]


CURVE = np.reshape(plot_helper_3(plot_range, get_weights_MAP(input_data, POLY_DEGREE, output_data),POLY_DEGREE),200)
UNCERTAINTY = np.reshape(plot_helper_4(input_data, plot_range, POLY_DEGREE), 200)
PLUS_UNCERTAINTY = np.add(CURVE, UNCERTAINTY)
MINUS_UNVERTAINTY = np.subtract(CURVE, UNCERTAINTY)

plt.figure()
plt.plot(plot_range, CURVE)
#plt.plot(plot_range, PLUS_UNCERTAINTY)
plt.fill_between(plot_range, PLUS_UNCERTAINTY, MINUS_UNVERTAINTY, facecolor='lightsalmon')
axes = plt.gca()
axes.set_ylim([-2,2])
#plt.fill(plot_range,PLUS_UNCERTAINTY, plot_range, MINUS_UNVERTAINTY)
#plt.plot(plot_range, MINUS_UNVERTAINTY)

plt.scatter(input_data, np.squeeze(np.asarray(output_data)))
