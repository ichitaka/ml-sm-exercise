#%%%
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

#%%
# TODO do list operation
def root_mean_square(poly, target):
    linrange = target[:,0]
    targets = target[:,1]
    pred = []
    for x in linrange:
        pred.append(np.polyval(poly, x))
    error = []
    for idx, x in enumerate(pred):
        error.append((pred[idx]-targets[idx])**2)
    return np.sqrt(np.sum(error)*1/len(target))

input_data, output_data = training[:, 0], training[:, 1]

polyrange = range(1,22)

errors_test = []
errors_train = []
for x in polyrange:
    weights = np.polyfit(input_data, output_data, x+1)
    err_test = root_mean_square(weights, test)
    err_train = root_mean_square(weights, training)
    errors_test.append(err_test)
    errors_train.append(err_train)
    print(str(x) + ': test - ' + str(err_test) + ' train - ' + str(err_train))
print('degree of min loss: ' + str(np.argmin(errors_test)+1))

plt.figure()
plt.plot(polyrange, errors_test, color='blue')
plt.plot(polyrange, errors_train, color='red')
plt.show()

#%%
best_weights = np.polyfit(input_data, output_data, 19)

def plot_helper(ran):
    output = []
    for x in ran:
        output.append(np.polyval(best_weights, x))
    return output

scatter_data = np.array(data)

plot_range = np.arange(0, 2, 0.01)

plt.figure()
plt.scatter(scatter_data[:,0],scatter_data[:,1])
plt.plot(plot_range, plot_helper(plot_range))

plt.show()
