from random import random
from math import exp
import json
import matplotlib.pyplot as plt
import logging
from os.path import exists

logging.getLogger('matplotlib.font_manager').disabled = True
# Create and configure logger
logging.basicConfig(filename="Activity.log", format='%(asctime)s %(message)s', filemode='a')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
logger.info("the program ClassifierNN has started")


def download_data():
    if exists("Config.py"):
        from Config import training_results_file_name, testing_results_file_name, train_set_file_name, test_set_file_name, \
             train_set_name, test_set_name, weight_values, test_results, correct_answers,error_values ,number_iterations 

        if  exists(training_results_file_name):
            training_results_file = training_results_file_name
        else:
            logger.error("training_results_file_name does not exist")
            logger.info("the program has finished running")
            exit()
        if exists(testing_results_file_name):
            testing_results_file = testing_results_file_name
        else:
            logger.error("testing_results_file_name does not exist")
            logger.info("the program has finished running")
            exit()
        if exists(train_set_file_name):
            train_set_file = train_set_file_name
        else:
            logger.error("train_set_file_name does not exist")
            logger.info("the program has finished running")
            exit()
        if train_set_name != '':
            train_set_name = train_set_name
        else:
            logger.error("train_set_name does not exist")
            logger.info("the program has finished running")
            exit()
        if test_set_name != '':
            test_set_name = test_set_name
        else:
            logger.error("test_set_name does not exist")
            logger.info("the program has finished running")
            exit()   
        if weight_values != '':
            weight_values = weight_values
        else:
            logger.error("weight_values does not exist")
            logger.info("the program has finished running")
            exit()
        if test_results != '':
            test_results = test_results
        else:
            logger.error("test_results does not exist")
            logger.info("the program has finished running")
            exit()
        if correct_answers != '':
            correct_answers = correct_answers
        else:
            logger.error("correct_answers does not exist")
            logger.info("the program has finished running")
            exit()
        if error_values != '':
            error_values = error_values
        else:
            logger.error("error_values does not exist")
            logger.info("the program has finished running")
            exit()

        if number_iterations  != '':
            number_iterations  = number_iterations 
        else:
            logger.error("number_iterations  does not exist")
            logger.info("the program has finished running")
            exit()
        
        
        training_results_file = open(training_results_file_name, "r")
        training_results = json.load(training_results_file)
        training_results_file.close()

        testing_results_file = open(testing_results_file_name, "r")
        testing_results = json.load(testing_results_file)
        testing_results_file.close()

        train_set_file = open(train_set_file, "r")
        data_3 = json.load(train_set_file)
        train_dataset = data_3[train_set_name]
        train_set_file.close()
        return training_results, testing_results, train_dataset, training_results_file_name, testing_results_file_name, \
             test_set_file_name, test_set_name, weight_values, test_results, correct_answers,error_values ,number_iterations 
    else:
        logger.error("File Config.py does not exist")
        logger.info("the program has finished running")
        exit()

training_results, testing_results, train_dataset, training_results_file_name, testing_results_file_name, test_set_file_name, \
     test_set_name, weight_values, test_results, correct_answers, error_values ,number_iterations  = download_data()

w = {
    'w1': [random()-0.5 for _ in range(7)],
    'w2': [random()-0.5 for _ in range(7)],
    'w3': [random()-0.5 for _ in range(7)],
    'w4': [random()-0.5 for _ in range(7)],
    'w5': [random()-0.5 for _ in range(7)],

    'w6': [random()-0.5 for _ in range(5)],
    'w7': [random()-0.5 for _ in range(5)],
    'w8': [random()-0.5 for _ in range(5)],


    'w10': [random()-0.5],
    'w20': [random()-0.5],
    'w30': [random()-0.5],
    'w40': [random()-0.5],
    'w50': [random()-0.5],

    'w60': [random()-0.5],
    'w70': [random()-0.5],
    'w80': [random()-0.5]
}

v = {
    'v1': [],
    'v2': [],
    'v3': [],
    'v4': [],
    'v5': []
}

y = {
    'y1': [],
    'y2': [],
    'y3': []
}
all_gradients = {}

error = {
    'error': []
}

result = {
    'scores': []
}

def activate(weights, inputs):
    s = 0
    for i in range(len(weights)):
        s += weights[i] * inputs[i]
    return s


def sigmoid(s, beta):
    return 1.0 / (1.0 + exp(-s * beta))


def forward_propagate(w, row, beta):
    x = row
    b = [1]
    v['v1'] = activate(w['w1'], x[:-1])
    v['v1'] += activate(w['w10'], b)
    v['v1'] = sigmoid(v['v1'], beta)

    v['v2'] = activate(w['w2'], x[:-1])
    v['v2'] += activate(w['w20'], b)
    v['v2'] = sigmoid(v['v2'], beta)

    v['v3'] = activate(w['w3'], x[:-1])
    v['v3'] += activate(w['w30'], b)
    v['v3'] = sigmoid(v['v3'], beta)

    v['v4'] = activate(w['w4'], x[:-1])
    v['v4'] += activate(w['w40'], b)
    v['v4'] = sigmoid(v['v4'], beta)

    v['v5'] = activate(w['w5'], x[:-1])
    v['v5'] += activate(w['w50'], b)
    v['v5'] = sigmoid(v['v5'], beta)

    y['y1'] = activate(w['w6'], [v['v1'], v['v2'], v['v3'],v['v4'], v['v5']])
    y['y1'] += activate(w['w60'], b)
    y['y1'] = sigmoid(y['y1'], beta)

    y['y2'] = activate(w['w7'],[v['v1'], v['v2'], v['v3'], v['v4'], v['v5']])
    y['y2'] += activate(w['w70'], b)
    y['y2'] = sigmoid(y['y2'], beta)

    y['y3'] = activate(w['w8'],[v['v1'], v['v2'], v['v3'], v['v4'], v['v5']])
    y['y3'] += activate(w['w80'], b)
    y['y3'] = sigmoid(y['y3'], beta)

    y_ = [y['y1'], y['y2'], y['y3']]
    v_ = [v['v1'],v['v2'],v['v3'],v['v4'],v['v5']]
    return v_, y_


def sigmoid_derivative(f_s, beta):
    return beta * f_s * (1.0 - f_s)


def backward_propagate(d, beta, x, error, iterations,v,y):
    # computing gradients for the hidden-output layer
    iterations.append((iterations[-1] + 1))
    training_results[number_iterations] = iterations[-1] 
    e = 0.0
    for j in range(len(y)):
        e += (d[j]-y[j]) ** 2
        gradient_vy_list = []
        gradient_vy_bias = []
        for k in v:
            gradient_vy_list.append(-(d[j]-y[j])*sigmoid_derivative(y[j], beta) * k)
        gradient_vy_bias.append(-(d[j]-y[j])*sigmoid_derivative(y[j], beta) * 1)
        all_gradients[f'g{j+6}'] = gradient_vy_list
        all_gradients[f'g{j*10+60}'] = gradient_vy_bias
    e = 0.5 * e
    e = round(e, 4)

    error['error'].append(e)
    training_results[error_values] = error['error']

    # hidden-output layer weight
    weights_v_y = [w['w6'], w['w7'], w['w8']]

    # counting gradients for input-hidden layer
    for i in range(len(v)):
        gradient_xv_list = []
        gradient_xv_bias_list = []
        for j in range(len(x)):
            gradient_xv = 0
            for m in range(len(y)):
                gradient_xv += -(d[m]-y[m])*sigmoid_derivative(y[m], beta) * weights_v_y[m][i]
            gradient_xv = gradient_xv*sigmoid_derivative(v[i], beta)*x[j]
            gradient_xv_bias = gradient_xv*sigmoid_derivative(v[i], beta)*1
            gradient_xv_list.append(gradient_xv)
            gradient_xv_bias_list.append(gradient_xv_bias)
        all_gradients[f'g{i + 1}'] = gradient_xv_list
        all_gradients[f'g{i * 10 + 10}'] = gradient_xv_bias_list


def update_weight(w, gamma, all_gradients):
    # lists for updating input-hidden weight
    weights_updat_vy = [w['w6'], w['w7'], w['w8']]
    weights_updat_bias_vy = [w['w60'], w['w70'], w['w80']]

    gradients_vy = [all_gradients['g6'], all_gradients['g7'], all_gradients['g8']]
    gradients_bias_vy = [all_gradients['g60'], all_gradients['g70'], all_gradients['g80']]

    # update of hidden-output weights
    for i in range(len(weights_updat_vy)):
        for j in range(len(weights_updat_vy[i])):
            weights_updat_vy[i][j] = weights_updat_vy[i][j] - gamma * gradients_vy[i][j]
            weights_updat_vy[i][j] = round(weights_updat_vy[i][j],4)

    # update of hidden-output bias weights
    for i in range(len(weights_updat_bias_vy)):
        for j in range(len(weights_updat_bias_vy[i])):
            weights_updat_bias_vy[i][j] = weights_updat_bias_vy[i][j] - gamma * gradients_bias_vy[i][j]
            weights_updat_bias_vy[i][j] = round(weights_updat_bias_vy[i][j], 4)

    # lists for updating input-hidden weight
    weights_updat_xv = [w['w1'], w['w2'], w['w3'], w['w4'], w['w5']]
    weights_updat_bias_xv = [w['w10'], w['w20'], w['w30'], w['w40'], w['w50']]

    gradients_xv = [all_gradients['g1'], all_gradients['g2'], all_gradients['g3'], all_gradients['g4'], all_gradients['g5']]
    gradients_bias_xv = [all_gradients['g10'], all_gradients['g20'], all_gradients['g30'], all_gradients['g40'], all_gradients['g50']]

    # input-hidden weight update
    for i in range(len(weights_updat_xv)):
        for j in range(len(weights_updat_xv[i])):
            weights_updat_xv[i][j] = weights_updat_xv[i][j] - gamma * gradients_xv[i][j]
            weights_updat_xv[i][j] = round(weights_updat_xv[i][j],4)

    # input-hidden weights update bias
    for i in range(len(weights_updat_bias_xv)):
        for j in range(len(weights_updat_bias_xv[i])):
            weights_updat_bias_xv[i][j] = weights_updat_bias_xv[i][j] - gamma * gradients_bias_xv[i][j]
            weights_updat_bias_xv[i][j] = round(weights_updat_bias_xv[i][j],4)

def train_network(train_data, gamma, beta, e, iterations, number_epochs, value_of_difference):
    flag = True
    list_of_error_in_epoch = []
    while flag:
        list_y = []
        list_d = []
        for row in train_data:
            v, y = forward_propagate(w, row, beta)
            max_out = y.index(max(y))
            y_2 = [0, 0, 0]
            y_2[max_out] = 1
            y_ = y_2
            list_y.append(y_)

            d = row[-1]
            list_d.append(d)
            backward_propagate(d, beta, row[:-1], e, iterations, v, y)
            update_weight(w, gamma, all_gradients)
        print(f'Current epoch {int(iterations[-1] / 168)}')
        scores_of_error = 0
        if len(list_y) >= len(train_dataset):
            for i in range(len(list_y[-168:])):
                if list_y[-168 + i] != list_d[-168 + i]:
                    scores_of_error += 1
        print('scores_of_error', scores_of_error)
        list_of_error_in_epoch.append(scores_of_error)

        significant_change = [0]
        if len(list_of_error_in_epoch) >= number_epochs:
            for i in range(len(list_of_error_in_epoch[:-1])):
                if abs(list_of_error_in_epoch[i] - list_of_error_in_epoch[i+1]) >= value_of_difference:
                    significant_change.append(1)
            if sum(significant_change) == 0:
                training_results[weight_values] = w
                flag = False
                break
            else:
                del list_of_error_in_epoch[0]
 

def save_data():
    training_results_save = open(training_results_file_name, "w")
    json.dump(training_results, training_results_save)
    training_results_save.close()


def testing_neural_network(w, row, beta, s):
    d = row[-1]
    v, y = forward_propagate(w, row, beta)
    max_out = y.index(max(y))
    y_2 = [0, 0, 0]
    y_2[max_out] = 1

    d_3 = 0
    y_3 = 0
    if y_2 == [1,0,0]:
        y_3 = 'Kama'
    elif y_2 == [0,1,0]:
        y_3 = 'Rosa'
    elif y_2 == [0,0,1]:
        y_3 = 'Canadian'
    if d == [1,0,0]:
        d_3 = 'Kama'
    elif d == [0,1,0]:
        d_3 = 'Rosa'
    elif d == [0,0,1]:
        d_3 = 'Canadian'
    print('Expected output:', d_3)
    print('Neural network output: ', y_3)
    y = y_2
    if y == d:
        result['scores'].append(1)
        s.append(s[-1]+1)
        print('Classified correctly')
    else:
        result['scores'].append(0)
        print('Classified incorrectly')


def test_NN():
    file_4 = open(test_set_file_name, "r")
    data_4 = json.load(file_4)
    test_dataset = data_4[test_set_name]
    file_4.close()


    score = [0]

    for i in range(len(test_dataset)):
        print('Set number: ', i+1)
        testing_neural_network(training_results[weight_values], test_dataset[i], beta, score)
    print('Number of correctly classified sets: ',score[-1])

    testing_results[test_results] = result['scores']
    testing_results[correct_answers] = score[-1]

 
def save_results():
    training_results_save = open(training_results_file_name, "w")
    json.dump(training_results,  training_results_save)
    training_results_save.close()


    testing_results_save = open(testing_results_file_name, "w")
    json.dump(testing_results, testing_results_save)
    testing_results_save.close()

if __name__=="__main__":
    iterations = [0]
    gamma = 0.7
    beta = 0.5
    number_epochs = 10
    value_of_difference = 5

    train_network(train_dataset, gamma, beta, error, iterations, number_epochs, value_of_difference)
    save_data()
    save_results()
    test_NN()

    iterations_for_plot = iterations[1:]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_for_plot, error['error'])
    plt.grid(True)
    plt.xlabel("The number of iterations")
    plt.ylabel("Error value e")
    plt.title(f"The error value for each iteration \n for Beta = {beta} and Gamma = {gamma}")
    plt.savefig("Chart.jpg", dpi=164)
    plt.show()

    logger.info("the program ClassifierNN has finished running")
