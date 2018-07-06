import csv
import math
import random
import operator
import copy

'''
K-NN utilizando o dataset de atributos de jogadores para identificar se possuem características para ser atacantes ou defensores
'''

'''
#====== INÍCIO DE FUNÇÕES PARA CALCULAR DISTÂNCIAS ENTRE VETORES
'''
def minkowski_distance(vector1, vector2, p):
    if len(vector1) != len(vector2):
        return -1
    diff = 0
    for index in range(len(vector1)):
        diff += math.pow(vector2[index] - vector1[index], p)

    distance = math.pow(diff, 1/p)
    return distance


def euclidean_distance(vector1, vector2):
    return minkowski_distance(vector1, vector2, 2)


def manhattan_distance(vector1, vector2):
    return minkowski_distance(vector1, vector2, 1)


'''
#====== FIM DAS FUNÇÕES PARA CALCULAR DISTÂNCIAS ENTRE VETORES
'''
def load_dataset(filename, quant_numbers):
    dataset = []

    with open(filename, 'r') as csv_ds_file:
        lines = csv.reader(csv_ds_file)
        aux_dataset = list(lines)

        for x in range(len(aux_dataset)):
            for y in range(quant_numbers):
                aux_dataset[x][y] = float(aux_dataset[x][y])

            dataset.append(aux_dataset[x])

    return dataset


def load_train_and_test_dataset(filename, quant_numbers, split):
    training_set = []
    test_set = []

    with open(filename, 'r') as csv_ds_file:
        lines = csv.reader(csv_ds_file)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(quant_numbers):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])

    return training_set, test_set


def get_min_max_dataset(dataset, quant_numbers):
    max_values = [0 for i in range(quant_numbers)]
    min_values = [0 for i in range(quant_numbers)]

    for i in range(quant_numbers):
        max_values[i] = max(data[i] for data in dataset)
        min_values[i] = min(data[i] for data in dataset)

    return max_values, min_values
 

def min_max_normalize_dataset(dataset, quant_numbers, max_values, min_values):
    new_dataset = copy.deepcopy(dataset)
    for i in range(len(dataset)):
        for j in range(quant_numbers):
          new_dataset[i][j] = (dataset[i][j] - min_values[j])/(max_values[j]-min_values[j])
    
    return new_dataset


def normalize_dataset(dataset, quant_numbers):
    max_values = [0 for i in range(quant_numbers)]
    min_values = [0 for i in range(quant_numbers)]

    for i in range(quant_numbers):
        max_values[i] = max(data[i] for data in dataset)
        min_values[i] = min(data[i] for data in dataset)

    for i in range(len(dataset)):
        for j in range(quant_numbers):
            dataset[i][j] = (dataset[i][j] - min_values[j])/(max_values[j]-min_values[j])

    return dataset

def normalize_attributes(element_1):
    new_element = element_1

    max_val = max(new_element)
    min_val = min(new_element)

    for x in range(len(element_1)):
        new_element[x] = (element_1[x] - min_val)/(max_val-min_val)

    return new_element

def get_similarity(element_1, element_2):
    return euclidean_distance(element_1[:-1], element_2[:-1])


def get_neighbors(training_set, test_element, k):
    distances = []
    for x in range(len(training_set)):
        dist = get_similarity(test_element, training_set[x])
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0


def training_test():
    # prepare data
    training, test = load_train_and_test_dataset('jogadores-data.csv',5, 0.7)
    training = normalize_dataset(training, 5)
    test = normalize_dataset(test, 5)
    print("+------------+")
    print("Valores de treino e teste")
    print(training)
    print(test)
    print("+------------+")
    print('Train set: ' + repr(len(training)))
    print('Test set: ' + repr(len(test)))
    # generate predictions
    predictions = []
    k = 5

    for x in range(len(test)):
        neighbors = get_neighbors(training, test[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(test[x][-1]))
    accuracy = get_accuracy(test, predictions)

    print('Accuracy: ' + repr(accuracy) + '%')


def main():
    dataset = normalize_dataset(load_dataset('jogadores-data.csv', 5),5)
    classifications = normalize_dataset(load_dataset('instancias_para_classificar.csv', 5),5)

    # generate predictions
    predictions = []
    k = 5

    for x in range(len(classifications)):
        neighbors = get_neighbors(dataset, classifications[x], k)
        result = get_response(neighbors)
        predictions.append((classifications[x][:-1], result))

    print_on_file_list(predictions,'test01.csv')

    print(predictions)


def print_on_file_list(custom_list,file_name):
    with open(file_name, 'a') as f:
        print(custom_list, file = f)


#training_test()
#main()
dataset = load_dataset('jogadores-data.csv', 5)
predict = load_dataset('instancias_para_classificar.csv', 5)

'''
NORMALIZANDO DATASET
'''
max_values, min_values = get_min_max_dataset(dataset,5)
predict = min_max_normalize_dataset(predict, 5, max_values, min_values)
dataset = min_max_normalize_dataset(dataset, 5, max_values, min_values)


predictions = []
k = 11

for x in range(len(predict)):
    neighbors = get_neighbors(dataset, predict[x], k)
    result = get_response(neighbors)
    predictions.append((predict[x][:-1], result))

for x in range(len(predictions)):
   print(predictions[x])

