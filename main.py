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
        diff += math.pow((vector2[index] - vector1[index]), p)

    distance = float(diff)**(1/float(p))
    return distance


def euclidean_distance(vector1, vector2):
    return minkowski_distance(vector1, vector2, 2)


def manhattan_distance(vector1, vector2):
    return minkowski_distance(vector1, vector2, 1)


'''
#====== CARREGANDO OS DATASETS
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

'''
#====== NORMALIZAÇÃO DO DATASET
'''

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

'''
#====== FUNÇÃO DO ALGORITMO KNN
'''


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

'''
def get_response(neighbors):
    class_votes = {}

    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    #print (sorted_votes)
    return sorted_votes[0][0]
'''

'''
Usando pesos. Desta forma, os mais próximos tem mais valor de voto do que os mais distantes. 
https://www.python-course.eu/k_nearest_neighbor_classifier.php
'''
def get_response(neighbors):
    class_votes = {}
    weight  = 1
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += float(1/weight)
        else:
            class_votes[response] = float(1/weight)

        weight += 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_votes[0][0]



'''
#====== FUNÇÕES MISCELANEAS
'''


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0


def training_test():
    # prepare data
    n = random.randint(1,3)
    training = load_dataset(('treino-0{0}.csv'.format(n)), 5)
    test = load_dataset(('teste-0{0}.csv'.format(n)), 5)

    max_values, min_values = get_min_max_dataset(training, 5)
    training = min_max_normalize_dataset(training, 5, max_values, min_values)
    test = min_max_normalize_dataset(test, 5, max_values, min_values)

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
    dataset = load_dataset('jogadores-data.csv', 5)
    instances = load_dataset('instancias_para_classificar.csv', 5)

    max_values, min_values = get_min_max_dataset(dataset, 5)

    dataset = min_max_normalize_dataset(dataset, 5, max_values, min_values)
    classifications = min_max_normalize_dataset(instances, 5, max_values, min_values)

    # generate predictions
    predictions = []
    k = 7

    for x in range(len(classifications)):
        neighbors = get_neighbors(dataset, classifications[x], k)
        result = get_response(neighbors)
        predictions.append((instances[x][:-1], result))

    print_on_file_list(predictions,'report.csv')

    for data in predictions:
        print(', '.join(str(int(x)) for x in data[0]) + ', ' + str(data[1]))


def print_on_file_list(predictions,file_name):
    with open(file_name, 'w') as f:
        for data in predictions:
            print(', '.join(str(int(x)) for x in data[0])+', '+str(data[1]), file = f)


#training_test()
main()
#dataset = load_dataset('jogadores-data.csv', 5)
#predict = load_dataset('instancias_para_classificar.csv', 5)

'''
NORMALIZANDO DATASET
'''

'''
max_values, min_values = get_min_max_dataset(dataset,5)
predictions = min_max_normalize_dataset(predict, 5, max_values, min_values)
dataset = min_max_normalize_dataset(dataset, 5, max_values, min_values)

print("*-- Dataset")
for ds in dataset:
    print (ds)
print("*-- Predictions")
for predict in predictions:
    print (predict)
'''