import csv
import math
import random
import operator

'''
K-NN utilizando o Iris dataset, apenas um teste para avaliar conhecimentos nesse algoritmo de classificação
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


def load_train_and_test_dataset(filename, split):
    training_set = []
    test_set = []

    with open(filename, 'r') as csv_ds_file:
        lines = csv.reader(csv_ds_file)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(5):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])

    return training_set, test_set


def normalize_attributes(element_1):
    new_element = element_1

    max_val = max(new_element)
    min_val = min(new_element)

    for x in range(len(element_1)):
        new_element[x] = (element_1[x] - min_val)/(max_val-min_val)

    return new_element

def get_similarity(element_1, element_2):
    elm1 = normalize_attributes(element_1[:-1])
    elm2 = normalize_attributes(element_2[:-1])
    return euclidean_distance(elm1, elm2)


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
    training, test = load_train_and_test_dataset('jogadores-data.csv', 0.7)
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
    classifications = load_dataset('instancias_para_classificar.csv', 5)
    print('Dataset: ' + repr(len(dataset)))
    print('Classification set: ' + repr(len(classifications)))

    # generate predictions
    predictions = []
    k = 5

    for x in range(len(classifications)):
        neighbors = get_neighbors(dataset, classifications[x], k)
        result = get_response(neighbors)
        predictions.append((classifications[x][:-1], result))

    with open('vale-nada.txt','a') as f:
        print(predictions,file = f)

    print(predictions)


main()

