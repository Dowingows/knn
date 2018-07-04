import csv
import math
import random
import operator

'''
K-NN utilizando o Iris dataset, apenas um teste para avaliar conhecimentos nesse algoritmo de classificação
'''


def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        return -1
    diff = 0
    for index in range(len(vector1)):
        diff += math.pow(vector2[index] - vector1[index], 2)

    distance = math.sqrt(diff)
    return distance


def load_dataset(filename, split):
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


def main():
    # prepare data
    training, test = load_dataset('jogadores-data.csv', 0.7)

    print('Train set: ' + repr(len(training)))
    print('Test set: ' + repr(len(test)))
    # generate predictions
    predictions = []
    k = 7
    for x in range(len(test)):
        neighbors = get_neighbors(training, test[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(test[x][-1]))
    accuracy = get_accuracy(test, predictions)

    print('Accuracy: ' + repr(accuracy) + '%')


main()

