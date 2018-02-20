import numpy as np
import matplotlib.pyplot as plt

step = 0.2
nodes_num = 100
attr_num = 84
np.random.seed(35)

def readAnimals():
    animal_num = 32
    animals_file = open('animals.dat', 'r')
    try:
        str_animals = animals_file.read()
        array_animals = str_animals.split(',')

        attr_num = len(array_animals) / animal_num
        animals = np.zeros([animal_num, int(attr_num)])
        animal_index = 0
        for i in range(len(array_animals)):
            animals[animal_index, int(i % attr_num)] = int(array_animals[i])
            if i % attr_num == attr_num - 1:
                animal_index += 1
    finally:
        animals_file.close()
    return animals


def readAnimalnames():
    animalsName_file = open('animalnames.txt', 'r')
    names = []
    try:
        while True:
            name = animalsName_file.readline()
            if not name:
                break
            names.append(name.split('\t\n')[0])
    finally:
        animalsName_file.close()
    return names


def getNeighbourSize(loop, neuro_num, cur_loop):
    size = cur_loop * ((neuro_num - 1) / (loop - 1)) + neuro_num
    return int(size)

def updateWeight(weights, max_index, neighbours, input):
    w_row, w_col = weights.shape
    for i in range(neighbours):
        if max_index - i < 0:
            neighbour_front = w_row + (max_index - i)
        else:
            neighbour_front = max_index - i
        neighbour_behind = (max_index + i) % w_row
        weights[neighbour_front, :] += step * (input - weights[neighbour_front, :])
        weights[neighbour_behind, :] += step * (input - weights[neighbour_behind, :])

    return weights

def SOM_Training(animals, weights):
    a_row, a_col = animals.shape
    w_row, w_col = weights.shape
    epoch = 20
    for i in range(epoch):
        for j in range(a_row):
            similarity = np.zeros([w_row, 1])
            for k in range(w_row):
                diff = animals[j, :] - weights[k, :]
                similarity[k] = np.sqrt(np.sum(diff ** 2))
            # max_value = similarity.max()
            max_index = similarity.argmax()
            weights = updateWeight(weights, max_index, getNeighbourSize(epoch, nodes_num, i), animals[j, :])

    return weights

def SOM_Testing(animals, weights):
    a_row, a_col = animals.shape
    w_row, w_col = weights.shape
    ret = [0 for i in range(a_row)]
    for i in range(a_row):
        similarity = np.zeros([w_row, 1])
        for j in range(w_row):
            diff = animals[i, :] - weights[j, :]
            similarity[j] = np.sqrt(np.sum(diff ** 2))
        ret[i] = similarity.argmax()

    return ret


animals = readAnimals()
animalnames = readAnimalnames()

weights = np.random.random((nodes_num, attr_num))
weights = SOM_Training(animals, weights)
ret = SOM_Testing(animals, weights)

pairs = []
for i in range(len(animalnames)):
    pairs.append({'name': animalnames[i], 'class': ret[i]})

sorted_pairs = sorted(pairs, key=lambda c: c['class'])

for i in range(len(animalnames)):
    print(sorted_pairs[i]['name'], ": ", sorted_pairs[i]['class'])

# print(animals)
# print(animalnames)
