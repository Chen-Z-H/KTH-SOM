import numpy as np
import matplotlib.pyplot as plt

step = 0.2
nodes_num = 10
epoch = 20
np.random.seed(40)

def readCoordinates():
    coordinates = []
    cities_file = open('cities.dat', 'r')
    try:
        str_coordinates = cities_file.read()
        array_coordinates = str_coordinates.split(';\n')
        for i in range(len(array_coordinates)):
            co_pairs = array_coordinates[i].split(',')
            if len(co_pairs) == 2:
                coordinates.append([float(co_pairs[0].strip()), float(co_pairs[1].strip())])
    finally:
        cities_file.close()
    return coordinates


def getNeighbourSize(loop, max_size, cur_loop):
    size = cur_loop * (- max_size / (loop - 1)) + max_size
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


def SOM_Training(cities, weights):
    a_row, a_col = cities.shape
    w_row, w_col = weights.shape
    # epoch = 2
    for i in range(epoch):
        for j in range(a_row):
            similarity = np.zeros([w_row, 1])
            for k in range(w_row):
                diff = cities[j, :] - weights[k, :]
                similarity[k] = np.sqrt(np.sum(diff ** 2))
            # max_value = similarity.max()
            max_index = similarity.argmax()
            weights = updateWeight(weights, max_index, getNeighbourSize(epoch, 2, i), cities[j, :])

    return weights

def SOM_Testing(cities, weights):
    a_row, a_col = cities.shape
    w_row, w_col = weights.shape
    ret = [0 for i in range(a_row)]
    for i in range(a_row):
        similarity = np.zeros([w_row, 1])
        for j in range(w_row):
            diff = cities[i, :] - weights[j, :]
            similarity[j] = np.sqrt(np.sum(diff ** 2))
        ret[i] = similarity.argmax()

    return ret


coordinates = readCoordinates()
coordinates = np.array(coordinates)
co_row, co_col = coordinates.shape
weights = np.random.random((nodes_num, co_col))

weights = SOM_Training(coordinates, weights)
ret = SOM_Testing(coordinates, weights)
# for i in range(len(coordinates)):
#     print(coordinates[i][0], ",", coordinates[i][1])

pairs = []
for i in range(len(ret)):
    pairs.append({'co': coordinates[i, :], 'rank': ret[i]})

sorted_pairs = sorted(pairs, key=lambda c: c['rank'])

x_plot = []
y_plot = []
for i in range(len(ret)):
    x_plot.append(sorted_pairs[i]['co'][0])
    y_plot.append(sorted_pairs[i]['co'][1])
    print(sorted_pairs[i]['co'], ": ", sorted_pairs[i]['rank'])

x_plot.append(x_plot[0])
y_plot.append(y_plot[0])
plt.plot(x_plot, y_plot, 'o-')
plt.show()
