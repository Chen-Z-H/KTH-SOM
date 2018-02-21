import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

step = 0.2
nodegrid_size = 10
ini_neigh_size = 10
final_neigh_size = 0
epoch = 20
np.random.seed(45)

def readPM(filename):
    pm = []
    try:
        pm_file = open(filename, 'r', encoding='ISO8859')
        while True:
            str_pm = pm_file.readline()
            if not str_pm:
                break
            pm.append(str_pm.strip())
    finally:
        pm_file.close()

    return pm


def readVotes():
    # return a 349 * 31 matrix
    votes_num_per_mp = 31
    try:
        votes_file = open('votes.dat', 'r')
        str_votes = votes_file.read()
        array_votes = str_votes.split(',')
        mp_num = int(len(array_votes) / votes_num_per_mp)
        votes = np.zeros([mp_num, votes_num_per_mp])

        mp_index = 0
        for i in range(len(array_votes)):
            votes[mp_index, int(i % votes_num_per_mp)] = float(array_votes[i])
            if i % votes_num_per_mp == votes_num_per_mp - 1:
                mp_index += 1
    finally:
        votes_file.close()

    return votes


def getNeighbourSize(cur_loop):
    size = cur_loop * (final_neigh_size - ini_neigh_size / (epoch - 1)) + ini_neigh_size
    return int(size)


def updateWeight(weights, max_index, neighbours, input):
    max_y = max_index[0]
    max_x = max_index[1]

    if max_x - neighbours < 0:
        row_lower = 0
    else:
        row_lower = max_x - neighbours

    if max_y - neighbours < 0:
        col_lower = 0
    else:
        col_lower = max_y - neighbours

    if max_x + neighbours >= nodegrid_size:
        row_upper = nodegrid_size - 1
    else:
        row_upper = max_x + neighbours

    if max_y + neighbours >= nodegrid_size:
        col_upper = nodegrid_size - 1
    else:
        col_upper = max_y + neighbours

    for i in range(row_lower, row_upper + 1):
        for j in range(col_lower, col_upper + 1):
            weights[i, j, :] += step * (input - weights[i, j, :])

    return weights


def SOM_Training(votes, weights):
    v_row, v_col = votes.shape
    for i in range(epoch):
        for j in range(v_row):
            similarity = np.zeros([nodegrid_size, nodegrid_size])
            for m in range(nodegrid_size):
                for n in range(nodegrid_size):
                    diff = votes[j, :] - weights[m, n, :]
                    similarity[m, n] = np.sqrt(np.sum(diff ** 2))
            # max_value = similarity.max()
            max_index = [int(similarity.argmax() / nodegrid_size), similarity.argmax() % nodegrid_size]
            weights = updateWeight(weights, max_index, getNeighbourSize(i), votes[j, :])

    return weights


def getLeftRight(pm):
    # return 0 for left, 1 for right, -1 for no party
    party = int(pm['party'])
    if party == 1 or party == 2 or party == 6 or party == 7:
        return 1
    if party == 3 or party == 4 or party == 5:
        return 0
    if party == 0:
        return -1


def SOM_Testing(votes, weights):
    a_row, a_col = votes.shape
    ret = [0 for i in range(a_row)]
    for i in range(a_row):
        similarity = np.zeros([nodegrid_size, nodegrid_size])
        for m in range(nodegrid_size):
            for n in range(nodegrid_size):
                diff = votes[i, :] - weights[m, n, :]
                similarity[m, n] = np.sqrt(np.sum(diff ** 2))
        ret[i] = similarity.argmax()

    return ret

def plot_leftRight(sorted_pm):
    name_list = []
    for i in sorted_pm:
        if i['vote'] not in name_list:
            name_list.append(i['vote'])

    left = [0 for i in range(len(name_list))]
    right = [0 for i in range(len(name_list))]

    for i in range(len(name_list)):
        for pm in sorted_pm:
            if pm['vote'] == name_list[i] and getLeftRight(pm) == 0:
                left[i] += 1
            if pm['vote'] == name_list[i] and getLeftRight(pm) == 1:
                right[i] += 1

    for i in range(len(name_list)):
        name_list[i] = str(name_list[i])

    x_plot = list(range(len(name_list)))
    width = 0.3
    error_config = {'ecolor': '0.3'}

    plt.bar(x_plot, left, width=width, alpha=0.4, error_kw=error_config, label='left', tick_label=name_list,
            fc='r')
    for i in range(len(x_plot)):
        x_plot[i] += width
    plt.bar(x_plot, right, width=width, alpha=0.4, error_kw=error_config, label='right', tick_label=name_list,
            fc='b')
    plt.legend()
    plt.show()

def plot_party(sorted_pm, party):
    name_list = []
    for i in sorted_pm:
        if i['vote'] not in name_list:
            name_list.append(i['vote'])

    votes = [0 for i in range(len(name_list))]
    for i in range(len(name_list)):
        for pm in sorted_pm:
            if pm['party'] == party and pm['vote'] == name_list[i]:
                votes[i] += 1

    x_plot = list(range(len(name_list)))
    error_config = {'ecolor': '0.3'}
    plt.bar(x_plot, votes, width=0.5, alpha=0.4, error_kw=error_config, label=party, tick_label=name_list,
            fc='b')
    plt.legend()
    plt.show()

def plot_gender(sorted_pm):
    name_list = []
    for i in sorted_pm:
        if i['vote'] not in name_list:
            name_list.append(i['vote'])

    # print(name_list)
    votes_male = [0 for i in range(len(name_list))]
    votes_female = [0 for i in range(len(name_list))]

    for i in range(len(name_list)):
        for pm in sorted_pm:
            if pm['vote'] == name_list[i] and pm['gender'] == '1':
                votes_female[i] += 1
            if pm['vote'] == name_list[i] and pm['gender'] == '0':
                votes_male[i] += 1

    x_plot = list(range(len(name_list)))
    width = 0.3
    error_config = {'ecolor': '0.3'}

    plt.bar(x_plot, votes_male, width=width, alpha=0.4, error_kw=error_config, label='male', tick_label=name_list,
            fc='r')
    for i in range(len(x_plot)):
        x_plot[i] += width
    plt.bar(x_plot, votes_female, width=width, alpha=0.4, error_kw=error_config, label='female', tick_label=name_list,
            fc='b')
    plt.legend()
    plt.show()


def plot_district(sorted_pm, vote):
    labels = []
    for i in range(1, 30):
        labels.append(str(i))

    sizes = [0 for i in range(29)]
    for i in range(len(sizes)):
        for pm in sorted_pm:
            if pm['vote'] == vote and int(pm['district']) == (i + 1):
                sizes[i] += 1

    for i in range(len(sizes)):
        sizes[i] = sizes[i] / sum(sizes) * 100

    for i in range(len(sizes)):
        if sizes[i] == 0:
            labels[i] = ''
    explode = (0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    plt.axis('equal')
    plt.pie(sizes, labeldistance=1.1, explode=explode, labels=labels, shadow=False, startangle=90, pctdistance=0.6)
    plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
    plt.grid()
    plt.show()


party = readPM('mpparty.dat')
gender = readPM('mpsex.dat')
district = readPM('mpdistrict.dat')
names = readPM('mpnames.txt')
votes = readVotes()

# weights matrix
weights = np.random.random((nodegrid_size, nodegrid_size, 31))
weights = SOM_Training(votes, weights)
ret = SOM_Testing(votes, weights)

pm = []
for i in range(len(names)):
    pm.append({'name': names[i],
               'gender': gender[i],
               'district': district[i],
               'party': party[i],
               'vote': ret[i]})

sorted_pm = sorted(pm, key=lambda c: c['vote'])

for i in range(len(pm)):
    print(sorted_pm[i]['name'], ", ",
          sorted_pm[i]['gender'], ", ",
          sorted_pm[i]['district'], ", ",
          sorted_pm[i]['party'], ", ",
          sorted_pm[i]['vote'])


x_plot = []
y_plot = []
for i in range(len(ret)):
    x_plot.append(int(ret[i] / nodegrid_size))
    y_plot.append(ret[i] % nodegrid_size)

plt.hist2d(x_plot, y_plot, bins=10, norm=LogNorm())
plt.colorbar()
plt.show()

plot_gender(sorted_pm)
plot_leftRight(sorted_pm)

# for i in range(8):
#     plot_party(sorted_pm, str(i))

votes_unique = []
for i in sorted_pm:
    if i['vote'] not in votes_unique:
        votes_unique.append(i['vote'])

for v in votes_unique:
    plot_district(sorted_pm, v)

