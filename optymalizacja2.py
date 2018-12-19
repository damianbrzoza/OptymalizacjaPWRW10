import pandas as pd
import numpy as np
import random
import matplotlib.pyplot
import itertools as it

def min_distance_2_town(lista,data):
    def generate_random(i):
        trasa = []
        while not(len(trasa)==i):
            rand = random.randrange(len(data[0]))
            if (rand not in trasa ) and rand != lista[0] and rand != lista[1]:
                trasa.append(rand)
        trasa = [lista[0]] + trasa
        trasa.append(lista[1])
        i=0
        while not (trasa in container ) and i<15:
            random.shuffle(trasa[1:-1])
            container.append(trasa)
            i = i+1
        distance = 0
        for i in range(i):
            distance = distance + data[trasa[i], trasa[i + 1]]
        print(distance,trasa)
        return distance, trasa

    def generate_best (i):
        def count_distance(new):
            distance = 0
            for a in range(len(new)-1):
                distance = distance + data[new[a], new[a + 1]]
            return distance
        pos_towns = list(range(len(data[0])))
        pos_towns.remove(lista[0])
        pos_towns.remove(lista[1])
        perm = list(it.permutations(pos_towns,i-2))
        perm = list(map(list,perm))
        for i in range(len(perm)):
            perm[i]=[lista[0]]+perm[i]
            perm[i].append(lista[1])
        distances = list(map(count_distance, perm))
        return perm[np.argmin(distances)],np.amin(distances)

    num_town = len(data[0])
    d = []
    r = []
    container = []
    for i in range(2,num_town):
        route, distance = generate_best(i)
        d.append(distance)
        r.append(route)
    return r[np.argmin(d)],np.amin(d)


#wczytywanie danych z pliku

data = pd.read_csv('exp.csv',sep=";")
np_data = np.array(data[1:-1])
np_data_clear = np_data[:,:-2].astype(int)
route = np.array(data[-1:])
route = route[0][0:-1].astype(int)
#generowanie listy problemów
problem_list=[]
for i in range (len(route)-1):
    problem_list.append([route[i],route[i+1]])
routes = []
distance = 0
for problem in problem_list:
    r,d = min_distance_2_town(problem,np_data_clear)
    routes.append(r)
    distance = distance + d
res = [routes[0][0]]
for result in routes:
    if res[-1] != result[0]:
        res.append(result[:][0])
    else:
        res.append(result[1:][0])
res.append(routes[-1][-1])
print(res)
print(distance)