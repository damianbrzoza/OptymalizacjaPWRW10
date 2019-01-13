import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot
import time


def generate_random_route(data):
    trasa = list(range(1,len(data[0])))
    random.shuffle(trasa)
    trasa = [0] + trasa
    trasa.append(0)
    distance = 0
    for i in range((len(trasa)-1)):
        distance = distance + data[trasa[i],trasa[i+1]]
    return distance,trasa

def random_search(data):
    iterations = 0
    best_dist,trasa = generate_random_route(data)
    y = [best_dist]
    act_time = time.time()
    while (time.time() < act_time + time_of_testing):
        best_dist1,trasa1 = generate_random_route(data)
        iterations = iterations + 1
        if best_dist1 < best_dist:
             best_dist,trasa = best_dist1,trasa1
        y.append(best_dist)
    prop = (time_of_testing * 60) / len(y)
    x_data  = []
    for i in range(len(y)):
        x_data.append(i * prop)
    matplotlib.pyplot.plot(x_data, y, label = "random_search")
    matplotlib.pyplot.ylabel('distance')
    matplotlib.pyplot.xlabel('time[ms]')
    print(iterations)
    return best_dist, trasa

def sym_wyz_swap(data):
    def swap(trasa):
        trasa2 = []
        trasa2 = trasa2 + trasa
        r = random.randrange(1, len(trasa)-2)
        r1 = r
        while r == r1:
            r1 = random.randrange(1, len(trasa) - 2)
        trasa2[r] = trasa[r1]
        trasa2[r1] = trasa[r]
        distance2 = 0
        for i in range((len(trasa) - 1)):
            distance2 = distance2 + data[trasa2[i], trasa2[i + 1]]
        return distance2, trasa2
    iterations = 0
    distance,trasa = generate_random_route(data)
    best_distance, best_route = distance, trasa
    y = [distance]
    t = 10000
    wsp = 0.99
    act_time = time.time()
    while (time.time() < act_time + time_of_testing):
        iterations = iterations + 1
        distance2, trasa2 = swap(trasa)
        dist_diff = distance2-distance
        #print("delta = " + str(dist_diff) + "distnowy: " + str(distance2) + "dist: " + str(distance))
        if dist_diff <= 0:
            distance, trasa = distance2, trasa2
        else:
            if t>0.01:
                x = (random.randrange(0, 1000))/1000
                if x < math.exp(-dist_diff/t):
                    distance, trasa = distance2, trasa2
        if best_distance > distance:
            best_distance, best_route =distance, trasa
        y.append(best_distance)
        t = wsp*t
    prop = (time_of_testing * 60) / len(y)
    x_data = []
    for i in range(len(y)):
        x_data.append(i * prop)
    matplotlib.pyplot.plot(x_data, y, label = 'symulator_wyzazania z swap')
    matplotlib.pyplot.ylabel('distance')
    matplotlib.pyplot.xlabel('time[ms]')
    return best_distance,best_route

def sym_wyz(data):
    iterations = 0
    distance,trasa = generate_random_route(data)
    best_distance, best_route = distance,trasa
    y = [distance]
    t = 400000
    wsp = 0.99
    act_time = time.time()
    while (time.time() < act_time + time_of_testing):
        iterations = iterations + 1
        distance2, trasa2 = generate_random_route(data)
        dist_diff = distance2-distance
        if dist_diff <= 0:
            distance, trasa = distance2, trasa2
        else:
            if t>0.1:
                x = (random.randrange(0, 1000))/1000
                if x < math.exp(-dist_diff/t):
                    distance, trasa = distance2, trasa2
        if best_distance > distance:
            best_distance, best_route =distance, trasa
        y.append(best_distance)
        t = wsp*t
    prop = (time_of_testing * 60) / len(y)
    x_data = []
    for i in range(len(y)):
        x_data.append(i * prop)
    matplotlib.pyplot.plot(x_data, y, label = 'symulator_wyzazania')
    matplotlib.pyplot.ylabel('distance')
    matplotlib.pyplot.xlabel('time[ms]')
    return best_distance,best_route

def genetic_algorithm (data):
    #help functions
    def start_population():
        pop_t = []
        pop_d = []
        for i in range(num_population):
            dist,trasa = generate_random_route(data)
            pop_t.append(trasa)
            pop_d.append(dist)
        return pop_t,pop_d

    def crossing(route1,route2):
        granica = random.randrange(len(data[0])-3) + 1
        random1 = random.randrange(10)
        if random1>5 :
            new_route = route1[:granica]
            i = 0
            while (len(route2)-1 != i):
                if route2[i] not in new_route:
                    new_route.append(route2[i])
                i = i+1
            new_route.append(0)
        else:
            i = 0
            new_route = route1[granica:]
            while (len(route2) - 1 != i):
                if route2[i] not in new_route:
                    new_route = [route2[i]]+new_route
                i = i + 1
            new_route = [0] + new_route
        distance = 0
        for i in range((len(new_route) - 1)):
            distance = distance + data[new_route[i], new_route[i + 1]]
        return distance,new_route

    def new_pop():
        #Stosuje metodę rankingową
        new_pop_dist = population_dist[0:2]
        new_pop_trasa = population_trasa[0:2]
        #mutacja
        for i in range(int(num_population/5)):
            di,ro = generate_random_route(data)
            new_pop_dist.append(di)
            new_pop_trasa.append(ro)
        #cross pozostałych
        while not (len(new_pop_dist) == len(population_dist)):
            c = random.randrange(int((len(data[0]))))
            d = random.randrange(int((len(data[0]))))
            dist,route = crossing(population_trasa[c],population_trasa[d])
            new_pop_dist.append(dist)
            new_pop_trasa.append(route)
        new_pop_dist, new_pop_trasa = zip(*sorted(zip(new_pop_dist, new_pop_trasa)))
        new_pop_dist = list(new_pop_dist)
        new_pop_trasa = list(new_pop_trasa)
        return new_pop_dist,new_pop_trasa

    #Set start parameters
    iterations = 0
    num_population = 10
    population_trasa,population_dist = start_population()
    iterations = iterations + num_population
    #Sortowanie
    population_dist, population_trasa = zip(*sorted(zip(population_dist, population_trasa)))
    population_dist = list(population_dist)
    population_trasa = list (population_trasa)
    y = [population_dist[0]]
    #print(population_dist)
    act_time = time.time()
    while (time.time() < act_time + time_of_testing):
        iterations = iterations + num_population
        population_dist, population_trasa = new_pop()
        y.append(population_dist[0])
    prop = (time_of_testing * 60) / len(y)
    x_data = []
    for i in range(len(y)):
        x_data.append(i * prop)
    matplotlib.pyplot.plot(x_data,y,label = 'genetic algorithm')

    return population_dist[0],population_trasa[0]


data = pd.read_csv('Exp.csv',sep=";")
np_data = np.array(data[1:-1])
np_data_clear = np_data[:,:-2].astype(int)
time_of_testing = 0.1 #Czas w sekundach testu


#Random_search
best,route = random_search(np_data_clear)
print("Random Search")
print("Best distance:" + str(best))
print("Best Route"+str(route))

#GeneticAlgorithm

best,route = genetic_algorithm(np_data_clear)
print("Genetic Algorithm")
print("Best distance:" + str(best))
print("Best Route"+str(route))
#Sym_wyz

best,route = sym_wyz(np_data_clear)
print("Symulator Wyzarzania")
print("Best distance:" + str(best))
print("Best Route"+str(route))

#Sym_wyz + swapem

best,route = sym_wyz_swap(np_data_clear)
print("Symulator Wyzarzania + swap")
print("Best distance:" + str(best))
print("Best Route"+str(route))

matplotlib.pyplot.ylim(30000, 50000)
matplotlib.pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
matplotlib.pyplot.show()


