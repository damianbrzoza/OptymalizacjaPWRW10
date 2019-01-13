import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot
import time

def generate_random_route2(data,numberOfDrivers=2):
    trasa = list(range(1,len(data[0])))
    for i in range (numberOfDrivers-1):
        trasa.append(0)
    random.shuffle(trasa)
    trasa = [0] + trasa
    trasa.append(0)
    distances = np.zeros(numberOfDrivers)
    j = -1
    for i in range((len(trasa)-1)):
        if trasa[i] == 0:
            j = j+1
        distances[j] = distances[j]+ data[trasa[i],trasa[i+1]]
    distances = distances
    return distances,trasa

def sym_wyz(data):
    iterations = 0
    distance,trasa = generate_random_route2(data)
    best_distance,best_trasa =  distance,trasa
    y = [np.amax(distance)]
    t = 3000
    wsp = 0.80
    act_time = time.time()
    while (time.time() < act_time + time_of_testing):
        iterations = iterations + 1
        distance2, trasa2 = generate_random_route2(data)
        dist_diff = np.amax(distance2)-np.amax(distance)
        if dist_diff <= 0:
            distance, trasa = distance2, trasa2
        else:
            if t>0.1:
                x = (random.randrange(0, 1000))/1000
                if x < math.exp(-dist_diff/t):
                    distance, trasa = distance2, trasa2
        y.append(np.amax(distance))
        t = wsp*t
        if np.amax(best_distance)>np.amax(distance):
            best_distance, best_trasa = distance, trasa
    prop = (time_of_testing * 60) / len(y)
    x_data = []
    for i in range(len(y)):
        x_data.append(i * prop)
    matplotlib.pyplot.plot(x_data, y, label = 'symulator_wyzazania')
    matplotlib.pyplot.ylabel('distance')
    matplotlib.pyplot.xlabel('time[ms]')
    print("Liczba iteracji:" + str(iterations))
    return best_distance,best_trasa


data = pd.read_csv('Exp.csv',sep=";")
np_data = np.array(data[1:-1])
np_data_clear = np_data[:,:-2].astype(int)
time_of_testing = 0.2 #Czas w sekundach testu


dist,tr = generate_random_route2(np_data_clear,2)
dist,tr = sym_wyz(np_data_clear)

print(dist)
print(np.amax(dist))
print(tr)

matplotlib.pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
matplotlib.pyplot.show()
