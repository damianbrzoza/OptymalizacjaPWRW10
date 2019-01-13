import pandas as pd
import numpy as np
import commonFunction
import random
import math
import matplotlib.pyplot
import interface


#####################################################
# load data
#####################################################


#Wybierz który plik ma być sprawdzany (1-9):
num_of_data = 4
#Wyniki, których spodziewamy się na wyjściu:
true_exit_values = [47,637,1163,1805,1137,1471,2397,5967,6369]
data1 = pd.read_csv('NEH'+str(num_of_data)+'.DAT',sep=" ",header=None)
if num_of_data != 1:
    data1 = data1.iloc[:, :-1]
#Nadanie nowych nazw dataframowi
num_of_tasks,num_of_divices = data1.shape
tasks_list = []
for i in range(num_of_tasks):
    tasks_list.append('Task '+str(i))
divices_list = []
for i in range(num_of_divices):
    divices_list.append('Divice '+str(i))

data1.columns = divices_list
data1.index = tasks_list

#####################################################
# algorithm SW
#####################################################

def random_seq(data,num_of_task,nb_mach):
    seq = list(range(0,num_of_task))
    random.shuffle(seq)
    time_of_seq = commonFunction.makespan(seq, data, nb_mach)[nb_mach - 1][len(seq)]
    return seq,time_of_seq

def sym_wyz(data,num_of_task,nb_mach):

    seq,time_of_seq = random_seq(data,num_of_task,nb_mach)
    best_seq, best_time_of_seq = seq,time_of_seq
    y = [time_of_seq]
    t = 3000
    wsp = 0.9
    for i in range(1000):
        seq2, time_of_seq2 = random_seq(data,num_of_task,nb_mach)
        time_diff = time_of_seq2-time_of_seq
        if time_diff <= 0:
            seq, time_of_seq = seq2, time_of_seq2
        else:
            if t>0.1:
                x = (random.randrange(0, 1000))/1000
                if x < math.exp(-time_diff/t):
                    seq, time_of_seq = seq2, time_of_seq2
        y.append(time_of_seq)
        t = wsp*t
        if best_time_of_seq > time_of_seq:
            best_seq, best_time_of_seq = seq, time_of_seq
    x_data = []
    for i in range(len(y)):
        x_data.append(i)
    matplotlib.pyplot.plot(x_data, y, label = 'symulator_wyzazania')
    matplotlib.pyplot.ylabel('distance')
    matplotlib.pyplot.xlabel('time[ms]')
    return best_seq,best_time_of_seq


#####################################################
# algorithm for neh
#####################################################


def sum_processing_time(index_job, data, nb_machines):
    sum_p = 0
    for i in range(nb_machines):
        sum_p += data[i][index_job]
    return sum_p

#Funkcja sortująca zbiór zadań według malejących sum pracy na urządzeniach
def order_neh(data, nb_machines, nb_jobs):
    my_seq = []
    for j in range(nb_jobs):
        my_seq.append(j)
    return sorted(my_seq, key=lambda x: sum_processing_time(x, data, nb_machines), reverse=True)


def insertion(sequence, index_position, value):
    new_seq = sequence[:]
    new_seq.insert(index_position, value)
    return new_seq


def neh(data, nb_machines, nb_jobs):
    order_seq = order_neh(data, nb_machines, nb_jobs)
    #Ustawiamy task z najwyższą sumą jako pierwszy element listy
    seq_current = [order_seq[0]]
    for i in range(1, nb_jobs):
        #Zainicjowanie wartości minimalnej jako infinity
        min_cmax = float("inf")
        for j in range(0, i + 1):
            tmp_seq = insertion(seq_current, j, order_seq[i])
            cmax_tmp = commonFunction.makespan(tmp_seq, data, nb_machines)[nb_machines - 1][len(tmp_seq)]
            #print(tmp_seq, cmax_tmp)
            if min_cmax > cmax_tmp:
                best_seq = tmp_seq
                min_cmax = cmax_tmp
        seq_current = best_seq
    return seq_current, commonFunction.makespan(seq_current, data, nb_machines)[nb_machines - 1][nb_jobs]


# run NEH
time_table = np.array(data1).transpose()


seq, cmax = neh(time_table, num_of_divices, num_of_tasks)
print("nbMachines:", num_of_divices)
print("nbJobs:", num_of_tasks)
#print("data: p_ij, the processing time of jth job on ith machine\n", time_table)
print("neh: ", seq, cmax)
#print(cmax)
interface.graphic("NEH", seq, num_of_tasks, num_of_divices, commonFunction.makespan(seq, time_table, num_of_divices), time_table)


seque,t = sym_wyz(time_table, num_of_tasks, num_of_divices)
print("symul: ", seque, t)
matplotlib.pyplot.show()