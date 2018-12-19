import pandas as pd
import numpy as np
import commonFunction
import interface


#####################################################
# load data
#####################################################


#Wybierz który plik ma być sprawdzany (1-9):
num_of_data = 5
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
            print(tmp_seq, cmax_tmp)
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
print("data: p_ij, the processing time of jth job on ith machine\n", time_table)
print("neh:", seq, cmax)
interface.graphic("NEH", seq, num_of_tasks, num_of_divices, commonFunction.makespan(seq, time_table, num_of_divices), time_table)