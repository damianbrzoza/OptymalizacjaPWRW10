import numpy


# calculate the c_ij table.
def makespan(my_seq, p_ij, nbm):
    c_ij = numpy.zeros((nbm, len(my_seq) + 1))

    for j in range(1, len(my_seq) + 1):
        c_ij[0][j] = c_ij[0][j - 1] + p_ij[0][my_seq[j - 1]]
    for i in range(1, nbm):
        for j in range(1, len(my_seq) + 1):
            c_ij[i][j] = max(c_ij[i - 1][j], c_ij[i][j - 1]) + p_ij[i][my_seq[j - 1]]
    return c_ij
