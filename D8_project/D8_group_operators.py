# define group operators for the link hilbert space in D8 in the group element basis
# basis order: {1, a, a^2, a^3, x, ax, a^2x, a^3x} = {1, -iy, -1, iy,  z, x, -z, -x}
import numpy as np
a = np.array([[0, -1], [1, 0]])
x = np.array([[1, 0], [0, -1]])


# build D matrices
def D(i):
    p = i % 4
    q = int(i > 3)
    d_temp = np.linalg.matrix_power(a, p)@np.linalg.matrix_power(x, q)
    return(d_temp)


# build theta matrices
def TL(i):
    TL_temp = np.zeros([8,8])
    for ii in range(8):
        for jj in range(8):
            if (D(ii)==D(i).T @ D(jj)).all():
                TL_temp[ii,jj] = 1
    return(TL_temp)

def TR(i):
    TR_temp = np.zeros([8, 8])
    for ii in range(8):
        for jj in range(8):
            if (D(ii) == D(jj) @ D(i).T).all():
                TR_temp[ii, jj] = 1
    return (TR_temp)



