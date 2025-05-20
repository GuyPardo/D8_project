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




#Functions usefull for the change of basis we need:

def comm_mat(m, n):
    # commutation matrix (code example from wikipedia):

    # determine permutation applied by K
    w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m * n)[w, :]

def multi_kron(*args):
    output = np.kron(args[0], args[1])
    if len(args) > 2:
        for i in range(2, len(args)):
            output = np.kron(output, args[i])
    return output



def change_basis(A):
    change_basis_left = multi_kron(np.eye(8), comm_mat(2,64), np.eye(4))@multi_kron(np.eye(4), comm_mat(2,16), np.eye(32))@multi_kron(np.eye(2), comm_mat(2,4), np.eye(256))
    change_basis_right = multi_kron(np.eye(2), comm_mat(4,2), np.eye(256))@multi_kron(np.eye(4), comm_mat(16,2), np.eye(32))@multi_kron(np.eye(8), comm_mat(64,2), np.eye(4))

    return change_basis_left@A@change_basis_right


