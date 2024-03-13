from functools import reduce

import numpy as np
import D8_group_operators as go

def multi_kron(*op_lst):
    return(reduce(np.kron, op_lst))

def W0(i):
    return multi_kron(go.TL(i),np.eye(8),np.eye(8) )
def W1(i):
    return multi_kron(go.TR(i).T,go.TL(i),np.eye(8) )
def W2(i):
    return multi_kron(np.eye(8),go.TR(i).T, go.TL(i))
def W3(i):
    return multi_kron(np.eye(8),np.eye(8), go.TR(i).T)




G0 = 1/2 * (W0(3) + W0(1)) + 1/2 * (W0(5) - W0(7))
G1 = 1/2 * (W1(3) + W1(1)) - 1/2 * (W1(5) - W1(7))
G2 = 1/2 * (W2(3) + W2(1)) + 1/2 * (W2(5) - W2(7))
G3 = 1/2 * (W3(3) + W3(1)) - 1/2 * (W3(5) - W3(7))


# convert from bdbdbd to bbbddd
def comm_mat(m, n):
    # commutation matrix (code example from wikipedia):

    # determine permutation applied by K
    w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m * n)[w, :]


def change_basis(A):
    change_basis_left = multi_kron(np.eye(4), comm_mat(
        2, 16), np.eye(4)) @ multi_kron(np.eye(2), comm_mat(2, 4), np.eye(32))
    change_basis_right = multi_kron(np.eye(2), comm_mat(4, 2), np.eye(32)) @ multi_kron(np.eye(4), comm_mat(16, 2),
                                                                                            np.eye(4))
    return change_basis_left@A@change_basis_right

# flip order of levels to fit with quspin convention
U2 = np.fliplr(np.eye(2**3))  # for the qubits
U4 = np.fliplr(np.eye(4**3))  # for the qudits

UU = np.kron(U2, U4)

G0 = UU@change_basis(G0)@UU
G1 = UU@change_basis(G1)@UU
G2 = UU@change_basis(G2)@UU
G3 = UU@change_basis(G3)@UU