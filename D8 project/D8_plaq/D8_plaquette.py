import numpy as np
import scipy
from numpy import ndarray
from quspin.basis import spin_basis_1d, tensor_basis, basis_int_to_python_int, spin_basis_general
from functools import reduce
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmath
#mpl.use('TkAgg')

L = 4

# define local qudit operations, if you don't specify the number of levels is automatically 4
def local_X(level_indices, n_levels: int = 4):
    # pauli X on two levels
    mat = np.zeros([n_levels, n_levels], dtype=complex)
    mat[level_indices[0], level_indices[1]] = 1
    mat[level_indices[1], level_indices[0]] = 1
    return mat


def local_Y(level_indices, n_levels: int = 4):
    # pauli Y on two levels
    mat = np.zeros([n_levels, n_levels], dtype=complex)
    mat[level_indices[0], level_indices[1]] = -1j
    mat[level_indices[1], level_indices[0]] = 1j
    return mat


def local_Z(level_indices, n_levels: int = 4):
    # pauli z on two levels
    mat = np.zeros([n_levels, n_levels], dtype=complex)
    mat[level_indices[0], level_indices[0]] = 1
    mat[level_indices[1], level_indices[1]] = -1
    return mat


def local_P(level_indices, n_levels: int = 4):
    # projector to the subspace of two levels
    mat = np.zeros([n_levels, n_levels])
    mat[level_indices[0], level_indices[0]] = 1
    mat[level_indices[1], level_indices[1]] = 1
    return mat

def local_xx():
    return local_X([0,2])+local_X([1,3])

def local_xp():
    return local_X([0,2])+local_P([1,3])

def local_px():
    return local_P([0,2])+local_X([1,3])



# define QuSpin basis objects
basis_qubits = spin_basis_general(L, S="1/2")
basis_qudits = spin_basis_general(L, S="3/2")


# build global operations from local ones:
def single_site_op(local_op: ndarray, site: int, base: spin_basis_general):
    """
    the tensor product of local_op on the site given by the variable site, and identity on all other sites.
    :param local_op: ndarray  - a matrix representation of a local operator
    :param site: int - site index
    :param base: quspin basis object
    :return: ndarray
    """
    if (site < 0) or (site > base.N - 1): # treat open BC
        return np.eye(base.Ns)
    else:
        local_dim = base.sps
        lattice_length = base.N
        # TODO: input verification: local_dim has to equal dim(local_op)
        local_op_lst = [np.eye(local_dim) for i in range(lattice_length)]
        local_op_lst[site] = local_op
        return reduce(np.kron, local_op_lst)



def X(site: int, level_indices=None):
    """
    returns a matrix corresponding to pauli X on one site between two levels given by level_indices. level indices
    are not supplied: this becomes an operation on the qubit register instead of the qudits. if site is outside the
    range, returns identify (this is helpful for boundary terms)
    """
    if level_indices is None:
        base = basis_qubits
        level_indices = [0, 1]
    else:
        base = basis_qudits
    #
    # if site < 0 or site > base.N - 1:
    #     return 0*np.eye(base.Ns)  # treat the boundary case
    # else:
    return single_site_op(local_X(level_indices, n_levels=base.sps), site, base)


def Y(site: int, level_indices=None):
    """
    returns a matrix corresponding to pauli Y on one site between two levels given by level_indices. level indices
    are not supplied: this becomes an operation on the qubit register instead of the qudits. if site is outside the
    range, returns identify (this is helpful for boundary terms)
    """
    if level_indices is None:
        base = basis_qubits
        level_indices = [0, 1]
    else:
        base = basis_qudits
    #
    # if site < 0 or site > base.N - 1:
    # #     return 0*np.eye(base.Ns)  # treat the boundary case
    # else:
    return single_site_op(local_Y(level_indices, n_levels=base.sps), site, base)


def Z(site: int, level_indices=None):
    """
    returns a matrix corresponding to pauli Z on one site between two levels given by level_indices. level indices
    are not supplied: this becomes an operation on the qubit register instead of the qudits. if site is outside the
    range, returns identify (this is helpful for boundary terms)
    """
    if level_indices is None:
        base = basis_qubits
        level_indices = [0, 1]
    else:
        base = basis_qudits
    #
    # if site < 0 or site > base.N - 1:
    #     return 0*np.eye(base.Ns)  # treat the boundary case
    # else:
    return single_site_op(local_Z(level_indices, n_levels=base.sps), site, base)


def P(site: int, level_indices=None):
    """
    returns a matrix corresponding to projector into two levels on one site between if site is outside the
    range, returns identify (this is helpful for boundary terms)
    """
    base = basis_qudits
    # if site < 0 or site > base.N - 1:
    #     return 0*np.eye(base.Ns)  # treat the boundary case
    # else:
    return single_site_op(local_P(level_indices, n_levels=base.sps), site, base)

def XX(site: int):
    base = basis_qudits
    return single_site_op(local_xx(), site, base)

def XP(site: int):
    base = basis_qudits
    return single_site_op(local_xp(), site, base)

def PX(site: int):
    base = basis_qudits
    return single_site_op(local_px(), site, base)



############################  build hamiltonian ######################################
# we build the qubit and qudit part separately.
# BEFORE we combine them (!), we have to flip the basis order in each of them, to agree with the QuSpin conventions
# this is easiest to do with a change-of-basis matrix:
U2 = np.fliplr(np.eye(basis_qubits.Ns))  # for the qubits
U4 = np.fliplr(np.eye(basis_qudits.Ns))  # for the qudits


# build electric hamiltonian: OK
def H_E(h, flip=True):
    H_E_qudits = np.zeros([basis_qudits.Ns, basis_qudits.Ns])
    for i in range(L):
        H_E_qudits = H_E_qudits - h / 2 * (X(i, [0, 2]) + X(i, [1, 3]))
    if flip:
        H_E_qudits = U4 @ H_E_qudits @ U4 

    return np.kron(np.eye(basis_qubits.Ns), H_E_qudits)  # combine qudit part and qubit part



#build mass hamiltonian: OK
I_qudits = np.eye(basis_qudits.Ns)
I_qubits = np.eye(basis_qubits.Ns)
def H_M(M, flip=True):
    qubits_parts = []
    qudits_parts = [] 

    
    H_M_qubits0 = X(0)@X(3)
    H_M_qudits0 = XP(0)@XP(3)
    qubits_parts.append(H_M_qubits0)
    qudits_parts.append(H_M_qudits0)
    
    H_M_qubits1 = X(0)@X(1)
    H_M_qudits1 = XX(0)@XP(1)
    qubits_parts.append(H_M_qubits1)
    qudits_parts.append(H_M_qudits1)
    
    H_M_qubits2 = X(1)@X(2)
    H_M_qudits2 = XX(1)@XX(2)
    qubits_parts.append(H_M_qubits2)
    qudits_parts.append(H_M_qudits2)
    
    H_M_qubits3 = X(2)@X(3)
    H_M_qudits3 = XP(2)@XX(3)
    qubits_parts.append(H_M_qubits3)
    qudits_parts.append(H_M_qudits3)
        
    H_M_qubits4 = X(0)@X(3)
    H_M_qudits4 = PX(0)@PX(3)
    qubits_parts.append(H_M_qubits4)
    qudits_parts.append(H_M_qudits4)
    
    H_M_qubits5 = X(0)@X(1)
    H_M_qudits5 = PX(1)
    qubits_parts.append(H_M_qubits5)
    qudits_parts.append(H_M_qudits5)
    
    H_M_qubits6 = X(1)@X(2)
    H_M_qudits6 = I_qudits
    qubits_parts.append(H_M_qubits6)
    qudits_parts.append(H_M_qudits6)
    
    H_M_qubits7 = X(2)@X(3)
    H_M_qudits7 = PX(2)
    qubits_parts.append(H_M_qubits7)
    qudits_parts.append(H_M_qudits7)
    
    full_HM = np.zeros([basis_qubits.Ns * basis_qudits.Ns, basis_qubits.Ns * basis_qudits.Ns])
    for i in range(8):
        if flip:
            H_qubits = U2 @ qubits_parts[i] @ U2
            H_qudits = U4 @ qudits_parts[i] @ U4
        else:
            H_qubits = qubits_parts[i]
            H_qudits = qudits_parts[i]

        full_HM = full_HM - M / 2 * np.kron(H_qubits, H_qudits)

    return full_HM



#build the interaction hamiltonian:
#The one related more to the first link:
def H1(J, flip=True):
    qubits_parts = []
    qudits_parts = []
    
    H1_qubits = -X(0)@X(3)
    H1_qudits = Y(0, [0,2])@PX(3)
    qubits_parts.append(H1_qubits)
    qudits_parts.append(H1_qudits)
    
    
    H2_qubits = -X(0)@X(1)
    H2_qudits = Y(0, [0,2])@XP(1)@XX(3)
    qubits_parts.append(H2_qubits)
    qudits_parts.append(H2_qudits)

    H3_qubits = -Z(0)@X(1)@X(3)
    H3_qudits = Y(0,[1,3])@XP(1)@PX(3)
    qubits_parts.append(H3_qubits)
    qudits_parts.append(H3_qudits)

    H4_qubits = -Z(0)
    H4_qudits = Y(0,[1,3])@XX(1)@XX(3)
    qubits_parts.append(H4_qubits)
    qudits_parts.append(H4_qudits)

    H5_qubits = -X(1)@X(3)
    H5_qudits = Y(0,[1,3])@XP(1)@PX(3)
    qubits_parts.append(H5_qubits)
    qudits_parts.append(H5_qudits)

    H6_qubits = -I_qubits
    H6_qudits = Y(0,[1,3])
    qubits_parts.append(H6_qubits)
    qudits_parts.append(H6_qudits)

    H7_qubits = Y(0)@X(1)
    H7_qudits = Z(0, [0,2])@XP(1)
    qubits_parts.append(H7_qubits)
    qudits_parts.append(H7_qudits)

    H8_qubits = Y(0)@X(3)
    H8_qudits = Z(0, [0, 2])@XX(1)@PX(3)
    qubits_parts.append(H8_qubits)
    qudits_parts.append(H8_qudits)
    
    full_H1 = np.zeros([basis_qubits.Ns * basis_qudits.Ns, basis_qubits.Ns * basis_qudits.Ns])
    for i in range(8):
        if flip:
            H_qubits = U2 @ qubits_parts[i] @ U2
            H_qudits = U4 @ qudits_parts[i] @ U4
        else:
            H_qubits = qubits_parts[i]
            H_qudits = qudits_parts[i]

        full_H1 = full_H1 + J / 2 * np.kron(H_qubits, H_qudits)

    return full_H1



#The one more related to the second link:
def H2(J, flip=True):
    qubits_parts = []
    qudits_parts = []
    
    H1_qubits = X(0)@X(1)
    H1_qudits = Y(1,[0,2])
    qubits_parts.append(H1_qubits)
    qudits_parts.append(H1_qudits)
    
    
    H2_qubits = X(1)@X(2)
    H2_qudits = XX(0)@Y(1,[0,2])@XX(2)
    qubits_parts.append(H2_qubits)
    qudits_parts.append(H2_qudits)

    H3_qubits = -Z(1)
    H3_qudits = XX(0)@Y(1, [1, 3])@XX(2)
    qubits_parts.append(H3_qubits)
    qudits_parts.append(H3_qudits)

    H4_qubits = -X(0)@Z(1)@X(2)
    H4_qudits = Y(1,[1,3])@XX(2)
    qubits_parts.append(H4_qubits)
    qudits_parts.append(H4_qudits)

    H5_qubits = -X(0)@X(2)
    H5_qudits = Y(1, [1, 3])@XX(2)
    qubits_parts.append(H5_qubits)
    qudits_parts.append(H5_qudits)

    H6_qubits = -I_qubits
    H6_qudits = Y(1, [1, 3])
    qubits_parts.append(H6_qubits)
    qudits_parts.append(H6_qudits)

    H7_qubits = -X(0)@Y(1)
    H7_qudits = Z(1, [0,2])@XX(2)
    qubits_parts.append(H7_qubits)
    qudits_parts.append(H7_qudits)

    H8_qubits = -Y(1)@X(2)
    H8_qudits =  Z(1, [0, 2])@XX(2)
    qubits_parts.append(H8_qubits)
    qudits_parts.append(H8_qudits)
    
    full_H2 = np.zeros([basis_qubits.Ns * basis_qudits.Ns, basis_qubits.Ns * basis_qudits.Ns])
    for i in range(8):
        if flip:
            H_qubits = U2 @ qubits_parts[i] @ U2
            H_qudits = U4 @ qudits_parts[i] @ U4
        else:
            H_qubits = qubits_parts[i]
            H_qudits = qudits_parts[i]

        full_H2 = full_H2 + J / 2 * np.kron(H_qubits, H_qudits)

    return full_H2



#The one related more to the third link:
def H3(J, flip=True):
    qubits_parts = []
    qudits_parts = []
    
    H1_qubits = X(2)@X(3)
    H1_qudits = XX(1)@Y(2, [0, 2])
    qubits_parts.append(H1_qubits)
    qudits_parts.append(H1_qudits)
    
    
    H2_qubits = X(1)@X(2)
    H2_qudits = Y(2, [0, 2])@XX(3)
    qubits_parts.append(H2_qubits)
    qudits_parts.append(H2_qudits)

    H3_qubits = -X(1)@Z(2)@X(3)
    H3_qudits = Y(2, [1, 3])
    qubits_parts.append(H3_qubits)
    qudits_parts.append(H3_qudits)

    H4_qubits = -Z(2)
    H4_qudits = Y(2, [1, 3])@XX(3)
    qubits_parts.append(H4_qubits)
    qudits_parts.append(H4_qudits)

    H5_qubits = -X(1)@X(3)
    H5_qudits = Y(2, [1, 3])
    qubits_parts.append(H5_qubits)
    qudits_parts.append(H5_qudits)

    H6_qubits = -I_qubits
    H6_qudits = XX(1)@Y(2, [1, 3])
    qubits_parts.append(H6_qubits)
    qudits_parts.append(H6_qudits)

    H7_qubits = -X(1)@Y(2)
    H7_qudits = Z(2, [0, 2])
    qubits_parts.append(H7_qubits)
    qudits_parts.append(H7_qudits)

    H8_qubits = -Y(2)@X(3)
    H8_qudits = Z(2, [0, 2])
    qubits_parts.append(H8_qubits)
    qudits_parts.append(H8_qudits)
    
    full_H3 = np.zeros([basis_qubits.Ns * basis_qudits.Ns, basis_qubits.Ns * basis_qudits.Ns])
    for i in range(8):
        if flip:
            H_qubits = U2 @ qubits_parts[i] @ U2
            H_qudits = U4 @ qudits_parts[i] @ U4
        else:
            H_qubits = qubits_parts[i]
            H_qudits = qudits_parts[i]

        full_H3 = full_H3 + J / 2 * np.kron(H_qubits, H_qudits)

    return full_H3



#The one related more to the fourth link:
def H4(J, flip=True):
    qubits_parts = []
    qudits_parts = []
    
    H1_qubits = -X(0)@X(3)
    H1_qudits = XP(0)@Y(3, [0, 2])
    qubits_parts.append(H1_qubits)
    qudits_parts.append(H1_qudits)
    
    
    H2_qubits = -X(2)@X(3)
    H2_qudits = XP(2)@Y(3, [0, 2])
    qubits_parts.append(H2_qubits)
    qudits_parts.append(H2_qudits)

    H3_qubits = -Z(3)
    H3_qudits = XX(2)@Y(3, [1, 3])
    qubits_parts.append(H3_qubits)
    qudits_parts.append(H3_qudits)

    H4_qubits = -X(0)@X(2)@Z(3)
    H4_qudits = XP(0)@XP(2)@Y(3, [1, 3])
    qubits_parts.append(H4_qubits)
    qudits_parts.append(H4_qudits)

    H5_qubits = -I_qubits
    H5_qudits = XX(0)@Y(3, [1, 3])
    qubits_parts.append(H5_qubits)
    qudits_parts.append(H5_qudits)

    H6_qubits = -X(0)@X(2)
    H6_qudits = XP(0)@XP(2)@Y(3, [1, 3])
    qubits_parts.append(H6_qubits)
    qudits_parts.append(H6_qudits)

    H7_qubits = X(0)@Y(3)
    H7_qudits = XP(0)@XX(2)@Z(3, [0, 2])
    qubits_parts.append(H7_qubits)
    qudits_parts.append(H7_qudits)

    H8_qubits = X(2)@Y(3)
    H8_qudits = XX(0)@XP(2)@Z(3, [0, 2])
    qubits_parts.append(H8_qubits)
    qudits_parts.append(H8_qudits)
    
    full_H4 = np.zeros([basis_qubits.Ns * basis_qudits.Ns, basis_qubits.Ns * basis_qudits.Ns])
    for i in range(8):
        if flip:
            H_qubits = U2 @ qubits_parts[i] @ U2
            H_qudits = U4 @ qudits_parts[i] @ U4
        else:
            H_qubits = qubits_parts[i]
            H_qudits = qudits_parts[i]

        full_H4 = full_H4 + J / 2 * np.kron(H_qubits, H_qudits)

    return full_H4


#build the magnetic hamiltonian:

def HB(lambdaB, flip=True):
    qubits_parts = []
    qudits_parts = []
    
    H1_qubits = I_qubits
    H1_qudits = Z(0, [0, 2])@Z(1, [0, 2])@Z(2, [0, 2])@Z(3, [0, 2])
    qubits_parts.append(H1_qubits)
    qudits_parts.append(H1_qudits)
    
    
    H2_qubits = -Z(3)
    H2_qudits = Z(0, [0, 2])@Z(1, [0, 2])@Z(2, [1, 3])@Z(3, [1, 3])
    qubits_parts.append(H2_qubits)
    qudits_parts.append(H2_qudits)

    H3_qubits = Z(1)@Z(2)
    H3_qudits = Z(0, [0, 2])@Z(1, [1, 3])@Z(2, [1, 3])@Z(3, [0, 2])
    qubits_parts.append(H3_qubits)
    qudits_parts.append(H3_qudits)

    H4_qubits = Z(1)@Z(2)@Z(3)
    H4_qudits = Z(0, [0, 2])@Z(1, [1, 3])@Z(2, [0, 2])@Z(3, [1, 3])
    qubits_parts.append(H4_qubits)
    qudits_parts.append(H4_qudits)

    H5_qubits = -Z(0)
    H5_qudits =  Z(0, [1, 3])@Z(1, [1, 3])@Z(2, [0, 2])@Z(3, [0, 2])
    qubits_parts.append(H5_qubits)
    qudits_parts.append(H5_qudits)

    H6_qubits = Z(0)@Z(3)
    H6_qudits = Z(0, [1, 3])@Z(1, [1, 3])@Z(2, [1, 3])@Z(3, [1, 3])
    qubits_parts.append(H6_qubits)
    qudits_parts.append(H6_qudits)

    H7_qubits = Z(0)@Z(1)@Z(2)
    H7_qudits = Z(0, [1, 3])@Z(1, [0, 2])@Z(2, [1, 3])@Z(3, [0, 2])
    qubits_parts.append(H7_qubits)
    qudits_parts.append(H7_qudits)

    H8_qubits = Z(0)@Z(1)@Z(2)@Z(3)
    H8_qudits = Z(0, [1, 3])@Z(1, [0, 2])@Z(2, [0, 2])@Z(3, [1, 3])
    qubits_parts.append(H8_qubits)
    qudits_parts.append(H8_qudits)
    
    H9_qubits = I_qubits
    H9_qudits = Z(0, [1, 3])@Z(1, [0, 2])@Z(2, [0, 2])@Z(3, [1, 3])
    qubits_parts.append(H9_qubits)
    qudits_parts.append(H9_qudits)
    
    H10_qubits = Z(3)
    H10_qudits = Z(0, [1, 3])@Z(1, [0, 2])@Z(2, [1, 3])@Z(3, [0, 2])
    qubits_parts.append(H10_qubits)
    qudits_parts.append(H10_qudits)
    
    H11_qubits = Z(1)@Z(2)
    H11_qudits = Z(0, [1, 3])@Z(1, [1, 3])@Z(2, [1, 3])@Z(3, [1, 3])
    qubits_parts.append(H11_qubits)
    qudits_parts.append(H11_qudits)
    
    H12_qubits = -Z(1)@Z(2)@Z(3)
    H12_qudits = Z(0, [1, 3])@Z(1, [1, 3])@Z(2, [0, 2])@Z(3, [0, 2])
    qubits_parts.append(H12_qubits)
    qudits_parts.append(H12_qudits)
    
    H13_qubits = Z(0)
    H13_qudits = Z(0, [0, 2])@Z(1, [1, 3])@Z(2, [0, 2])@Z(3, [1, 3])
    qubits_parts.append(H13_qubits)
    qudits_parts.append(H13_qudits)
    
    H14_qubits = Z(0)@Z(3)
    H14_qudits = Z(0, [0, 2])@Z(1, [1, 3])@Z(2, [1, 3])@Z(3, [0, 2])
    qubits_parts.append(H14_qubits)
    qudits_parts.append(H14_qudits)
    
    H15_qubits = -Z(0)@Z(1)@Z(2)
    H15_qudits = Z(0, [0, 2])@Z(1, [0, 2])@Z(2, [1, 3])@Z(3, [1, 3])
    qubits_parts.append(H15_qubits)
    qudits_parts.append(H15_qudits)
    
    H16_qubits = Z(0)@Z(1)@Z(2)@Z(3)
    H16_qudits = Z(0, [0, 2])@Z(1, [0, 2])@Z(2, [0, 2])@Z(3, [0, 2])
    qubits_parts.append(H16_qubits)
    qudits_parts.append(H16_qudits)
    
    full_HB = np.zeros([basis_qubits.Ns * basis_qudits.Ns, basis_qubits.Ns * basis_qudits.Ns])
    for i in range(16):
        if flip:
            H_qubits = U2 @ qubits_parts[i] @ U2
            H_qudits = U4 @ (XX(0)@XX(1)@qudits_parts[i]) @ U4
        else:
            H_qubits = qubits_parts[i]
            H_qudits = qudits_parts[i]

        full_HB = full_HB + 2*lambdaB.real * np.kron(H_qubits, H_qudits)

    return full_HB