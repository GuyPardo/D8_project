from functools import reduce
from numpy import ndarray
import numpy as np
from scipy import special
import scipy
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
from math import pi
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt
import D8_plaquette as D8
import D8_group_operators as D8_op
from quspin.basis import spin_basis_1d, tensor_basis, basis_int_to_python_int, spin_basis_general
import gc





#%%
# configuration
L = 4  # number of links
M = 1  # mass constant
J = 1  # interaction strength
h = 1  # electric term strength (lambda_E in the notes)
lambdaB = 1 #magnetic term strength (lambda_B in the notes)


H_J =  D8.H1(J, True) + D8.H2(J, True) + D8.H3(J, True) + D8.H4(J, True)


# full Hamiltonian:
H = D8.H_E(h, True) + D8.HB(lambdaB , flip=True) + D8.H_M(J , flip=True) + H_J
# H = H_E(0, True) + H_J_single_link(J,1)
# check hermiticity
print("Hermiticity check:")
print(scipy.linalg.ishermitian(H))
#%%
TL1 = D8_op.TL(1)
TL3 = D8_op.TL(3)
TL5 = D8_op.TL(5)
TL7 = D8_op.TL(7)
TR1 = D8_op.TR(1)
TR3 = D8_op.TR(3)
TR5 = D8_op.TR(5)
TR7 = D8_op.TR(7)
I_link = np.eye(8)


#Bulding the extra constraint on the different verteces(these are written in the BDBDBD basis, B = qubit and D = qudit)

#first vertex (even):
g1 = (1/2)*(np.kron(np.kron(np.kron( TL1, I_link), I_link), TL1) + np.kron(np.kron(np.kron( TL3, I_link), I_link), TL3) + np.kron(np.kron(np.kron( TL5, I_link), I_link), TL5) - np.kron(np.kron(np.kron( TL7, I_link), I_link), TL7))

#second vertex (even):
g2 = (1/2)*(np.kron(np.kron(np.kron( TR1.T, TL1), I_link), I_link) + np.kron(np.kron(np.kron( TR3.T, TL3), I_link), I_link) - np.kron(np.kron(np.kron( TR5.T, TL5), I_link), I_link) + np.kron(np.kron(np.kron( TR7.T, TL7), I_link), I_link))

#third vertex (even):
g3 = (1/2)*(np.kron(np.kron(np.kron(I_link, TR1.T), TR1.T), I_link) + np.kron(np.kron(np.kron(I_link, TR3.T), TR3.T), I_link) + np.kron(np.kron(np.kron(I_link, TR5.T), TR5.T), I_link) - np.kron(np.kron(np.kron(I_link, TR7.T), TR7.T), I_link))

#fourth vertex (odd site)

g4 = (1/2)*(np.kron(np.kron(np.kron(I_link, I_link), TL1), TR1.T) + np.kron(np.kron(np.kron(I_link, I_link), TL3), TR3.T) - np.kron(np.kron(np.kron(I_link, I_link), TL5), TR5.T) + np.kron(np.kron(np.kron(I_link, I_link), TL7), TR7.T))

#The Hamiltonian has been written in the BBBBDDDD basis and therefore we define a transformation that allows us to
#pass to this basis.

# flip order of levels to fit with the quspin convention
U2 = np.fliplr(np.eye(2**4))  # for the qubits
U4 = np.fliplr(np.eye(4**4))  # for the qudits
UU = np.kron(U2, U4)

G1 = UU@D8_op.change_basis(g1)@UU
G2 = UU@D8_op.change_basis(g2)@UU
G3 = UU@D8_op.change_basis(g3)@UU
G4 = UU@D8_op.change_basis(g4)@UU



#%%
I = np.eye(8**4)
# GG = (G1-I)@(G1-I) + (G2-I)@(G2-I) +(G3-I)@(G3-I) +(G4-I)@(G4-I)
# GG = (I-G1) + (I-G2) + (I-G3) + (I-G4)


def comm(A,B):
    return A@B-B@A

comm(H,G2).any()