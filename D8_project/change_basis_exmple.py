import numpy as np
import matplotlib.pyplot as plt


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


# my guess for the correct change of basis matrix for 3 links:
# basically converting BDBDBD to BBBDDD using 2 string swaps...
change_basis_left = multi_kron(np.eye(4), comm_mat(
    2,16), np.eye(4))@multi_kron(np.eye(2), comm_mat(2,4), np.eye(32))
change_basis_right = multi_kron(np.eye(2), comm_mat(4,2), np.eye(32))@multi_kron(np.eye(4), comm_mat(16,2), np.eye(4))


# test it:

pauli_x = np.array([[0,1], [1,0]])

# pauli_x on the second qubit:
X2 = multi_kron(np.eye(8),pauli_x, np.eye(32))
plt.imshow(X2)
plt.title("pauli_x on the second qubit in BDBDBD basis")
plt.show()


# convert:
X2_converted  = change_basis_left@X2@change_basis_right
plt.imshow(X2_converted)
plt.title("pauli_x on the second qubit: converted to the BBBDDD basis")
plt.show()


