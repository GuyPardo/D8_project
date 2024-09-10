
import D8_plaquette as D8
import numpy as np

h=1
J=1
lam_B = 1
m = 1
H = D8.H_M(m) + D8.H1(J) +D8.H2(J) + D8.H3(J) + D8.H4(J) + D8.H_E(h) + D8.HB(lam_B)

#%%
vals, vecs = np.linalg.eig(np.real(H))
