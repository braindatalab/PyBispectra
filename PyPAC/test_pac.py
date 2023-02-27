import numpy as np
from pac import BispectraPAC

np.random.RandomState(44)

data = np.random.rand(10, 3, 200)
sfreq = 100

bs_pac = BispectraPAC(data, sfreq)
bs_pac.compute_pac([10, 20], [15, 30])
