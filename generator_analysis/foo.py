import numpy as np
from scipy.stats import rankdata

foo = np.array([1, 2, 3, 0, 10, 10])
print(foo.max())
print(foo[0])
print(rankdata(foo))
