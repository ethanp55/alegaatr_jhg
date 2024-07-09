import numpy as np

foo = np.array([[]])
foo2 = np.concatenate((foo, np.array([[1, 2, 3]])), axis=0)

print(foo2)
