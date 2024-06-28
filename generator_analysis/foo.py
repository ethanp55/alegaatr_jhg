import numpy as np

foo = np.array([[0, 1, 2]])
print(foo)
print(len(foo.shape))
print([foo])

for row in [foo]:
    print(row)
