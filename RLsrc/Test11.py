import numpy as np

for i in range(5):
    np.random.seed(1234)
    print("---------InnerLoop-------------")
    for j in range(10):
        print(np.random.randint(1, 100))