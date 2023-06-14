import numpy as np

np.random.seed(321)
for i in range(3):
    print(np.random.randint(1,5))
for i in range(3):
    print(np.random.randint(5,20))
print("-----------------------------")
np.random.seed(321)
for i in range(4):
    print(np.random.randint(1,5))
for i in range(3):
    print(np.random.randint(5,20))
print("----------------------")
np.random.seed(321)
for i in range(4):
    print(np.random.randint(1,5))
for i in range(5):
    print(np.random.randint(5,20))