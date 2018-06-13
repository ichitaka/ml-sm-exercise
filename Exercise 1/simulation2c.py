import numpy as np
import matplotlib.pyplot as plt

prob_fail = 1/3

a = np.random.binomial(1, prob_fail, 100000)
b = np.random.binomial(1, prob_fail, 100000)
c = np.random.binomial(1, prob_fail, 100000)
d = np.random.binomial(1, prob_fail, 100000)

only_a = 0
all_others = 0

for x in range(0,100000):
    if a[x] == 1:
        if b[x] == 0:
            if c[x] == 0:
                if d[x] == 0:
                    only_a = only_a + 1
                else:
                    all_others = all_others + 1 
            else:
                    all_others = all_others + 1
        else:
                    all_others = all_others + 1
    if b[x] == 1 or c[x] == 1 or d[x] == 1:
        all_others = all_others +1

print(only_a)
print(all_others)
print(only_a/all_others)

