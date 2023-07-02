import numpy as np
sigma = 0.01

def f(z):
    return np.exp(-z**2/(2*sigma**2)) * (-sigma**2 + z**2)/(sigma**4)/np.sqrt(2*np.pi)

#means = []
#while True:
#    means.append()
T = 10000000

import tqdm
ans = 0
for i in range(T):
    ans += np.abs(f(i)) + np.abs(f(-i)) 
    print(ans)
    
print(ans)