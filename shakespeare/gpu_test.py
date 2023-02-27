import numpy as np
import cupy as cp
import time

### Numpy and CPU
s = time.time()
x_cpu = np.ones((1000,1000, 100))
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu = cp.ones((1000, 1000, 100))
#cp.cuda.Stream.null.synchronize()
e = time.time()
print(e - s)

X = np.random.randn(1000, 1000)
Z = np.random.randn(1000, 1000)

s = time.time()
X = np.random.randn(1000, 1000)
Z = np.random.randn(1000, 1000)
c = np.dot(X.T, Z)
e = time.time()
print(e - s)

s = time.time()
X = cp.ones((1000, 1000, 50))
Z = cp.ones((1000, 1000, 50))
c = np.dot(X.T, Z)
e = time.time()
print(e - s)