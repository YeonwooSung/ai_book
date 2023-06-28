import numpy as np

a = np.array([1,2,3,4])
b = np.array([[1,2,3,4], [5,6,7,8]])
c = np.array([1,2,3.14,4])
d = np.array([1,2,3,4], dtype=np.float64)

print(a, a.dtype, a.shape)  # --- a
print(b, b.dtype, b.shape)  # --- b
print(c, c.dtype, c.shape)  # --- c
print(d, d.dtype, d.shape)  # --- d
