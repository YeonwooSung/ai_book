import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10) # 0,1,2,3,4,5,6,7,8,9
y = x**2         # 0,1,4,9,16,25,36,49,64,81
plt.plot(x,y)     # generate plot
plt.show()
