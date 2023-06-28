import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)

plt.subplot(2,2,1)
plt.plot(x,x**2)

plt.subplot(2,2,2)
plt.plot(x,x*5)

plt.subplot(223)
plt.plot(x, np.sin(x))

plt.subplot(224)
plt.plot(x,np.cos(x))

plt.show()
