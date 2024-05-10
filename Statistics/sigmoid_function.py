import numpy as np 
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)  # Adjust the range to cover a suitable range for the sigmoid
f = 4 / (1 + np.exp(-x))  # Use np.exp() instead of math.exp() for array input

plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.show()
