# 1. Create local python environment (for installing packages):     Run python3 -m venv .venv
# 2. Change Python interpreter.                                     cmd + shift + p, Python: Select Interpreter, use .venv interpreter
# 3. Install matplotlib:                                            python3 -m pip install matplotlib
# 
# Here is a test script.
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()                   # Display the plot